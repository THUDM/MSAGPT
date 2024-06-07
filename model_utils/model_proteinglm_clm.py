import math
import copy
import torch
from torch.nn import functional as F
import torch.nn as nn
import contextlib

from sat import mpu
from sat.transformer_defaults import standard_attention, attention_fn_default
from sat.mpu.utils import split_tensor_along_last_dim, divide
from sat.mpu.layers import ColumnParallelLinear
from sat.model.base_model import BaseModel, BaseMixin
from sat.model.position_embedding import RotaryEmbedding
from sat.model.position_embedding import apply_rotary_pos_emb_index
from sat.ops import LayerNorm


class RotaryEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        fp16,
        hidden_size,
        num_attention_heads,
        model_parallel_size,
        rotary_embedding_2d=True,
    ):
        super().__init__()
        hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.rotary_embedding_2d = rotary_embedding_2d
        self.num_attention_heads_per_partition = divide(num_attention_heads, model_parallel_size)
        self.rotary_emb = RotaryEmbedding(
            # hidden_size_per_attention_head,
            hidden_size_per_attention_head // 2
            if rotary_embedding_2d
            else hidden_size_per_attention_head,
            base=10000,
            precision=torch.half if fp16 else torch.bfloat16,
            learnable=False,
            device=torch.cuda.current_device(),
        )


    def attention_forward(self, hidden_states, mask, **kw_args):
        attn = self.transformer.layers[kw_args["layer_id"]].attention
        attention_fn = attention_fn_default
        if "attention_fn" in attn.hooks:
            attention_fn = attn.hooks["attention_fn"]

        # [seq, b, 3 * hn * np]
        mixed_raw_layer = attn.query_key_value(hidden_states)

        # [seq, b, (np * 3 * hn)] --> [seq, b, np, 3 * hn]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
        # print(key_layer.shape)
        dropout_fn = attn.attention_dropout if attn.training else None
        if self.rotary_embedding_2d:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=kw_args["position_ids"].max() + 1)
            position_ids, block_position_ids = \
                kw_args["position_ids"][:, 0, :].transpose(0, 1).contiguous(), \
                kw_args["position_ids"][:, 1, :].transpose(0, 1).contiguous()
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:
            kw_args["position_ids"] = kw_args["position_ids"].transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=kw_args["position_ids"].max() + 1)
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, kw_args["position_ids"])

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)
        output = attn.dense(context_layer)

        if attn.training:
            output = attn.output_dropout(output)

        return output


class GEGLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = F.gelu

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class DeepNormWithGLUMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size, inner_hidden_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size * 2 // 3
        self.inner_hidden_size = inner_hidden_size

    def reinit(self):
        for layer in self.transformer.layers:
            del layer.mlp.dense_h_to_4h
            layer.mlp.dense_h_to_4h = ColumnParallelLinear(
                self.hidden_size,
                2 * self.inner_hidden_size,
                gather_output=False,
                bias=True,
                params_dtype=torch.half,
                module=self,
                name="dense_h_to_4h",
                skip_init=True,
            )
            del layer.mlp.activation_func
            layer.mlp.activation_func = GEGLU()

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        """
        hidden_states: [seq_len, batch, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
        """
        layer = self.transformer.layers[kw_args["layer_id"]]
        # Layer norm at the begining of the transformer layer.

        attention_input = layer.input_layernorm(hidden_states)

        # Self attention.
        attention_output = layer.attention(attention_input, mask, **kw_args)

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = layer.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = layer.mlp(mlp_input, **kw_args)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output


class SelfAttentionWithFP32SoftmaxMixin(BaseMixin):
    def __init__(self, fp16, hidden_size, num_attention_heads, model_parallel_size):
        super().__init__()
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.hidden_size_per_partition = divide(hidden_size, model_parallel_size)
        self.scale_mask_softmax = None
        self.fp16 = fp16

    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        attention_scores.masked_fill_(attention_mask, -10000.0)
        return attention_scores

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        mems=None,
        **kwargs
    ):

        mem = mems[kwargs["layer_id"]] if mems is not None else None

        # seqlen, batch, head, hidden_size
        seq_len, b, nh, hidden_size = key_layer.shape

        # stack, seqlen, b, head, hidden
        # b, seqlen, stack, head, hidden
        cache_kv = (
            torch.stack((key_layer, value_layer))
            .permute(2, 1, 0, 3, 4)
            .detach()
            .contiguous()
            .view(b, seq_len, nh * hidden_size * 2)
        )
        kwargs["output_this_layer"]["mem_kv"] = cache_kv

        if mem is not None:  # the first time, mem is None
            # might change batch_size
            # b, seqlen, stack, head, hidden -> stack, seqlen, b, head, hidden
            mem = mem.expand(b, -1, -1).reshape(b, mem.shape[1], 2, nh, hidden_size).permute(2, 1, 0, 3, 4)
            memk, memv = mem[0], mem[1]
            key_layer = torch.cat((memk, key_layer), dim=0)
            value_layer = torch.cat((memv, value_layer), dim=0)


        # check if use flash attention
        is_low_triangle = (attention_mask == ~torch.ones_like(attention_mask, dtype=torch.bool).tril()).all()
        is_full = (attention_mask is None) or (attention_mask == 0).all()
        if int(torch.__version__.split('.')[0]) >= 2 and (is_full or is_low_triangle):
            # Pytorch 2.0 attention uses very much memory if attention_mask is float, and has NaN bug if attention_mask is None.
            dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
            #[b, np, sq, hn]
            query_layer, key_layer, value_layer = query_layer.permute(1,2,0,3).contiguous(), key_layer.permute(1,2,0,3).contiguous(), value_layer.permute(1,2,0,3).contiguous()
            batch_size, num_query_heads = query_layer.shape[:2] # [b, np, s, hn]
            num_kv_heads = key_layer.shape[1] # [b, np, s, hn]
            key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *key_layer.shape[2:])
            value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *value_layer.shape[2:])

            if dropout_p > 0 and mpu.get_cuda_rng_tracker is not None:
                context = mpu.get_cuda_rng_tracker().fork()
            else:
                context = contextlib.nullcontext()

            with context:
                context_layer = torch.nn.functional.scaled_dot_product_attention(
                    query_layer, key_layer, value_layer, 
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=not is_full
                )


            #[sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (-1,)
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer

        else:
            # standard attention
            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            query_key_layer_scaling_coeff = float(kwargs["layer_id"] + 1)


            if scaling_attention_score:
                query_layer = query_layer / (math.sqrt(self.hidden_size_per_attention_head) * query_key_layer_scaling_coeff)
            # ===================================
            # Raw attention scores. [b, np, s, s]
            # ===================================
            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            matmul_result = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=torch.cuda.current_device(),
            )

            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=1.0,
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)
            
            if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
                # if auto-regressive, skip
                attention_scores.masked_fill_(attention_mask.bool(), -float("inf"))

            attention_scores = attention_scores.float()
            attention_scores = attention_scores * query_key_layer_scaling_coeff
        

            attention_probs = F.softmax(attention_scores, dim=-1)

            if self.fp16:
                attention_probs = attention_probs.half()
            else:
                attention_probs = attention_probs.bfloat16()

            if attention_dropout is not None:
                if mpu.get_cuda_rng_tracker() is not None:
                    with mpu.get_cuda_rng_tracker().fork():
                        attention_probs = attention_dropout(attention_probs)
                else:
                    attention_probs = attention_dropout(attention_probs)
                
            # =========================
            # Context layer. [sq, b, hp]
            # =========================
            
            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
                
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer



class FinalForwardMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    def final_forward(self, logits, **kw_args):
        return F.linear(logits, self.transformer.word_embeddings.weight).transpose(0, 1).contiguous()


class UntieFinalForwardMixin(BaseMixin):
    def __init__(self, hidden_size, vocab_size, untie_head_num, layernorm_epsilon=1.0e-5):
        super().__init__()

        self.lm_head = nn.ModuleList()
        for i in range(untie_head_num):
            self.lm_head.append(
                ColumnParallelLinear(
                hidden_size,
                2 * hidden_size,
                gather_output=True,
                bias=False,
                module=self,
                name=f"lm_head.{i}",
                )
             ) # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.

        self.head_layernorm = nn.ModuleList()
        for i in range(untie_head_num):
            self.head_layernorm.append(
                LayerNorm(
                hidden_size,
                eps=layernorm_epsilon
                )
            )
        self.activation_func=GEGLU()


    def final_forward(self, logits, **kwargs):
        logits = self.lm_head[1](logits)
        logits = self.activation_func(logits)
        logits = self.head_layernorm[1](logits)
        return F.linear(logits, self.transformer.word_embeddings.weight).transpose(0, 1).contiguous()


class NonePositionEmbedding(BaseMixin):
    def __init__(self):
        super().__init__()

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None


class WordEmbedding(BaseMixin):
    def __init__(self):
        super().__init__()

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        return self.transformer.word_embeddings(input_ids).transpose(0, 1)


class ProteinGLMForGeneration(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(
            args,
            transformer=transformer,
            **kwargs
        )
        self.add_mixin("glu-deepnorm", DeepNormWithGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size))
        self.add_mixin(
            "fp32-softmax",
            SelfAttentionWithFP32SoftmaxMixin(args.fp16, args.hidden_size, args.num_attention_heads, args.model_parallel_size),
        )
        if args.untie_head:
            self.add_mixin("final-forward", UntieFinalForwardMixin(args.hidden_size, args.vocab_size, args.head_num))
        else:
            self.add_mixin("final-forward", FinalForwardMixin())
        self.add_mixin("non-position-embedding", NonePositionEmbedding())
        del self.transformer.position_embeddings
        self.add_mixin("word-embedding", WordEmbedding())
        self.add_mixin(
            "rotary-embedding",
            RotaryEmbeddingMixin(
                args.fp16,
                args.hidden_size,
                args.num_attention_heads,
                args.model_parallel_size,
                args.rotary_embedding_2d
            ),
        )
        self.get_mixin("glu-deepnorm").reinit()

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ProteinGLMForGeneration', 'ProteinGLMForGeneration Configurations')
        group.add_argument('--untie-head', action='store_true', help='untie-heads')
        group.add_argument('--head-num', default=1, type=int, help='head>1')
        group.add_argument('--infer-type', default=1, type=int, help='1 for Generation')
        group.add_argument('--rotary-embedding-2d', action='store_true',
                help='If set, use 2D rotary embedding for ProtenGLM.') 
        return super().add_model_specific_args(parser)
