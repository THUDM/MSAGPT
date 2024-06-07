import os
import torch
import stat
import re
import time
import argparse
import numpy as np

from functools import partial
from typing import List, Tuple

import torch.distributed as dist
from sat.helpers import print_rank0
from sat import mpu, get_args, get_tokenizer
from sat.generation.utils import timed_name, generate_continually
from sat.generation.autoregressive_sampling import update_mems, get_masks_and_position_ids_default

from .utils import move_cursor_up, move_cursor_down


def get_masks_and_position_ids(seq, msa_len, max_gen_length, gmask=False):
    context_length = seq.shape[1]
    query_len = msa_len
    max_msa_num = (max_gen_length - 2) // query_len
    max_gen_length = max_msa_num * query_len + 2
    tokens = torch.nn.functional.pad(seq, (0, max_gen_length - context_length), mode="constant", value=-1)
    attention_mask = torch.ones((1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    # <gMASK> + <SOP>
    position_ids = np.zeros(max_gen_length, dtype=int)
    block_position_ids = np.zeros(max_gen_length, dtype=int)
    pre = 0
    for msa_idx in range(max_msa_num):
        position_ids[(1 + pre): (1 + pre + query_len)] =  np.arange(query_len, dtype = int)
        block_position_ids[(1 + pre): (1 + pre + query_len)] = msa_idx
        pre += query_len   
    position_ids = np.stack((position_ids, block_position_ids), axis=0)
    position_ids = torch.from_numpy(position_ids).to(tokens.device)
    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids



def generation_sequence(
        model,
        seqs,
        strategy,
        max_memory_length=100000,
        get_masks_and_position_ids=get_masks_and_position_ids,
        stream=False,
        mems=None,
        **kw_args
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    assert len(seqs.shape) == 2
    # building the initial tokens, attention_mask, and position_ids
    batch_size, context_length = seqs.shape
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs)
    tokens = seqs[..., :context_length]
    # initialize generation
    counter = context_length # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    num_beams = 1
    # step-by-step generation
    while counter < seqs.shape[1] - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        model.eval()
        with torch.no_grad():
            logits, *output_per_layers = model(
                tokens[:, index:],
                position_ids[..., index: counter],
                attention_mask[..., index: counter, :counter], # TODO memlen
                mems=mems,
                **kw_args
            )
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        logits = logits[:, -1]
        index = counter
        counter += 1
        logits = logits.reshape(batch_size, num_beams, -1)
        tokens = tokens.reshape(batch_size, num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size, num_beams, mems.shape[-2], mems.shape[-1])
        tokens, mems = strategy.forward(logits, tokens, mems)
        if len(tokens.shape) == 3 and num_beams == 1:
            num_beams = tokens.shape[1]
            position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, 2, -1).reshape(batch_size * num_beams, 2, -1)
            attention_mask_shape = attention_mask.shape[-3:]
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1, -1, -1).reshape(
                batch_size * num_beams, *attention_mask_shape)
        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)


def stream_generation_sequence(
        model,
        seqs,
        strategy,
        max_memory_length=100000,
        get_masks_and_position_ids=get_masks_and_position_ids,
        stream=False,
        mems=None,
        **kw_args
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    assert len(seqs.shape) == 2
    # building the initial tokens, attention_mask, and position_ids
    batch_size, context_length = seqs.shape
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs)
    tokens = seqs[..., :context_length]
    # initialize generation
    counter = context_length # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    num_beams = 1
    # step-by-step generation
    while counter < seqs.shape[1] - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        model.eval()
        with torch.no_grad():
            logits, *output_per_layers = model(
                tokens[:, index:],
                position_ids[..., index: counter],
                attention_mask[..., index: counter, :counter], # TODO memlen
                mems=mems,
                **kw_args
            )
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        logits = logits[:, -1]
        index = counter
        counter += 1
        logits = logits.reshape(batch_size, num_beams, -1)
        tokens = tokens.reshape(batch_size, num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size, num_beams, mems.shape[-2], mems.shape[-1])
        tokens, mems = strategy.forward(logits, tokens, mems, is_first=False)
        if len(tokens.shape) == 3 and num_beams == 1:
            num_beams = tokens.shape[1]
            position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, 2, -1).reshape(batch_size * num_beams, 2, -1)
            attention_mask_shape = attention_mask.shape[-3:]
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1, -1, -1).reshape(
                batch_size * num_beams, *attention_mask_shape)
        yield tokens, mems
        if strategy.is_done:
            break



def autoregressive_sampling(args, raw_text: str, model, tokenizer, strategy, stream=False) -> Tuple[List[str], List[str], List[List[str]]]:
    # add MASK
    generation_mask = "[gMASK]"
    seq = []
    msa_len = len(raw_text[0]) + 1
    seq += [tokenizer.get_command(generation_mask)] + [tokenizer.get_command("sop")]
    for each in raw_text:
        seq += tokenizer.tokenize(each) + [tokenizer.get_command('<M>')]

    output_list = [seq]
    num_output = args.num_beams if args.sampling_strategy == "BeamSearchStrategy" else 1
    seq = output_list[0]
    # detect mask position
    mask_token = tokenizer.get_command(generation_mask)
    mask_position = seq.index(mask_token)

    last_pos, answers, blanks, output_list = (
        [0] * num_output,
        ["" for _ in range(num_output)],
        [[] for _ in range(num_output)],
        []
    )
    icl_msas = len(raw_text)
    input_seq = torch.tensor(
        [seq],
        dtype = torch.long,
        device=args.device,
    )
    if args.stream_chat:
        if args.chinese:
            print(f"{'生成的MSA'.center(20, '*')}", flush=True)
        else:
            print(f"{'Virtual MSA'.center(20, '*')}", flush=True)
        output_stream = stream_generation_sequence(
            model = model,
            seqs = input_seq,
            strategy=strategy,
            get_masks_and_position_ids=partial(
                get_masks_and_position_ids,
                msa_len = msa_len,
                max_gen_length=args.max_gen_length,
                gmask=True
            )
        )  
        offset = -1   
        for tmp_res, mems in output_stream:
            if isinstance(tmp_res, torch.Tensor):  
                output = tmp_res.tolist()
            output_list = output[0]
            for i in range(len(output_list)):
                output = output_list[i].tolist() if isinstance(output_list[i], torch.Tensor) else output_list[i]
                bog = output.index(tokenizer.get_command("sop"))
                try:
                    unfinished = output.index(-1)
                except ValueError:
                    unfinished = len(output)
                output_list[i] = output[:mask_position] + output[bog + 1 : unfinished]
            for i, output in enumerate(output_list):
                if output[-1] == tokenizer.get_command("eos"):
                    output = output[:-1]
                answers[i] = tokenizer.detokenize(output)
            tmp_ret = answers[0] # only support streaming output first line.
            if mpu.get_model_parallel_rank() == 0:
                if not args.multiline_stream:
                    vit_msa = tmp_ret[offset if offset>0 else -1:]
                    print(vit_msa, end='', flush=True)
                    offset = len(tmp_ret)
                else:
                    print_len = 0
                    vit_msa = tmp_ret.split('[<M>]')[icl_msas:]
                    vit_msa = [_ for _ in vit_msa if len(_) > 0]
                    for _ in vit_msa:
                        print(_) 
                        print_len += 1
                    move_cursor_up(print_len)

        move_cursor_down(print_len)
        print('\n')
        output = strategy.finalize(tmp_res, mems)[0]
    else:      
        output, _ = generation_sequence(
            model = model,
            seqs = input_seq,
            strategy=strategy,
            get_masks_and_position_ids=partial(
                get_masks_and_position_ids,
                msa_len = msa_len,
                max_gen_length=args.max_gen_length,
                gmask=True
            )
        )
    last_pos, answers, blanks, output_list = (
        [0] * num_output,
        ["" for _ in range(num_output)],
        [[] for _ in range(num_output)],
        []
    )
    if isinstance(output, torch.Tensor):  # different strategies
        output = output.tolist()
    output = output[0]  # batch_size = 1
    output_list.extend(output)
    # clip -1s and fill back generated things into seq
    for i in range(len(output_list)):
        output = output_list[i].tolist() if isinstance(output_list[i], torch.Tensor) else output_list[i]
        try:
            unfinished = output.index(-1)
        except ValueError:
            unfinished = len(output)
        # if output[unfinished - 1] in strategy.end_tokens:
        #     unfinished -= 1
        bog = output.index(tokenizer.get_command("sop"))

        prefix = tokenizer.detokenize(output[last_pos[i] : mask_position])
        blank = tokenizer.detokenize(output[bog + 1 : unfinished])
        blanks[i].append(blank)
        last_pos[i] = mask_position + unfinished - (bog + 1)
        output_list[i] = output[:mask_position] + output[bog + 1 : unfinished]


    for i, output in enumerate(output_list):
        if output[-1] == tokenizer.get_command("eos"):
            output = output[:-1]
        answers[i] = tokenizer.detokenize(output)
    return answers


def offline_generation(args, temp, top_p, top_k, func):
    os.makedirs(args.output_path, exist_ok=True)
    with open(args.input_source, 'r', encoding="utf-8") as fin:
        inputs = fin.readlines()
    output_path = os.path.join(args.output_path, f"tmp_{temp}_p_{top_p}_k_{top_k}")
    fin = open(output_path, 'w')
    start_time = time.time()
    for line_no, raw_text in enumerate(inputs):
        if line_no % mpu.get_data_parallel_world_size() != mpu.get_data_parallel_rank():
            continue
        rk = dist.get_rank()
        raw_text = raw_text.strip()
        raw_text = raw_text.split('<M>')
        main_seq = raw_text[0]

        msa_len = len(main_seq) + 1
        icl_msas = len(raw_text)
        require_min_gen_length = msa_len * (icl_msas + 1) + 2
        if args.max_gen_length < require_min_gen_length:
            args.max_gen_length = require_min_gen_length # at least generate 1 msa.

        if mpu.get_model_parallel_rank() == 0:
            print(f'Processing No. {line_no} on model group {rk} input main seq: "{main_seq}" few-shot prompt: "{"<M>".join(raw_text[1:])}"')
        if len(raw_text) == 0:
            continue
        ret = func(raw_text)
        if mpu.get_model_parallel_rank() == 0:
            if args.print_all_beams:
                for idx, vit_msa in enumerate(ret):
                    vit_msa = vit_msa.split('[<M>]')[icl_msas:]
                    vit_msa = [_ for _ in vit_msa if len(_) > 0]
                    vit_msa_len = len(vit_msa)
                    vit_msa_str = '<M>'.join(vit_msa)   
                    print('Beam: {} #Vitural Length:{} | MSA: "{}" | (Temp, P, K)=({}, {}, {}) | Taken time {:.2f}'.format(idx, vit_msa_len, vit_msa_str, temp, top_p, top_k, time.time() - start_time), flush=True)
            else:
                vit_msa = ret[0]
                vit_msa = vit_msa.split('[<M>]')[icl_msas:]
                vit_msa = [_ for _ in vit_msa if len(_) > 0]
                vit_msa_len = len(vit_msa)
                vit_msa_str = '<M>'.join(vit_msa)       
                fin.write(f"{vit_msa_str}"+'\n')
                print('#Vitural Length:{} | MSA: "{}" | (Temp, P, K)=({}, {}, {}) | Taken time {:.2f}'.format(vit_msa_len, vit_msa_str, temp, top_p, top_k, time.time() - start_time), flush=True)
        print()
        fin.flush()
    dist.barrier()
    fin.close()


def online_generation(args, query, temp, top_p, top_k, func):
    raw_text = query.strip()
    raw_text = raw_text.split('<M>')
    main_seq = raw_text[0]
    msa_len = len(main_seq) + 1
    icl_msas = len(raw_text)
    require_min_gen_length = msa_len * (icl_msas + 1) + 2
    if args.max_gen_length < require_min_gen_length:
        args.max_gen_length = require_min_gen_length # at least generate 1 msa.
    ret = func(raw_text)
    response = []
    if mpu.get_model_parallel_rank() == 0:
        for idx, vit_msa in enumerate(ret):
            vit_msa = vit_msa.split('[<M>]')[icl_msas:]
            vit_msa = [_ for _ in vit_msa if len(_) > 0]
            response.append(vit_msa)
        return response
        

def chat_api(args, model, tokenizer, strategy, query=None): # TODO: Steam chat
    if args.input_source == 'chat':
        assert query is not None
        ret = online_generation(args, query, temp=args.temperature, top_p = args.top_p, top_k = args.top_k, func = partial(autoregressive_sampling, args, model = model, tokenizer = tokenizer, strategy = strategy))
        return ret
    else:
        assert not args.stream_chat, "Offline Generation don't support streaming output."
        offline_generation(args, temp=args.temperature, top_p = args.top_p, top_k = args.top_k, func = partial(autoregressive_sampling, args, model = model, tokenizer = tokenizer, strategy = strategy))
