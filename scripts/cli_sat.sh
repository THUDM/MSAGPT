#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

MP_SIZE=1
# MODEL_NAME="MSAGPT-"
# MODEL_NAME="MSAGPT-dpo"


SEED=12345
MAX_GEN_LENGTH=128
MIN_GEN_LENGTH=0

# BeamSearchStrategy args
NUM_BEAMS=4
LENGTH_PENALTY=1.0
NO_REPEAT_NGRAM=0

# BaseStrategy args 
TEMP=0.8
TOPK=0
TOPP=0.9


PORT=19865

MODEL_ARGS="--bf16 \
            --skip-init \
            --mode finetune \
            --rotary-embedding-2d"

       #      --mode inference \ TODO: sat ds_config bug?

GENERATION_ARGS="--seed $SEED \
              --sampling-strategy BaseStrategy \
              --max-gen-length $MAX_GEN_LENGTH \
              --min-gen-length $MIN_GEN_LENGTH \
              --num-beams $NUM_BEAMS \
              --length-penalty $LENGTH_PENALTY \
              --no-repeat-ngram-size $NO_REPEAT_NGRAM \
              --multiline_stream \
              --temperature $TEMP \
              --top_k $TOPK \
              --top_p $TOPP 
"
# --sampling-strategy BeamSearchStrategy \
# --no-gap


OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

ARGS="${main_dir}/cli_sat.py \
       $MODEL_ARGS \
       $GENERATION_ARGS \
       $*"

run_cmd="${OPTIONS_NCCL} torchrun --nproc_per_node $MP_SIZE --master_port=$PORT ${ARGS}"
echo  ${run_cmd}
eval ${run_cmd}
set +x