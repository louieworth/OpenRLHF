#!/bin/bash
set -x

AVAILABLE_GPUS="4,5,6,7"
MODEL_PATH="/data02/wenhao/jl/ckpt/rm/rm-tldr-Meta-Llama-3-8B-Instruct"
DATSET_PATH="when2rl/tldr-summarisation-preferences_reformatted"

checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

get_rewards_commands="examples/batch_inference.py \
    --eval_task rm_acc \
    --pretrain $MODEL_PATH \
    --bf16 \
    --max_len 2048 \
    --dataset $DATSET_PATH \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --eval \
    --post_processor eval \
    --micro_batch_size 8"
echo $get_rewards_commands
deepspeed --include localhost:$AVAILABLE_GPUS $get_rewards_commands
checkSuccess "RM_ACC"