#!/bin/bash
set -x

script_name=$(basename $0 .sh)
base_dir="/data02/wenhao/jl/ckpt/pythia_410m/tldr/Llama3-8B_RM_iter_dpo"
mkdir -p $base_dir
ITER_LOG_PATH=null
AVAILABLE_GPUS="4,5,6,7"

TRAINING_ITERS=4
BEST_OF_N=2
ROLLOUT_BATCH_SIZE=90000
TEMPERATURE=1

POLICY_MODEL_PATH="/data02/wenhao/jl/ckpt/pythia_410m/tldr/offline_dpo_1_epoch"
REWARD_MODEL_PATH="/data02/wenhao/jl/ckpt/rm/rm-tldr-Meta-Llama-3-8B-Instruct"
DATASET_PATH="when2rl/tldr-summarisation-preferences_reformatted"
REF_MODEL_PATH=$POLICY_MODEL_PATH

checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}

export PATH=$HOME/.local/bin/:$PATH

iter=0
if [ -f $ITER_LOG_PATH ]; then
    iter=$(cat $ITER_LOG_PATH)
fi

while (($iter < $TRAINING_ITERS)); do
    echo "Iter: $iter"
    # Create unique output paths for each iteration
    GENERATE_OUTPUT="${base_dir}/iter_${iter}_generate.jsonl"
    RM_OUTPUT="${base_dir}/iter_${iter}_rm.jsonl"
    MODEL_OUTPUT_PATH="${base_dir}/iter_${iter}_ckpt"

    # Use latest model if past first iteration
    if ((iter > 0)); then
        POLICY_MODEL_PATH="${base_dir}/iter_$((iter - 1))_ckpt"
    fi

    generate_commands="examples/batch_inference.py \
        --eval_task generate_vllm \
        --pretrain $POLICY_MODEL_PATH \
        --max_new_tokens 100 \
        --dataset $DATASET_PATH \
        --dataset_probs 1.0 \
        --temperature $TEMPERATURE \
        --tp_size 4 \
        --best_of_n $BEST_OF_N \
        --max_num_seqs 128 \
        --iter $iter \
        --rollout_batch_size $ROLLOUT_BATCH_SIZE \
        --output_path $GENERATE_OUTPUT"
    echo $generate_commands

    if [ $iter -eq 0 ] && [ -f "$GENERATE_OUTPUT" ]; then
        echo "Skipping generation as $GENERATE_OUTPUT already exists."
    else
        CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS python $generate_commands
        checkSuccess "GENERATE"
    fi

    get_rewards_commands="examples/batch_inference.py \
        --eval_task rm \
        --pretrain $REWARD_MODEL_PATH \
        --bf16 \
        --max_len 2048 \
        --dataset $GENERATE_OUTPUT \
        --dataset_probs 1.0 \
        --zero_stage 0 \
        --post_processor iter_dpo \
        --micro_batch_size 8 \
        --output_path $RM_OUTPUT"
    echo $get_rewards_commands

    if [ $iter -eq 0 ] && [ -f "$RM_OUTPUT" ]; then
        echo "Skipping generation as $RM_OUTPUT already exists."
    else
        deepspeed --include localhost:$AVAILABLE_GPUS $get_rewards_commands
        checkSuccess "RM"
    fi

    dpo_commands="examples/train_dpo.py \
        --max_len 2048 \
        --dataset $RM_OUTPUT \
        --dataset_probs 1.0 \
        --logging_steps 50 \
        --eval_steps 500 \
        --train_batch_size 128 \
        --micro_train_batch_size 8 \
        --pretrain $POLICY_MODEL_PATH \
        --ref_pretrain $REF_MODEL_PATH \
        --save_path $MODEL_OUTPUT_PATH \
        --rollout_batch_size $ROLLOUT_BATCH_SIZE \
        --zero_stage 3 \
        --beta 0.1 \
        --max_epochs 1 \
        --bf16 \
        --learning_rate 1e-5 \
        --flash_attn"
    echo $dpo_commands
    deepspeed --include localhost:$AVAILABLE_GPUS $dpo_commands
    checkSuccess "DPO"

    iter=$((iter + 1))
    if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH
    fi
done
