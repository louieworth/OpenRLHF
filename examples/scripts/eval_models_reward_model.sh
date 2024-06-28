#!/bin/bash
set -x

script_name=$(basename $0 .sh)


mkdir -p $eval_dir
ITER_LOG_PATH=null
AVAILABLE_GPUS="4,5,6,7"

# TRAINING_ITERS=2
BEST_OF_N=1
ROLLOUT_BATCH_SIZE=1000
TEMPERATURE=1

base_dir="/data02/wenhao/jl/ckpt/pythia_410m/tldr"
eval_dir="${base_dir}/ArmoRM_eval"
mkdir -p $eval_dir

pythia_410m/tldr/ArmoRM_iter_dpo
POLICY_1_MODEL_PATH="ArmoRM_iter_dpo/iter_0_ckpt"
POLICY_2_MODEL_PATH="ArmoRM_vanilla_iter_dpo/iter_0_ckpt"
REWARD_MODEL_PATH="RLHFlow/ArmoRM-Llama3-8B-v0.1"
DATASET_PATH="mnoukhov/openai_summarize_comparisons_tldrprompt_relabel1b"

POLICY_1_MODEL_FILENAME=$(echo "${POLICY_1_MODEL_PATH}" | tr '/' '_')
POLICY_2_MODEL_FILENAME=$(echo "${POLICY_2_MODEL_PATH}" | tr '/' '_')

POLICY_1_GENERATE_OUTPUT="${eval_dir}/${POLICY_1_MODEL_FILENAME}_generate.jsonl"
POLICY_2_GENERATE_OUTPUT="${eval_dir}/${POLICY_2_MODEL_FILENAME}_generate.jsonl"
POLICY_1_RM_OUTPUT="${eval_dir}/${POLICY_1_MODEL_FILENAME}_rm.jsonl"
POLICY_2_RM_OUTPUT="${eval_dir}/${POLICY_2_MODEL_FILENAME}_rm.jsonl"

POLICY_1_MODEL_PATH="${base_dir}/${POLICY_1_MODEL_PATH}"
POLICY_2_MODEL_PATH="${base_dir}/${POLICY_2_MODEL_PATH}"


checkSuccess() {
    if [[ $? != 0 ]]; then
        echo "FAILED $1"
        exit 1
    fi
}
export PATH=$HOME/.local/bin/:$PATH

generate_commands="examples/batch_inference.py \
    --eval_task generate_vllm \
    --pretrain $POLICY_1_MODEL_PATH \
    --max_new_tokens 48 \
    --dataset $DATASET_PATH \
    --dataset_probs 1.0 \
    --temperature $TEMPERATURE \
    --tp_size 4 \
    --best_of_n $BEST_OF_N \
    --max_num_seqs 128 \
    --eval \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --output_path $POLICY_1_GENERATE_OUTPUT"
echo $generate_commands
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS python $generate_commands
checkSuccess "GENERATE"

get_rewards_commands="examples/batch_inference.py \
    --eval_task rm \
    --pretrain $REWARD_MODEL_PATH \
    --bf16 \
    --max_len 2048 \
    --dataset $POLICY_1_GENERATE_OUTPUT \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --post_processor eval \
    --micro_batch_size 8 \
    --eval \
    --output_path $POLICY_1_RM_OUTPUT"
echo $get_rewards_commands
deepspeed --include localhost:$AVAILABLE_GPUS $get_rewards_commands
checkSuccess "RM"

generate_commands="examples/batch_inference.py \
    --eval_task generate_vllm \
    --pretrain $POLICY_2_MODEL_PATH \
    --max_new_tokens 48 \
    --dataset $DATASET_PATH \
    --dataset_probs 1.0 \
    --temperature $TEMPERATURE \
    --tp_size 4 \
    --best_of_n $BEST_OF_N \
    --max_num_seqs 128 \
    --eval \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --output_path $POLICY_2_GENERATE_OUTPUT"
echo $generate_commands
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS python $generate_commands
checkSuccess "GENERATE"

get_rewards_commands="examples/batch_inference.py \
    --eval_task rm \
    --pretrain $POLICY_2_MODEL_PATH \
    --bf16 \
    --max_len 2048 \
    --dataset $POLICY_2_GENERATE_OUTPUT \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --post_processor eval \
    --micro_batch_size 8 \
    --eval \
    --output_path $POLICY_2_RM_OUTPUT"
echo $get_rewards_commands
deepspeed --include localhost:$AVAILABLE_GPUS $get_rewards_commands
checkSuccess "RM"

win_rate_commands="examples/win_rate.py \
    --file1 $POLICY_1_RM_OUTPUT \
    --file2 $POLICY_2_RM_OUTPUT \
    --output ${eval_dir}/${POLICY_2_MODEL_FILENAME}_win_rate_over_${POLICY_1_MODEL_FILENAME}.csv"
echo $get_rewards_commands
python $win_rate_commands
checkSuccess "WIN_RATE"
