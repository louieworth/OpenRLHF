set -x




mkdir -p /data02/wenhao/jl/ckpt/pythia_410m/tldr/vanilla_iter_dpo
GENERATE_OUTPUT=/data02/wenhao/jl/ckpt/pythia_410m/tldr/vanilla_iter_dpo/generate.jsonl
RM_OUTPUT=/data02/wenhao/jl/ckpt/pythia_410m/tldr/vanilla_iter_dpo/rm.jsonl
MODEL_OUTPUT_PATH=/data02/wenhao/jl//ckpt/pythia_410m/tldr/vanilla_iter_dpo/ckpt
ITER_LOG_PATH=null

TRAINING_ITERS=2
ROLLOUT_BATCH_SIZE=1000

POLICY_MODEL_PATH=EleutherAI/pythia-410m
REWARD_MODEL_PATH=sfairXC/FsfairX-LLaMA3-RM-v0.1
DATASET_PATH=mnoukhov/openai_summarize_comparisons_tldrprompt_relabel1b
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
    # Use latest model if past first iteration
    if ((iter > 0)); then
        POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
    fi

    read -r -d '' generate_commands <<EOF
examples/batch_inference.py
    --eval_task generate_vllm \
    --pretrain $POLICY_MODEL_PATH \
    --max_new_tokens 1024 \
    --dataset $DATASET_PATH  \
    --dataset_probs 1.0 \
    --temperature 1.0 \
    --tp_size 4 \
    --best_of_n 8 \
    --max_num_seqs 128 \
    --iter $iter \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --output_path $GENERATE_OUTPUT 
EOF
    echo $generate_commands
    python $generate_commands
    checkSuccess "GENERATE"

    read -r -d '' get_rewards_commands <<EOF
examples/batch_inference.py
    --eval_task rm \
    --pretrain $REWARD_MODEL_PATH \
    --bf16 \
    --max_len 2048 \
    --dataset $GENERATE_OUTPUT  \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --post_processor iter_dpo \
    --micro_batch_size 8 \
    --output_path $RM_OUTPUT
EOF
    echo $get_rewards_commands
    deepspeed $get_rewards_commands
    checkSuccess "RM"

    read -r -d '' dpo_commands <<EOF
examples/train_dpo.py \
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
    --zero_stage 3 \
    --beta 0.1 \
    --max_epochs 1 \
    --bf16 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \ 
    --flash_attn \
    --use_wandb 9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc
EOF
    echo $dpo_commands
    deepspeed $dpo_commands
    checkSuccess "DPO"

    iter=$((iter + 1))
    if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH
    fi
done
