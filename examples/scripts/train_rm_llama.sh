set -x 
AVAILABLE_GPUS="0,1,2,3,4,5,6,7"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
OUPUT_PATH="/data02/wenhao/jl/ckpt/rm/rm-tldr-Meta-Llama-3-8B-Instruct"
DATASET_PATH="when2rl/tldr-summarisation-preferences_reformatted"

read -r -d '' training_commands <<EOF
examples/train_rm.py \
     --save_path $OUPUT_PATH \
     --save_steps -1 \
     --logging_steps 50 \
     --eval_steps 200 \
     --train_batch_size 128 \
     --micro_train_batch_size 4 \
     --pretrain $MODEL_PATH \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset $DATASET_PATH \
     --dataset_probs 1 \
     --flash_attn \
     --use_wandb 9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --include localhost:$AVAILABLE_GPUS $training_commands
fi
