set -x 
AVAILABLE_GPUS="1"
MODEL_PATH="sfairXC/FsfairX-LLaMA3-RM-v0.1"
OUPUT_PATH="/data02/wenhao/jl/ckpt/rm/rm-lmsys-FsfairX"
DATASET_PATH="/data02/wenhao/jl/datasets/lmsys_train.csv"
TEST_DATASET_PATH="/data02/wenhao/jl/datasets/lmsys_test.csv"

read -r -d '' training_commands <<EOF
examples/train_rm.py \
     --save_path $OUPUT_PATH \
     --save_steps -1 \
     --logging_steps 256 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 2 \
     --pretrain $MODEL_PATH \
     --bf16 \
     --max_epochs 10 \
     --max_len 1024 \
     --zero_stage 3 \
     --learning_rate 5e-6 \
     --dataset $DATASET_PATH \
     --test_dataset $TEST_DATASET_PATH \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_accumulation_steps 8 \
     --use_wandb 9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --include localhost:$AVAILABLE_GPUS $training_commands
fi
