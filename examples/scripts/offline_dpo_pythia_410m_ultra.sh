set -x 

AVAILABLE_GPUS="4"
script_name=$(basename $0 .sh)
# mkdir -p ./ckpt/pythia_410m/dpo/
MODEL_OUTPUT_PATH=/data02/wenhao/jl/ckpt/pythia_410m/ultra/offline_dpo_1_epoch

POLICY_MODEL_PATH=EleutherAI/pythia-410m
DATASET_PATH=HuggingFaceH4/ultrafeedback_binarized
ROLLOUT_BATCH_SIZE=100000
REF_MODEL_PATH=$POLICY_MODEL_PATH
read -r -d '' training_commands <<EOF
examples/train_dpo.py \
    --max_len 2048 \
    --dataset $DATASET_PATH \
    --dataset_probs 1.0 \
    --logging_steps 10 \
    --eval_steps 100 \
    --train_batch_size 128 \
    --micro_train_batch_size 16 \
    --learning_rate 1e-5 \
    --pretrain $POLICY_MODEL_PATH \
    --ref_pretrain $REF_MODEL_PATH \
    --save_path $MODEL_OUTPUT_PATH \
    --zero_stage 3 \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --beta 0.1 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --use_wandb 9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc \
    --wandb_run_name "${script_name}" 
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --include localhost:$AVAILABLE_GPUS $training_commands
fi
