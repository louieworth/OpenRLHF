set -x 

# mkdir -p ./ckpt/pythia_410m/dpo/
MODEL_OUTPUT_PATH=/data02/wenhao/jl/ckpt/pythia_410m/tldr/offline_dpo_1_epoch

POLICY_MODEL_PATH=EleutherAI/pythia-410m
DATASET_PATH=trl-internal-testing/tldr-preference-trl-style
REF_MODEL_PATH=$POLICY_MODEL_PATH
script_name=$(basename $0 .sh)
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
    --beta 0.5 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --use_wandb 9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc \
    --wandb_run_name "${script_name}" 
EOF
    #  --sanity_check \
    #  --gradient_checkpointing \
    #  --dataset_probs 0.72,0.08,0.12,0.08 \
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
