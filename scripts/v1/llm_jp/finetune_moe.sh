#!/bin/zsh

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01

cd ~/workspace/MoE-LLaVA-JP
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 
poetry run deepspeed moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.c_fc mlp.c_proj wg \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path ./LLaVA-llm-jp-1.3b-v1.0 \
    --version v1 \
    --data_path ./dataset/llava_instruct_150k_ja_sampled.json \
    --image_folder ./dataset/LLaVa-FineTuning-MoE \
    --image_tower google/siglip-so400m-patch14-384 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llm-jp-1.3b-finetune-moe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
