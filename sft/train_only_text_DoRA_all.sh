export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export NPROC_PER_NODE=6

MODEL_ID="/data0/dywang/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"

echo "Starting DoRA Training (SFT)..."

swift sft \
    --model $MODEL_ID \
    --train_type lora \
    --use_dora true \
    --output_dir output_dora_optimized_all \
    --dataset 'data/all_train_shuffled.jsonl' \
    --val_dataset 'data/all_dev_shuffled.jsonl' \
    --num_train_epochs 3 \
    --max_length 2048 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --neftune_noise_alpha 10 \
    --eval_steps 300 \
    --save_steps 300 \
    --save_total_limit 2 \
    --save_only_model true \
    --logging_steps 100 \
    --report_to wandb \
    --torch_dtype bfloat16 \
    --load_best_model_at_end true \
    --metric_for_best_model loss \
    --greater_is_better false 