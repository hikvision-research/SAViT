cd /path to deit_savit
GPU_NUM=1
python main.py \
        --finetune=/path to deit_small checkpoint \
        --batch-size=32 \
        --num_workers=16 \
        --data-path=/path to ImageNet \
        --model=deit_small_patch16_224 \
        --pruning_per_iteration=100 \
        --pruning_feed_percent=0.1 \
        --pruning_method=2 \
        --pruning_layers=3 \
        --pruning_flops_percentage=0.30 \
        --pruning_flops_threshold=0.0001 \
        --need_hessian  \
        --finetune_op=2 \
        --epochs=1 \
        --output_dir=/path to output