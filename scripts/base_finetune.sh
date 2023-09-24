cd /path to deit_savit
GPU_NUM=8
output_dir=/path to output
ck_dir=$output_dir/checkpoint.pth
# check if checkpoint exists
if [ -e $ck_dir ]; then
   CMD="--resume=${ck_dir}"
else
   CMD="--resume="
fi
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM}  --use_env  main_deploy.py \
        --dist-eval \
        $CMD \
        --masked_model=/path to pruned_model in previous step prune \
        --teacher-path=/path to regnet model as deit paper \
        --batch-size=128 \
        --num_workers=16 \
        --data-path=/path to ImageNet \
        --model=deit_base_patch16_224_deploy \
        --pruning_flops_percentage=0 \
        --finetune_op=1 \
        --epochs=300 \
        --warmup-epochs=0 \
        --cooldown-epochs=0 \
        --output_dir=$output_dir