export CUDA_VISIBLE_DEVICES='1'
mpiexec -n 8 python scripts/inverse.py \
    --batch_size 1 \
    --ckpt ./model/cd_cat256_lpips.pt \
    --task_config configs/super_resolution_config.yaml \
    --datasets_path "data/samples_cat" \
    --logdir results/lsun_cat_256/super_res \
    --save_img \
    --training_mode consistency_distillation \
    --steps 65 \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --use_scale_shift_norm False \
    --dropout 0.0 \
    --image_size 256 \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --num_samples 1 \
    --resblock_updown True \
    --use_fp16 False \
    --weight_schedule uniform 