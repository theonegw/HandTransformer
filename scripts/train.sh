#!/bin/bash
export CUDA_VISIBLE_DEVICES="7"

python train.py \
    --d_model 128 \
    --num_heads 8 \
    --num_layers 2 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --epochs 20 \
    --seed 42 \
    --train_subset_size 10000 \
    --val_subset_size 500 \
    --model_save_path "models/hand_transformer.pt"