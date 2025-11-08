#!/bin/bash
export CUDA_VISIBLE_DEVICES="7"
MODEL_PATH="models/hand_transformer.pt"

TEST_SENTENCE="Weather forecasts predict heavy rain and strong winds moving in from the west, expected to arrive by tomorrow morning."

python test.py \
    --sentence "$TEST_SENTENCE" \
    --model_path "$MODEL_PATH" \
    --d_model 128 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 512
