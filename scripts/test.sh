#!/bin/bash
export CUDA_VISIBLE_DEVICES="7"
MODEL_PATH="models/hand_transformer.pt"

TEST_SENTENCE="LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in 'Harry Potter and the Order of the Phoenix' To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties."

python test.py \
    --sentence "$TEST_SENTENCE" \
    --model_path "$MODEL_PATH" \
    --d_model 128 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 512
