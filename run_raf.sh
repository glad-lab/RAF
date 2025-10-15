#!/bin/bash
python -m experiment.main \
    --dataset json \
    --model llama-3.1-8b \
    --catalog coffee_machines \
    --target_product_idx 1 \
    --seed 42 \
    --topk 512 \
    --w_tar_1 300 \
    --w_tar_2 40 \
    --num_templates 10 \
    --control_loss_method last_token_ll \
    --single_template \
    --n_steps 300 \
    --random_order \
    --use_entropy_adaptive_weighting \
    --entropy_alpha 3.0 \
    "$@"