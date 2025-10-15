python test_multiple_ranking.py \
    --csv_file_template "result/raf_main/{model}/json/{catalog}/{target_idx}/raf_results.csv" \
    --model llama-3.1-8b \
    --catalog "coffee_machines" \
    --target_product_idx 1 2 3 4 5 6 7 8 9 10\
    --dataset json \
    --num_runs 10 \
    --random_order