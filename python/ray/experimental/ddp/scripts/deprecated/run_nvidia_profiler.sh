#!/bin/bash

mkdir -p results/profile

dtype=float32
num_iters=10
learning_rate=1e-5
num_actors=2

# num_layers, layer_size
layer_configs=(
    "2 1280"
    "4 1280"
    "16 1280"
    "64 1280"
    "16 2560"
)

for layer_config in "${layer_configs[@]}"; do
    read num_layer layer_size <<<$layer_config
    RAY_DEDUP_LOGS=0 \
        python3 ddp_profile.py \
        --num-layers $num_layer \
        --layer-size $layer_size \
        --dtype $dtype \
        --num-iters $num_iters \
        --learning-rate $learning_rate \
        --num-actors $num_actors \
        --breakdown-performance \
        --output-file results/profile/lat_${num_layer}_${layer_size}.csv \
        2>results/profile/run_${num_layer}_${layer_size}.log \
        >results/profile/out_${num_layer}_${layer_size}.log
done
