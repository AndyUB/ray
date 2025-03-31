#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/fsdp ]]; then
	echo "Please run in the python/ray/experimental/fsdp directory"
	exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

debug=false
while getopts "d" opt; do
	case $opt in
	d) debug=true ;;
	*)
		echo "Usage: $0 [-d]" >&2
		echo "  -d    Enable debug mode"
		exit 1
		;;
	esac
done

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

export RAY_DEDUP_LOGS=0

output_path=results/xuhui_n4/linear/torch/no_prefetch/exp_self
mkdir -p $output_path
rm -f $output_path/*.csv
rm -f $output_path/*.log

layer_size_values=(
	2560 1280 640 512 320 160 80
)
num_layers_values=(
	10 40 160 250 640 2560 10240
)

num_actors=4
num_iters=20

for i in "${!layer_size_values[@]}"; do
	layer_size="${layer_size_values[$i]}"
	num_layers="${num_layers_values[$i]}"

	latency_prefix=${timestamp}_ls${layer_size}_nl${num_layers}
	model_prefix=${output_path}/${latency_prefix}_model
	log_file=${output_path}/${latency_prefix}.log

	echo "Running layer_size $layer_size, num_layers $num_layers..."
	python -m ray.experimental.fsdp.src.main.linear.torch.no_prefetch \
		--layer-size $layer_size \
		--num-layers $num_layers \
		--num-actors $num_actors \
		--num-iters $num_iters \
		--output-path $output_path \
		--latency-prefix $latency_prefix \
		--model-prefix $model_prefix \
		>$log_file 2>&1
	status=$?
done
