#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
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

output_path=results/xuhui/linear/ray/no_overlap/exp_self
mkdir -p $output_path
rm -f $output_path/*.csv
rm -f $output_path/*.log

layer_size_values=(
	2560 1280 640 512 320 160 80
)
num_layers_values=(
	10 40 160 250 640 2560 10240
)

num_partitions=10
num_actors=4
num_iters=20

for i in "${!layer_size_values[@]}"; do
	layer_size="${layer_size_values[$i]}"
	num_layers="${num_layers_values[$i]}"

	latency_prefix=${timestamp}_ls${layer_size}_nl${num_layers}
	model_prefix=${output_path}/${latency_prefix}_model
	log_file=${output_path}/${latency_prefix}.log

	echo "Running layer_size $layer_size, num_layers $num_layers..."
	python -m ray.experimental.ddp.src.main.linear.ray.no_overlap \
		--layer-size $layer_size \
		--num-layers $num_layers \
		--num-partitions $num_partitions \
		--num-actors $num_actors \
		--num-iters $num_iters \
		--output-path $output_path \
		--latency-prefix $latency_prefix \
		--model-prefix $model_prefix \
		--tracing \
		>$log_file 2>&1
	status=$?
done
