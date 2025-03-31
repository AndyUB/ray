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

output_path=results/barbell/linear/correctness
mkdir -p $output_path
rm -f $output_path/*.log

num_partitions=2
ray_model_dir=results/barbell/linear/ray/fsdp/save_model
torch_model_dir=results/barbell/linear/torch/fsdp/save_model

layer_size=80
num_layers=2

latency_prefix=${timestamp}_ls${layer_size}_nl${num_layers}
log_file=${output_path}/${latency_prefix}.log

echo "Running layer_size $layer_size, num_layers $num_layers..."
python -m ray.experimental.fsdp.src.main.correctness \
	--layer-size $layer_size \
	--num-layers $num_layers \
	--num-units $num_partitions \
	--ray-model $ray_model_dir/*.pt \
	--torch-model $torch_model_dir/*.pt \
	>$log_file 2>&1
status=$?
