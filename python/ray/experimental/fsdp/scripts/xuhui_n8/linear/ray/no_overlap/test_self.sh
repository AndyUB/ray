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

output_path=results/xuhui_n4/linear/ray/no_overlap/test_self
mkdir -p $output_path

layer_size=1024
num_layers=8
num_partitions=4
num_actors=4
num_iters=10
latency_prefix=${timestamp}_ls${layer_size}_nl${num_layers}
model_prefix=$output_path/${timestamp}_model
log_file=$output_path/${timestamp}.log

RAY_CGRAPH_VISUALIZE_SCHEDULE=1 \
	python -m ray.experimental.fsdp.src.main.linear.ray.no_overlap \
	--layer-size $layer_size \
	--num-layers $num_layers \
	--num-partitions $num_partitions \
	--num-actors $num_actors \
	--num-iters $num_iters \
	--output-path $output_path \
	--latency-prefix $latency_prefix \
	--save-model \
	--model-prefix $model_prefix \
	--tracing \
	>$log_file 2>&1
status=$?

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

compare_files() {
	local file1="$1"
	local file2="$2"

	if [ ! -f "$file1" ]; then
		echo -e "${RED}Error: File '$file1' does not exist${NC}"
		exit 1
	fi
	if [ ! -f "$file2" ]; then
		echo -e "${RED}Error: File '$file2' does not exist${NC}"
		exit 1
	fi

	if ! diff "$file1" "$file2"; then
		echo -e "${RED}ER${NC}"
		if $debug; then
			code "$file1"
			code "$file2"
		fi
		exit 1
	fi
}

# file1="${output_path}/${timestamp}_model_0.log"
# file2="${output_path}/${timestamp}_model_1.log"
# compare_files "$file1" "$file2"

echo -e "${GREEN}AC${NC}"
