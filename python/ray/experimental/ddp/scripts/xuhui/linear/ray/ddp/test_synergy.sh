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

modes=(
	ray.ddp
	torch.ddp
)

output_path=results/xuhui/linear/ray/ddp/test_synergy
mkdir -p $output_path
rm -f $output_path/*.csv

layer_size=1024
num_layers=32
num_actors=4
num_iters=20

for mode in ${modes[@]}; do
	latency_prefix=${mode}_ls${layer_size}_nl${num_layers}
	model_file=$output_path/${timestamp}_${mode}_model.log
	model_prefix=$output_path/${timestamp}_${mode}_model
	log_file=$output_path/${timestamp}_${mode}.log

	if [ "$mode" != "torch.ddp" ]; then
		num_partitions=4
	else
		num_partitions=1
	fi

	python -m ray.experimental.ddp.src.main.linear.${mode} \
		--layer-size $layer_size \
		--num-layers $num_layers \
		--num-partitions $num_partitions \
		--num-actors $num_actors \
		--num-iters $num_iters \
		--output-path $output_path \
		--latency-prefix $latency_prefix \
		--save-model \
		--model-file $model_file \
		--model-prefix $model_prefix \
		--tracing \
		>$log_file 2>&1
	status=$?

	if [ $status -ne 0 ]; then
		echo -e "${RED}ER${NC}"
		if $debug; then
			code $log_file
		fi
		exit 1
	fi
done

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

file1="$output_path/${timestamp}_${modes[0]}_model_0.log"
file2="$output_path/${timestamp}_${modes[0]}_model_1.log"
compare_files "$file1" "$file2"

file1="$output_path/${timestamp}_${modes[0]}_model.log"
file2="$output_path/${timestamp}_${modes[1]}_model.log"
compare_files "$file1" "$file2"

echo -e "${GREEN}AC${NC}"
