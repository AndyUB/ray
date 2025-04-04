#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/vit ]]; then
	echo "Please run in the python/ray/experimental/vit directory"
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

output_path=results/xuhui_n4/v3/lld/$timestamp
mkdir -p $output_path
echo "Running $output_path..."

num_iters=20
num_dp_list=(1 2 4)

log_file=$output_path/actors_bs16_dp4.log
# python src/core/v3.py \
# 	--num_iters $num_iters \
# 	--bs_single 16 \
# 	--num_dp 4 \
# 	>$log_file 2>&1
python src/core/v3.py \
	--num_iters $num_iters \
	--bs_single 16 \
	--num_dp_vision 1 \
	--num_dp 2 \
	>$log_file 2>&1
status=$?
if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

# for num_dp in "${num_dp_list[@]}"; do
# 	log_file=$output_path/actors_bs8_dp${num_dp}.log
# 	python src/core/v3.py \
# 		--num_iters $num_iters \
# 		--bs_single 8 \
# 		--num_dp $num_dp \
# 		>$log_file 2>&1
# 	status=$?
# 	if [ $status -ne 0 ]; then
# 		echo -e "${RED}ER${NC}"
# 		exit 1
# 	fi
# done

echo -e "${GREEN}AC${NC}"
