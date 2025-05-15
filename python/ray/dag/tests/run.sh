# rm log/*.log

# for test in test*.py; do
#     RAY_DEDUP_LOGS=0 \
#         python3 $test >>log/$test.log 2>&1
# done

# rm log/experimental/*.log

# for test in experimental/test*.py; do
#     RAY_DEDUP_LOGS=0 \
#         python3 $test >>log/$test.log 2>&1
# done

gpu_tests="test_torch_tensor_dag.py test_multi_args_gpu.py test_execution_schedule_gpu.py"

rm log/gpu/*.log

for test in $gpu_tests; do
    RAY_PYTEST_USE_GPU=1 RAY_DEDUP_LOGS=0 \
        python3 experimental/$test >>log/gpu/$test.log 2>&1
done
