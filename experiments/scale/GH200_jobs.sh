#!/bin/bash

# Submit the single-node GH200 benchmarks (split into 3 parts)
sbatch experiments/scale/Job_GH200_single_part1.sh # 4 hours
sbatch experiments/scale/Job_GH200_single_part2.sh # 22 hours
# sbatch experiments/scale/Job_GH200_single_part3_optional.sh # optional

# Submit the base scaling job (1 GPU)
sbatch experiments/scale/Job_GH200_1.sh # 1 hour

# Submit weak and strong scaling jobs for multiple GPUs
sbatch experiments/scale/Job_GH200_2.sh # 3.5 hours
sbatch experiments/scale/Job_GH200_4.sh # 3.5 hours
sbatch experiments/scale/Job_GH200_8.sh # 3.5 hours
sbatch experiments/scale/Job_GH200_16.sh # 3.5 hours
sbatch experiments/scale/Job_GH200_32.sh # 3.5 hours
sbatch experiments/scale/Job_GH200_64.sh # 3.5 hours

# sbatch experiments/scale/Job_GH200_power.sh # 1 hour
sbatch experiments/scale/Job_GH200_power_jpwr.sh # 1 hour

echo "All GH200 scaling jobs submitted."
