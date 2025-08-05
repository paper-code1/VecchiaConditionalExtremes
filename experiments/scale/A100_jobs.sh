#!/bin/bash

# # Submit the single-node A100 benchmarks (split into 3 parts)
sbatch experiments/scale/Job_A100_single_part1.sh # 4:00:00
sbatch experiments/scale/Job_A100_single_part2.sh # 8:00:00
sbatch experiments/scale/Job_A100_single_part3.sh # 4:00:00

# Submit the base scaling job (1 GPU)
sbatch experiments/scale/Job_A100_1.sh # 2 hour

# Submit weak and strong scaling jobs for multiple GPUs
sbatch experiments/scale/Job_A100_2.sh # 2 hours
sbatch experiments/scale/Job_A100_4.sh # 2 hours
sbatch experiments/scale/Job_A100_8.sh # 2 hours
sbatch experiments/scale/Job_A100_16.sh # 2 hours
sbatch experiments/scale/Job_A100_32.sh # 2 hours
sbatch experiments/scale/Job_A100_64.sh # 2 hours
sbatch experiments/scale/Job_A100_128.sh # 2 hours
sbatch experiments/scale/Job_A100_256.sh # 2 hours
sbatch experiments/scale/Job_A100_512.sh # 2 hours

sbatch experiments/scale/Job_A100_power.sh # 1 hour 

echo "All A100 scaling jobs submitted."
