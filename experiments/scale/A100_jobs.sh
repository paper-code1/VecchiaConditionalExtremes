#!/bin/bash

# # Submit the single-node A100 benchmarks (split into 3 parts)
sbatch experiments/scale/Job_A100_single_part1.sh
sbatch experiments/scale/Job_A100_single_part2.sh
sbatch experiments/scale/Job_A100_single_part3.sh

# Submit the base scaling job (1 GPU)
sbatch experiments/scale/Job_A100_1.sh

# Submit weak and strong scaling jobs for multiple GPUs
sbatch experiments/scale/Job_A100_2.sh
sbatch experiments/scale/Job_A100_4.sh
sbatch experiments/scale/Job_A100_8.sh
sbatch experiments/scale/Job_A100_16.sh
sbatch experiments/scale/Job_A100_32.sh
sbatch experiments/scale/Job_A100_64.sh

echo "All A100 scaling jobs submitted."
