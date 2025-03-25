#!/bin/bash

# Submit the single-node GH200 benchmarks (split into 3 parts)
sbatch experiments/scale/Job_GH200_single_part1.sh
sbatch experiments/scale/Job_GH200_single_part2.sh
sbatch experiments/scale/Job_GH200_single_part3.sh

# Submit the base scaling job (1 GPU)
sbatch experiments/scale/Job_GH200_1.sh

# Submit weak and strong scaling jobs for multiple GPUs
sbatch experiments/scale/Job_GH200_2.sh
sbatch experiments/scale/Job_GH200_4.sh
sbatch experiments/scale/Job_GH200_8.sh
sbatch experiments/scale/Job_GH200_16.sh
sbatch experiments/scale/Job_GH200_32.sh
sbatch experiments/scale/Job_GH200_64.sh

echo "All GH200 scaling jobs submitted."
