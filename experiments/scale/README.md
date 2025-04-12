# GPU Scaling Experiments

This directory contains scripts and notebooks for evaluating the scaling performance of Block Vecchia methods on A100 and H100/GH200 GPUs.

## Directory Structure

- `Job_A100_*.sh`: Scripts for running experiments on A100 GPUs with different configurations
- `Job_GH200_*.sh`: Scripts for running experiments on H100/GH200 GPUs with different configurations
- `A100_jobs.sh`: Convenience script to launch multiple A100 jobs
- `GH200_jobs.sh`: Convenience script to launch multiple GH200 jobs
- `PerformanceVisu_A100_scaling.ipynb`: Notebook for visualizing A100 scaling results
- `PerformanceVisu_GH200_scaling.ipynb`: Notebook for visualizing GH200 scaling results
- `PerformacneGPU_A100.ipynb`: Analysis notebook for A100 performance
- `PerformacneGPU_GH200.ipynb`: Analysis notebook for GH200 performance
- `fig/`: Directory containing generated figures

## Prerequisites

- Slurm system with A100/GH200 GPUs

## Reproducing the Experiments

### Step 1: Run Scaling Experiments 

#### Option A: Use the convenience scripts to launch multiple jobs

```bash
# For A100 GPUs
sbatch A100_jobs.sh

# For GH200 GPUs
sbatch GH200_jobs.sh
```

#### Option B: Run individual configurations

For single-GPU experiments:
```bash
sbatch Job_A100_1.sh     # Single A100 GPU
sbatch Job_GH200_1.sh    # Single GH200 GPU
```

For multi-GPU experiments (2, 4, 8, 16, 32, 64 GPUs):
```bash
sbatch Job_A100_2.sh     # 2 A100 GPUs
sbatch Job_GH200_2.sh    # 2 GH200 GPUs
# ...and similarly for 4, 8, 16, 32, 64 configurations
```

For power measurement jobs:
```bash
sbatch Job_A100_power.sh
sbatch Job_GH200_power.sh
sbatch Job_GH200_power_jpwr.sh  # Alternative power measurement
```

### Step 2: Visualization and Analysis

Run the visualization notebooks to analyze the scaling performance:

```bash
# For A100 GPU scaling analysis
jupyter notebook PerformanceVisu_A100_scaling.ipynb

# For GH200 GPU scaling analysis
jupyter notebook PerformanceVisu_GH200_scaling.ipynb

# For detailed performance analysis
jupyter notebook PerformacneGPU_A100.ipynb
jupyter notebook PerformacneGPU_GH200.ipynb
```

These notebooks load results from `./log/A100_scaling/` and `./log/GH200_scaling/` and generate comparative visualizations saved to the `fig/` directory.

## Experiment Details

The scaling experiments measure:
- Strong scaling: Fixed problem size, increasing number of GPUs
- Weak scaling: Problem size grows with number of GPUs
- Single-GPU performance across different configurations
- Power efficiency and throughput

Key parameters across experiments:
- Problem sizes from 5M to 320M points
- Block counts proportional to point counts
- Nearest neighbors: 100, 200, 400
- Dimension: 10
- Kernel type: Matern72
- Multiple runs (typically 5) for statistical significance

## Output Files

Results are saved to:
- `./log/A100_scaling/`: Results from A100 GPU experiments
- `./log/GH200_scaling/`: Results from GH200 GPU experiments

Important metrics in output files:
- Runtime (seconds)
- Power consumption (for power measurement jobs)
- Convergence information

## Notes

- The time needed please see the comments in `A100_jobs.sh` and `GH200_jobs.sh`, which gives the details estimated time cost.