# Distributed Block Scaled Vecchia Approximation for Gassian Processes

This repository contains code for performing BSV for GPs

## Prerequisites

- gcc (>=10.2.0)
- mkl (>=2020)
- NLOPT (>=2.7.0)
- CMake (>=3.21)
- CUDA (>=11.6)
- openmpi (>= 4.1.0)

## Installation

Clone the repository and `make -j`

## Example
`bash ./experiments/Job_A100_singl.sh` (if you are not in slurm, please change the srun into mpirun)
