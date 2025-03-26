#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH -J scaling_a100_4
#SBATCH -o scaling_a100_4.%J.out
#SBATCH -e scaling_a100_4.%J.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
#SBATCH --mem=500G # try larger memory


N_base_strong=(4000000 4000000) # larger problem BSV 100/400 GH200
M_ests=(600 400)
N_bs=(300 100)
nn_multipliers=(500 500)
num_GPUs=4

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

for index in {0..1}; do
    # Calculate N_base_weak as N_base_strong[0] * (2^(0.6))
    N_base_weak=$((N_base_strong[$index]*num_GPUs))
    N_bs_weak=${N_bs[$index]}
    N_bc_weak=$((N_base_weak/N_bs_weak))
    M_est_weak=${M_ests[$index]}
    nn_multiplier_weak=${nn_multipliers[$index]}
    # print N_base_weak, N_bs_weak, N_bc_weak, M_est_weak
    echo "N_base_weak: $N_base_weak, N_bs_weak: $N_bs_weak, N_bc_weak: $N_bc_weak, M_est_weak: $M_est_weak, nn_multiplier_weak: $nn_multiplier_weak"
    for i in {1..3}; do
        srun --exclusive  ./bin/dbv \
            --num_total_points $N_base_weak \
            --num_total_blocks $N_bc_weak \
            --distance_scale $distance_scale \
            --distance_scale_init $distance_scale_init \
            --theta_init $theta_init \
            -m $M_est_weak \
            --dim $DIM \
            --mode estimation \
            --maxeval 500 \
            --xtol_rel 1e-8 \
            --ftol_rel 1e-8 \
            --kernel_type Matern72 \
            --seed $i \
            --nn_multiplier $nn_multiplier_weak \
            --log_append A100_scaling_4\
            --omp_num_threads 15 \
            --print=false
    done
done

for index in {0..1}; do
    N_base_strong=$((N_base_strong[$index]))
    N_bs_strong=${N_bs[$index]}
    N_bc_strong=$((N_base_strong/N_bs_strong))
    M_est_strong=${M_ests[$index]}
    nn_multiplier_strong=${nn_multipliers[$index]}
    # print N_base_strong, N_bs_strong, N_bc_strong, M_est_strong
    echo "N_base_strong: $N_base_strong, N_bs_strong: $N_bs_strong, N_bc_strong: $N_bc_strong, M_est_strong: $M_est_strong, nn_multiplier_strong: $nn_multiplier_strong"
    for i in {1..3}; do
        srun --exclusive  ./bin/dbv \
            --num_total_points $N_base_strong \
            --num_total_blocks $N_bc_strong \
            --distance_scale $distance_scale \
            --distance_scale_init $distance_scale_init \
            --theta_init $theta_init \
            -m $M_est_strong \
            --dim $DIM \
            --mode estimation \
            --maxeval 500 \
            --xtol_rel 1e-8 \
            --ftol_rel 1e-8 \
            --kernel_type Matern72 \
            --seed $i \
            --nn_multiplier $nn_multiplier_strong \
            --log_append A100_scaling_4\
            --omp_num_threads 15 \
            --print=false
    done
done

mkdir -p ./log/A100_scaling
mv ./log/*_A100_scaling_4.csv ./log/A100_scaling/
