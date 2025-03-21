#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --partition=batch
#SBATCH -J scaling_a100_8
#SBATCH -o scaling_a100_8.%J.out
#SBATCH -e scaling_a100_8.%J.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
#SBATCH --mem=1000G # try larger memory

N_base_strong=(2000000) # larger problem BSV 100/400 A100

M_ests=(400)
N_bs=(100)

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

N_base_weak=$(awk -v n="${N_base_strong[0]}" 'BEGIN {printf "%d\n", n * (8)}')
N_bs_weak=$(awk -v n="${N_bs[0]}" 'BEGIN {printf "%d\n", n}')
N_bc_weak=$((N_base_weak/N_bs_weak))
M_est_weak=$(awk -v n="${M_ests[0]}" 'BEGIN {printf "%d\n", n}')
# print N_base_weak, N_bs_weak, N_bc_weak, M_est_weak
echo "N_base_weak: $N_base_weak, N_bs_weak: $N_bs_weak, N_bc_weak: $N_bc_weak, M_est_weak: $M_est_weak"


# weak scaling
for N in ${N_base_weak[@]}; do
    # Scaled block Vecchia
    for M_est in ${M_est_weak[@]}; do
        for N_b in ${N_bs_weak[@]}; do
            for i in {1..3}; do
                bc=${N_bc_weak[0]}
                echo "N: $N, bc: $bc, m_bv: $M_est, N_b: $N_b"
                srun --exclusive  ./bin/dbv \
                    --num_total_points $N \
                    --num_total_blocks $bc \
                    --distance_scale $distance_scale \
                    --distance_scale_init $distance_scale_init \
                    --theta_init $theta_init \
                    -m $M_est \
                    --dim $DIM \
                    --mode estimation \
                    --maxeval 500 \
                    --xtol_rel 1e-8 \
                    --ftol_rel 1e-8 \
                    --kernel_type Matern72 \
                    --seed $i \
                    --nn_multiplier 500 \
                    --log_append A100_scaling_8\
                    --omp_num_threads 10 \
                    # --print=false
            done
        done
    done
done


# strong scaling
for N in ${N_base_strong[@]}; do
    # Scaled block Vecchia
    for M_est in ${M_ests[@]}; do
        for N_b in ${N_bs[@]}; do
            for i in {1..3}; do
                bc=$((N/N_b))
                m_bv=$M_est
                echo "N: $N, bc: $bc, m_bv: $m_bv"
                srun --exclusive ./bin/dbv \
                    --num_total_points $N \
                    --num_total_blocks $bc \
                    --distance_scale $distance_scale \
                    --distance_scale_init $distance_scale_init \
                    --theta_init $theta_init \
                    -m $m_bv \
                    --dim $DIM \
                    --mode estimation \
                    --maxeval 500 \
                    --xtol_rel 1e-8 \
                    --ftol_rel 1e-8 \
                    --kernel_type Matern72 \
                    --seed $i \
                    --nn_multiplier 70 \
                    --log_append A100_scaling_8\
                    --omp_num_threads 10 \
                    # --print=false
            done
        done
    done
done

mkdir -p ./log/A100_scaling
mv ./log/*_A100_scaling_8.csv ./log/A100_scaling/
