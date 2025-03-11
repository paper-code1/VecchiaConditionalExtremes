#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=batch
#SBATCH -J 2node_1gpu_v100
#SBATCH -o 2node_1gpu_v100.%J.out
#SBATCH -e 2node_1gpu_v100.%J.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=300G # try larger memory

N_all=(10000 20000 50000 80000 100000 200000 500000 800000 1000000 2000000 3000000 4000000) #
N_bs=(100)
M_ests=(400)

source ~/.bashrc
make

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

for N in ${N_all[@]}; do
    # Scaled block Vecchia
    for M_est in ${M_ests[@]}; do
        for N_b in ${N_bs[@]}; do
            for i in {1..5}; do
                bc=$((N/N_b))
                m_bv=$M_est
                echo "N: $N, bc: $bc, m_bv: $m_bv"
                srun --exclusive  ./bin/dbv --num_total_points $N --num_total_blocks $bc -m $m_bv --dim $DIM --mode estimation --maxeval 1 --kernel_type Matern72 --seed $i --nn_multiplier 50 --log_append V100_multi2 --omp_num_threads 10
            done
        done
    done
done

mkdir -p ./log/V100_multi2
mv ./log/*_V100_multi2.csv ./log/V100_multi2/
