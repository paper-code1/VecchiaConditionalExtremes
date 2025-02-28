#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=batch
#SBATCH -J Single_A100
#SBATCH -o Single_A100.%J.out
#SBATCH -e Single_A100.%J.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100,4gpus
#SBATCH --mem=300G # try larger memory

export OMP_NUM_THREADS=10

N_all=(10000 20000 50000 80000 100000 200000 500000 800000 1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 12000000) #
N_bs=(100)
M_ests=(200 400 600)

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
                echo "N: $N, bc: $bc, m_bv: $m_bv, seed: $i"
                srun --exclusive --cpu-bind=none --cpus-per-task=10 ./bin/dbv --num_total_points $N --num_total_blocks $bc -m $m_bv --dim $DIM --mode estimation --maxeval 1 --kernel_type Matern72 --seed $i --nn_multiplier 10 --log_append A100_single --omp_num_threads 10
            done
        done
    done

    # # Scaled Vecchia
    # if [ $N -le 500000 ]; then
    #     for i in {1..5}; do
    #         srun ./bin/dbv --num_total_points $N --num_total_blocks $N -m 60 --dim $DIM --mode estimation --maxeval 1 --kernel_type Matern72 --seed $i --nn_multiplier 10 --log_append A100_single
    #     done
    # fi
done

mkdir -p ./log/A100_single
mv ./log/*_A100_single.csv ./log/A100_single/
