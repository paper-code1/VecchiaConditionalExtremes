#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=batch
#SBATCH -J Single_A100
#SBATCH -o Single_A100.%J.out
#SBATCH -e Single_A100.%J.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --mem=100G # try larger memory

N_all=(10000 30000 70000 100000 300000 500000 700000 800000 1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 12000000 16000000) #
N_bs=(100)
M_ests=(100 200 400)

# N_all=(8000000 10000000 12000000 16000000) #
# N_bs=(100)
# M_ests=(100 200)

# make -j

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

for N in ${N_all[@]}; do
    # Scaled block Vecchia
    for M_est in ${M_ests[@]}; do
        for N_b in ${N_bs[@]}; do
            for i in {1..3}; do
                bc=$((N/N_b))
                m_bv=$M_est
                echo "N: $N, bc: $bc, m_bv: $m_bv, seed: $i"
                if [ \( $N -le 4000000 -a $m_bv -eq 400 \) -o \( $N -gt 8000000 -a $m_bv -eq 100 \) -o \( $N -gt 16000000 -a $m_bv -eq 100 \) ]; then
                    ./bin/dbv \
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
                        --nn_multiplier 500 \
                        --log_append GH200_single \
                        --omp_num_threads 10 \
                        --print=false
                fi
            done
        done
    done

    # Scaled Vecchia
    if [ $N -le 800000 ]; then
        for i in {1..3}; do
           ./bin/dbv \
                --num_total_points $N \
                --num_total_blocks $N \
                --distance_scale $distance_scale \
                --distance_scale_init $distance_scale_init \
                --theta_init $theta_init \
                -m 60 \
                --dim $DIM \
                --mode estimation \
                --maxeval 500 \
                --xtol_rel 1e-8 \
                --ftol_rel 1e-8 \
                --kernel_type Matern72 \
                --seed $i \
                --nn_multiplier 500 \
                --log_append GH200_single \
                --omp_num_threads 10 \
                --print=false
        done
    fi
done

mkdir -p ./log/A100_single
mv ./log/*_A100_single.csv ./log/A100_single/
