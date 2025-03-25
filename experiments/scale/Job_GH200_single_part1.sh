#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=batch
#SBATCH -J Single_GH200
#SBATCH -o Single_GH200.%J.out
#SBATCH -e Single_GH200.%J.err
#SBATCH --time=10:00:00
#SBATCH -A jureap137

N_all=(10000 30000 70000 100000 300000 500000 700000 800000)
N_bs=(100)
M_ests=(100 200 400)
nn_multipliers=(100 200 500)

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

for index in {0..2}; do
    N=${N_all[$index]}
    nn_multiplier=${nn_multipliers[$index]}
    # Scaled block Vecchia
    for M_est in ${M_ests[@]}; do
        for N_b in ${N_bs[@]}; do
            for i in {1..3}; do
                bc=$((N/N_b))
                m_bv=$M_est
                echo "N: $N, bc: $bc, m_bv: $m_bv, seed: $i"
                if [ \( $N -le 5000000 -a $m_bv -eq 400 \) -o \( $N -le 10000000 -a $m_bv -eq 200 \) -o \( $N -le 22000000 -a $m_bv -eq 100 \) ]; then
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
                        --nn_multiplier $nn_multiplier \
                        --log_append GH200_single \
                        --omp_num_threads 72 \
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
                --nn_multiplier 50 \
                --log_append GH200_single \
                --omp_num_threads 72 \
                --print=false
        done
    fi
done

mkdir -p ./log/GH200_single
mv ./log/*_GH200_single.csv ./log/GH200_single/
