#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -J scaling_GH200_power_jpwr
#SBATCH -o scaling_GH200_power_jpwr.%J.out
#SBATCH -e scaling_GH200_power_jpwr.%J.err
#SBATCH --time=1:00:00
#SBATCH -A jureap137

N_base_strong=(5000000 5000000 5000000) # larger problem BSV 100/400 GH200
M_ests=(100 200 400)
N_bs=(100 100 100)
nn_multipliers=(300 300 500)

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale
NUM_GPU=1
CPU_power_cap=(200) # please set the power cap for the CPU # 100 200

# Create monitoring directory
mkdir -p ./log/GH200_power_cpu_${CPU_power_cap}/monitoring

# base scaling
for index in {0..2}; do
    # Scaled block Vecchia
    N=${N_base_strong[$index]}
    m_bv=${M_ests[$index]}
    nn_multiplier=${nn_multipliers[$index]}
    N_b=${N_bs[$index]}
    for i in {1..1}; do
        bc=$((N/N_b))
        echo "N: $N, bc: $bc, m_bv: $m_bv"
        
        exp_id="N${N}_m${m_bv}_i${i}_gpu${NUM_GPU}_power"
        # Start the main process in background and get its PID
        srun --grace-power-cap=$CPU_power_cap jpwr --methods pynvml gh \
            --df-out ./log/GH200_power_cpu_${CPU_power_cap}/monitoring/ \
            --df-filetype csv --df-suffix m_${m_bv} -- ./bin/dbv \
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
            --log_append GH200_power_cpu_${CPU_power_cap}\
            --omp_num_threads 72 \
            --print=false
    done
done

mkdir -p ./log/GH200_power_cpu_${CPU_power_cap}
mv ./log/*_GH200_power_cpu_${CPU_power_cap}.csv ./log/GH200_power_cpu_${CPU_power_cap}/
