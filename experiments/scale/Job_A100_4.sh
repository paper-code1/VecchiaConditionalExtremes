#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=31
#SBATCH --partition=batch
#SBATCH -J scaling_a100_4
#SBATCH -o scaling_a100_4.%J.out
#SBATCH -e scaling_a100_4.%J.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
#SBATCH --mem=800G 


N_base_strong=(4000000 4000000 4000000) # larger problem BSV 100/400 GH200
M_ests=(100 200 400)
N_bs=(100 100 100)
nn_multipliers=(300 300 500)
num_GPUs=4

DIM=10
theta_init=1.0,0.001
distance_scale=0.05,0.01,0.05,5.0,5.0,5.0,5.0,5.0,5.0,5.0
distance_scale_init=$distance_scale

# Create monitoring directory
mkdir -p ./log/A100_scaling/monitoring

# Function to monitor CPU and GPU
monitor_resources() {
    local exp_id=$1
    
    # Monitor system-wide CPU and memory stats using vmstat (1 second intervals)
    vmstat -w 1 | awk '{now=strftime("%Y-%m-%d %H:%M:%S "); print now $0}' > "./log/A100_scaling/monitoring/cpu_mem_${exp_id}.log" &
    VMSTAT_PID=$!
    
    # Monitor GPU stats every 5 seconds
    nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv -l 5 > "./log/A100_scaling/monitoring/gpu_${exp_id}.log" &
    GPU_PID=$!
}

# Function to stop monitoring safely
stop_monitoring() {
    if [ ! -z "$VMSTAT_PID" ]; then
        kill $VMSTAT_PID 2>/dev/null || true
    fi
    if [ ! -z "$GPU_PID" ]; then
        kill $GPU_PID 2>/dev/null || true
    fi
}

# Weak scaling section
for index in {0..2}; do
    N_base_weak=$((N_base_strong[$index]*num_GPUs))
    N_bs_weak=${N_bs[$index]}
    N_bc_weak=$((N_base_weak/N_bs_weak))
    M_est_weak=${M_ests[$index]}
    nn_multiplier_weak=${nn_multipliers[$index]}
    echo "N_base_weak: $N_base_weak, N_bs_weak: $N_bs_weak, N_bc_weak: $N_bc_weak, M_est_weak: $M_est_weak, nn_multiplier_weak: $nn_multiplier_weak"
    
    for i in {1..1}; do
        exp_id="weak_N${N_base_weak}_m${M_est_weak}_i${i}_gpu${num_GPUs}"
        
        # Start the main process in background and get its PID
        srun --exclusive ./bin/dbv \
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
            --print=false &
        
        MAIN_PID=$!
        
        # Start monitoring with the process PID
        monitor_resources "$exp_id"
        
        # Wait for the main process to complete
        wait $MAIN_PID
        
        # Stop monitoring
        stop_monitoring
    done
done

# Strong scaling section
for index in {0..2}; do
    N_base_strong=$((N_base_strong[$index]))
    N_bs_strong=${N_bs[$index]}
    N_bc_strong=$((N_base_strong/N_bs_strong))
    M_est_strong=${M_ests[$index]}
    nn_multiplier_strong=${nn_multipliers[$index]}
    echo "N_base_strong: $N_base_strong, N_bs_strong: $N_bs_strong, N_bc_strong: $N_bc_strong, M_est_strong: $M_est_strong, nn_multiplier_strong: $nn_multiplier_strong"
    
    for i in {1..1}; do
        exp_id="strong_N${N_base_strong}_m${M_est_strong}_i${i}_gpu${num_GPUs}"
        
        # Start the main process in background and get its PID
        srun --exclusive ./bin/dbv \
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
            --print=false &
        
        MAIN_PID=$!
        
        # Start monitoring with the process PID
        monitor_resources "$exp_id"
        
        # Wait for the main process to complete
        wait $MAIN_PID
        
        # Stop monitoring
        stop_monitoring
    done
done

mkdir -p ./log/A100_scaling
mv ./log/*_A100_scaling_4.csv ./log/A100_scaling/
