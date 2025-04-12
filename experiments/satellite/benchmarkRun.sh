#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=batch
#SBATCH -J benchmark_data
#SBATCH -o benchmark_data.%J.out
#SBATCH -e benchmark_data.%J.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=400G # try larger memory

N=1800000
N_TEST=200000
BlockCount=(18000)
BlockCount_TEST=(40000)
NN_est=(100 200 400) #
NN_pred=(200 400 600)
DIM=8   
kernel_type=Matern72
maxeval=(1000 1000 1000)

# make clean && make -j
# export OMP_DISPLAY_AFFINITY=TRUE

SPECIES=("N2" "N" "O" "He" "O2" "H") 

# SPECIES=("N2") 

# estimation
for species in ${SPECIES[@]}
do
    for fold in {1..10}
    do
        train_metadata_path="./HST/${species}/hst${species}_fold${fold}_train.csv"
        test_metadata_path="./HST/${species}/hst${species}_fold${fold}_val.csv"
        theta_init="1,0"
        distance_scale_init="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
        distance_scale="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
        
        for bc_est in ${BlockCount[@]}
        do
            for nn_est in ${NN_est[@]}
            do
                echo "nn_est: $nn_est"
                        
                # run
                srun --cpus-per-task=40 ./bin/dbv --num_total_points "$N" \
                --num_total_blocks "$bc_est" \
                -m "$nn_est" \
                --omp_num_threads 40 \
                --theta_init "$theta_init" \
                --distance_scale "$distance_scale" \
                --distance_scale_init "$distance_scale_init" \
                --xtol_rel 1e-3 \
                --ftol_rel 1e-5 \
                --dim "$DIM" \
                --mode estimation \
                --maxeval 1000 \
                --train_metadata_path "$train_metadata_path" \
                --kernel_type "$kernel_type"\
                --seed "$fold"\
                --nn_multiplier 1000 \
                --log_append "$species"

                # read params for next run
                params_path="./log/theta_numPointsTotal1800000_numBlocksTotal${bc_est}_m${nn_est}_seed${fold}_isScaled1_${species}.csv"

                # Read the first line of the CSV file
                line=$(head -n 1 $params_path)

                # Extract the first two values
                theta_init=$(echo "$line" | cut -d',' -f1-2)

                # Extract the rest of the values
                distance_scale=$(echo "$line" | cut -d',' -f3-)
                distance_scale_init=$distance_scale
                echo "theta_init: $theta_init"
                echo "distance_scale: $distance_scale"
                echo "distance_scale_init: $distance_scale_init"
                echo "species: $species"
                echo "fold: $fold"
            done
        done
    done
    mkdir -p ./log/satellite/$species
    mv ./log/*_${species}.csv ./log/satellite/$species
done


# prediction
for species in ${SPECIES[@]}
do
    for fold in {1..10}
    do
        train_metadata_path="./HST/${species}/hst${species}_fold${fold}_train.csv"
        test_metadata_path="./HST/${species}/hst${species}_fold${fold}_val.csv"
       
        # estimated parameters
        for bc_est in ${BlockCount[@]}
        do        
            # for nn_est in ${NN_est[@]}
            for index_nn_est in {1..2}
            do
                nn_est=${NN_est[$index_nn_est]}
                nn_est_prev=${NN_est[$index_nn_est - 1]}
                echo "nn_est: $nn_est"

                params_path="./log/satellite/${species}/theta_numPointsTotal1800000_numBlocksTotal${bc_est}_m${nn_est}_seed${fold}_isScaled1_${species}.csv"
                params_path_prev="./log/satellite/${species}/theta_numPointsTotal1800000_numBlocksTotal${bc_est}_m${nn_est_prev}_seed${fold}_isScaled1_${species}.csv"

                # Read the first line of the CSV file
                line=$(head -n 1 $params_path)
                line_prev=$(head -n 1 $params_path_prev)

                # Extract the first two values
                theta_init=$(echo "$line" | cut -d',' -f1-2)
                theta_init_prev=$(echo "$line_prev" | cut -d',' -f1-2)

                # Extract the rest of the values
                distance_scale_init=$(echo "$line" | cut -d',' -f3-)
                distance_scale_prev=$(echo "$line_prev" | cut -d',' -f3-)
                distance_scale=$distance_scale_prev
                echo "theta_init: $theta_init"
                echo "theta_init_prev: $theta_init_prev"
                echo "distance_scale: $distance_scale"
                echo "distance_scale_prev: $distance_scale_prev"
                echo "distance_scale_init: $distance_scale_init"
                echo "species: $species"
                echo "fold: $fold"

                for bc_pred in ${BlockCount_TEST[@]}
                do
                    echo "bc_pred: $bc_pred"
                    for nn_pred in ${NN_pred[@]}
                    do
                        echo "nn_pred: $nn_pred"
                        # do prediction
                            srun --cpus-per-task=40 ./bin/dbv --num_total_points "$N" \
                            --num_total_points_test "$N_TEST" \
                            --num_total_blocks "$bc_est" \
                            --num_total_blocks_test "$bc_pred" \
                            -m "$nn_est" \
                            --m_test "$nn_pred" \
                            --omp_num_threads 40 \
                            --theta_init "$theta_init" \
                            --distance_scale "$distance_scale" \
                            --distance_scale_init "$distance_scale_init" \
                            --xtol_rel 1e-3 \
                            --ftol_rel 1e-5 \
                            --dim "$DIM" \
                            --mode prediction \
                            --train_metadata_path "$train_metadata_path" \
                            --test_metadata_path "$test_metadata_path" \
                            --kernel_type "$kernel_type"\
                            --seed "$fold"\
                            --nn_multiplier 200 \
                            --log_append "${species}_pred"

                    done
                done    
            done
        done
    done
    mkdir -p ./log/satellite/$species
    mv ./log/*_${species}_pred.csv ./log/satellite/$species
done