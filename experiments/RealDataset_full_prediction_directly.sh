#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=16        
#SBATCH --gpus-per-node=4  
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH -J RealDataset_full_prediction
#SBATCH -o RealDataset_full_prediction.%J.out
#SBATCH -e RealDataset_full_prediction.%J.err
#SBATCH --time=4:00:00
#SBATCH -A jureap137

N=45000000
N_TEST=5000000
BlockCount=(450000)
BlockCount_TEST=(250000)
NN_est=(100 200 400)
NN_pred=(100 200 400 600) # 
DIM=10
kernel_type=Matern72

seed=42
DATA_DIR="./log"

train_metadata_path="/p/fscratch/jureap137/data/train_combined.txt"
test_metadata_path="/p/fscratch/jureap137/data/test_combined.txt"

theta_init=2.0,0.1
distance_scale=0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2
echo "theta_init: $theta_init"
echo "distance_scale: $distance_scale"

for bc_est in ${BlockCount[@]}
do
    for index in $(seq 0 2)
    do
        nn_est=${NN_est[$index]}
        echo "nn_est: $nn_est"
        params_path="$DATA_DIR/RealDataset/theta_numPointsTotal${N}_numBlocksTotal${bc_est}_m${nn_est}_seed${seed}_isScaled1_RealDataset.csv"
        echo "params_path: $params_path"

        # Read the first line of the CSV file
        line=$(head -n 1 $params_path)
        # Extract the rest of the values
        theta_init=$(echo "$line" | cut -d',' -f1-2)
        distance_scale_init=$(echo "$line" | cut -d',' -f3-)
        echo "distance_scale_init: $distance_scale_init"
        # prediction
        for bc_pred in ${BlockCount_TEST[@]}
        do
            echo "bc_pred: $bc_pred"
            for nn_pred in ${NN_pred[@]}
            do
                echo "nn_pred: $nn_pred"
                srun --exclusive ./bin/dbv --num_total_points "$N" \
                    --num_total_points_test "$N_TEST" \
                    --num_total_blocks "$bc_est" \
                    --num_total_blocks_test "$bc_pred" \
                    -m "$nn_est" \
                    --m_test "$nn_pred" \
                    --omp_num_threads 72 \
                    --theta_init "$theta_init" \
                    --distance_scale "$distance_scale" \
                    --distance_scale_init "$distance_scale_init" \
                    --dim "$DIM" \
                    --mode prediction \
                    --train_metadata_path "$train_metadata_path" \
                    --test_metadata_path "$test_metadata_path" \
                    --kernel_type "$kernel_type"\
                    --seed "$seed" \
                    --log_append RealDataset_prediction \
                    --nn_multiplier 50
            done
        done
        distance_scale=$distance_scale_init
        echo "distance_scale: $distance_scale"
    done
done

mkdir -p ./log/RealDataset
mv ./log/*RealDataset_prediction.csv ./log/RealDataset/
