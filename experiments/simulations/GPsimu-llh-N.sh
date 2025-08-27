# !/bin/bash
range=(0.026270 0.017512 0.014290 0.078809 0.052537 0.042869 0.210158 0.140098 0.114318)
nu=(0.5 1.5 2.5 0.5 1.5 2.5 0.5 1.5 2.5)
n_array=(1 2 3 4 5 6 7 8 9)
precision=(float double)

N=(1000 5000 10000 20000 40000 80000 160000 320000 640000)
bc=(10 50 100 200 400 800 1600 3200 6400)
nlocs_array=(1 2 3 4 5 6 7 8 9)
m_bv=(120)

DIM=(2 4 8)

for p in ${precision[@]}; do
    for n in ${n_array[@]}; do 
        for _nlocs in ${nlocs_array[@]}; do
            for d in ${DIM[@]}; do
                _N=$((N[_nlocs-1]))
                _bc=$((bc[_nlocs-1]))
                range_init="${range[$((n-1))]}"
                nu_init="${nu[$((n-1))]}"
                if (( $(echo "$nu_init == 0.5" | bc -l) )); then
                    kernel_type="Matern12"
                elif (( $(echo "$nu_init == 1.5" | bc -l) )); then
                    kernel_type="Matern32"
                elif (( $(echo "$nu_init == 2.5" | bc -l) )); then
                    kernel_type="Matern52"
                fi
                echo $_N $_bc $d $kernel_type $p $nu_init
                _distance_scale=$(printf "%s," $(yes $range_init | head -n $d) | sed 's/,$//')
                ./bin/dbv --num_total_points $_N \
                --num_total_blocks $_bc \
                -m $m_bv \
                --dim $d \
                --mode estimation \
                --maxeval 1 \
                --theta_init 1.0,0.0 \
                --distance_scale $_distance_scale \
                --distance_scale_init $_distance_scale \
                --kernel_type $kernel_type \
                --nn_multiplier 999999 \
                --log_append "kl-$kernel_type-simu-$p-$n-dim$d-nlocs$_nlocs" \
                --precision $p
            done
        done
    done
done

mkdir -p ./log/kl-matern-simu-scaling
mv ./log/*.csv ./log/kl-matern-simu-scaling