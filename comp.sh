#!/bin/bash
lZero=3
lUct=5
zeros=(FLAT ZERO ZEROX)

for ((i=1; i<$lZero+1; i++)); do
    # time-invariant, zero vs zero
    for ((zi=1; zi<${#zeros[@]}+1; zi++)); do
        for ((zj=$zi+1; zj<${#zeros[@]}+1; zj++)); do
            sbatch fit.sh eval ${zeros[$zi-1]},$i ${zeros[$zj-1]},$i
        done
    done

    # # time-invariant, zero vs uct
    # for zero in "${zeros[@]}"; do
    #     # iterate over all zeros
    #     for zero2 in "${zeros[@]}"; do
    #         sbatch fit.sh eval $zero,$i $zero2,$i
    #     done

    #     for ((j=1; j<$lUct+1; j++)); do
    #         sbatch fit.sh eval $zero,$i UCT,$j
    #     done
    # done

    # time-variant, zero vs zero
    for ((j=$i+1; j<$lZero+1; j++)); do
        for zero in "${zeros[@]}"; do
            sbatch fit.sh eval $zero,$i $zero,$j
        done
    done
done


