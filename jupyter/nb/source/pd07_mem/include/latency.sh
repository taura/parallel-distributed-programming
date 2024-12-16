#!/bin/bash
#export OMP_TARGET_OFFLOAD=DISABLED
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_TEAMS=1
export OMP_NUM_THREADS=1
min_m=100
max_m=$((20 * 1000 * 1000))

m=${min_m}
while [ ${m} -lt ${max_m} ]; do
    echo "==== m=${m} ===="
    ./exe/latency_clang++_omp.exe --n-elements ${m} --seed -1
    m=$((m * 3 / 2))
done
      
