#!/bin/bash
set -e
export OMP_TARGET_OFFLOAD=DISABLED
#export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_NUM_TEAMS=1
export OMP_NUM_THREADS=1
min_m=100
max_m=$((1 << 25))
n=$((1 << 28))

m=${min_m}
while [ ${m} -lt ${max_m} ]; do
    echo "==== m=${m} ===="
    ./exe/latency_clang++_omp.exe --n-elements ${m} --min-accesses ${n}
    m=$((m * 5 / 4))
done
      
