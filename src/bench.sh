#!/bin/bash

echo "----[ BENCHMARKS ]" > output.txt ;


for i in {1..6}
do
    #OMP_NUM_THREADS=$i ./omp-sph 10000 50 2>&1 | tee -a output.txt ;
    mpirun -n $i ./mpi-sph 10000 50 2>&1 | tee -a output.txt ;
done

