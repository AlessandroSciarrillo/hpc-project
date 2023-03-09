#!/bin/bash

rm output.txt ;
echo "----[ BENCHMARKS ]" > output.txt ;


for i in {1..12}
do
    OMP_NUM_THREADS=$i ./omp-sph 1000 50 >> output.txt ;
    echo -e " " >> output.txt;
done

cat output.txt