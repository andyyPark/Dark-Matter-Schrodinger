#!/bin/bash
C=(5.0 10.0 15.0 20.0 25.0 30.0 40.0)
SPREAD=(1.0 2.0 3.5 5.0 7.5 10.0)

for c in ${C[*]}
do
    for spread in ${SPREAD[*]}
    do
        qsub -ckpt blcr ./para-serial.sh $1 $c $spread
    done
done