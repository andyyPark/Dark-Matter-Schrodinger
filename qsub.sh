#!/bin/bash
C=(5.0 10.0 15.0 20.0 25.0 30.0 40.0)
SPREAD=10

for c in ${C[*]}
do
    qsub -ckpt blcr ./para-serial.sh $1 $c $SPREAD
done