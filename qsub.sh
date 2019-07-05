#!/bin/bash

SPREAD=(1.0 2.0 3.5 5.0 7.5 10)

for spread in ${SPREAD[*]}
do
    qsub -ckpt blcr ./para-serial.sh $1 $spread
done