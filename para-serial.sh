#!/bin/bash
#$ -N DMSE
#$ -q free*,pub64
#$ -pe openmp 8-64

./run.sh $1 $2 $3> log$1-$2-$3.txt