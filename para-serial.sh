#!/bin/bash
#$ -N DMSE
#$ -q free*,pub64
#$ -pe openmp 8-64
#$ -m beas

./run.sh $1 $2 > log{$1}.txt