#!/bin/bash
#$ -N DMSE
#$ -q free64
#$ -m beas

./run.sh $1 > log{$1}.txt