#!/bin/bash

#C=(5 10 15 20 25 30 40)
#SPREAD=(1 2 3.5 5 7.5 10)
C=(5 10)
SPREAD=(1 2)

# Check if number of particles is given
if [[ $1 -eq "" ]]
	then
		echo "Number of particles not given"
		exit 1
fi

for c in ${C[*]}
do
	for spread in ${SPREAD[*]}
	do
		python3 DarkMatter.py $1 $c $spread
	done
done





