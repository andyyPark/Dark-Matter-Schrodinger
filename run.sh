#!/bin/bash

C=(5 10 15 20 25 30 40)
SPREAD=(1 2 3.5 5 7.5 10)

# Check if number of particles is given
if [[ $1 -eq "" ]]; then
	echo "Number of particles not given"
	exit 1
fi

# Run python script
for c in ${C[*]}
do
	for spread in ${SPREAD[*]}
	do
		python3 DarkMatter.py $1 $c $spread
	done
done

# Move all the iamges
if [[ -d "./images" ]]; then
	cd ./images 

	if [[ -d "n$1" ]]; then
		cd ../
		mv *.png ./images/n$1
			
	else
		mkdir n$1
		cd ../
		mv *.png ./images/n$1
	fi

else
	mkdir images
	cd ./images
	mkdir n$1
	cd ../
	mv *.png ./images/n$1
fi




