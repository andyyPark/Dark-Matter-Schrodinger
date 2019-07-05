#!/bin/bash

C=(5 10 15 20 25 30 40)
NUMBER='^[0-9]+$'
DECIAML='^[0-9]+(\.[0-9]+)?$'

# Check if number of particles is given
if ! [[ $1 =~ $NUMBER ]]; then
	echo "Number of particles not given"
	exit 1
fi

# Check if spread is given
if ! [[ $2 =~ $DECIAML ]]; then
	echo "Initial spread not given"
	exit 1
fi

# Run python script
for c in ${C[*]}
do
	echo "Running $c and $2"
	python3 DarkMatter.py $1 $c $2
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



