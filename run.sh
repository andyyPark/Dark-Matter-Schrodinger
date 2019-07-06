#!/bin/bash

NUMBER='^[0-9]+$'
DECIAML='^[0-9]+(\.[0-9]+)?$'

# Check if number of particles is given
if ! [[ $1 =~ $NUMBER ]]; then
	echo "Number of particles not given"
	exit 1
fi

# Check if c is given
if ! [[ $2 =~ $DECIAML ]]; then
	echo "C not given"
	exit 1
fi

# Check if spread is given
if ! [[ $3 =~ $DECIAML ]]; then
	echo "Initial spread not given"
	exit 1
fi

# Run python script

echo "Running $2 and $3"
python3 DarkMatter.py $1 $2 $3


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



