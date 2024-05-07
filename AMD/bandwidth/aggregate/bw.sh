#!/bin/bash

# Get the starting and ending values for l from the script parameters
start_l=$1

# First directory
cd globalMem/gpu-l2-cache

make

for ((i = 32; i < 33; i ++)){
    for ((l=start_l; l < start_l+1; l ++))
do
    ./hip-l2-cache $i $l > "$l"warps.log
done
}

# Change to the second directory
cd ../../L2/gpu-l2-cache/

make

for ((i = 32; i < 33; i ++)){
    for ((l=start_l; l < start_l+1; l ++))
do
    ./hip-l2-cache $i $l > "$l"warps.log
done
}