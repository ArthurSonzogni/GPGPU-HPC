#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

folder="Sequential HPC_OpenMP HPC_MPI_GRID HPC_MPI GPGPU_simple GPGPU_Grid"

steps=10

for agent in 1000 2000 3000 4000
do
    echo "+――――――――――――――――――――――――――+"
    echo "|   agent = "$agent
    echo "+――――――――――――――――――――――――――+"
    for i in $folder
    do
        echo "    +――――――――――――――――――――――――――+"
        echo "    |   program = "$i
        echo "    +――――――――――――――――――――――――――+"
        cd $here/../$i
        cmakeResult=$(cmake -DRUN_ARGS="-write;0;-agents;"$agent";-steps;"$steps .)
        result=$(make run)
    done
done
