#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

folder="Sequential HPC_OpenMP HPC_MPI_GRID HPC_MPI"

for i in $folder
do
    cd $here/../$i
    cmake -DRUN_ARGS="-write;0;-agents;1024" .
    make
    time make run
done
