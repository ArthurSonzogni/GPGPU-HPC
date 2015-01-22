#! /bin/bash

# check command line argument
if [[ $# -ne 1 ]]
then
    echo "Usage: `basename $0` Sequential|GPGPU|HPC"
    exit 1
fi

program=$1

here=$(dirname $(pwd)/$0)
cd $here

# compile and run the program
cd ../$program
cmake -DRUN_ARGS="-write;1" .
mkdir -p output
make run

# launch visualize hook
if [ -f ./visualizeHook.sh ]
then
    echo "Executing visualise hook ..."
    ./visualizeHook.sh
fi

# launch gnuplot
cd $here
mkdir -p output
outputFile="$here/output/$program".gif
dataFolder="../$program/output/"

gnuplot -e "outputFile=\"$outputFile\"" -e "dataFolder=\"$dataFolder\"" visualize.gp

# visualize the produced gif
eog $outputFile
