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
cmake .
mkdir -p output
make run

# launch gnuplot
cd $here
mkdir -p output
outputFile="$here/output/$program".gif
dataFolder="../$program/output/"

gnuplot -e "outputFile=\"$outputFile\"" -e "dataFolder=\"$dataFolder\"" visualize.gp

# visualize the produced gif
eog $outputFile
