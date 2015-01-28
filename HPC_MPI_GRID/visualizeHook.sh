#! /bin/bash
here=$(dirname $(pwd)/$0)
cd $here/output

for file in boids_*_0.xyz
do
    i=$(echo $file | cut -d'_' -f2)
    rm boids_${i}.xyz
    cat boids_${i}_*.xyz > boids_${i}.xyz
    rm boids_${i}_*.xyz
done
