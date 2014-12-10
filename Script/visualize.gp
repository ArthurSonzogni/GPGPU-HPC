set term gif animate delay 10 size 400, 400
set output outputFile
do for [n=0:499:1] {
    splot [0:1][0:1][0:1] dataFolder."boids_".n.".xyz" u 1:2:3  t sprintf("n=%i", n) 
}
