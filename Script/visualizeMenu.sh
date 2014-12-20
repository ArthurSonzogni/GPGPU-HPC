#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

echo ""

PS3="Programme ? "
select choix in \
   "Sequential" \
   "HPC_OpenMP"  \
   "HPC_MPI" \
   "GPGPU_simple"  \
   "Quitter"
do
   case $REPLY in
      1) ./visualize.sh Sequential ;;
      2) ./visualize.sh HPC_OpenMP ;;
      3) ./visualize.sh HPC_MPI ;;
      4) ./visualize.sh GPGPU_simple ;;
      5) exit ;;
      *) echo "Choix invalide"  ;;
   esac
done
