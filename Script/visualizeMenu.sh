#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

echo ""

PS3="Programme ? "
select choix in \
   "Sequential" \
   "HPC_OpenMP"  \
   "HPC_MPI" \
   "HPC_MPI_GRID" \
   "GPGPU_simple"  \
   "GPGPU_Grid"  \
   "Quitter"
do
   case $REPLY in
      1) ./visualize.sh Sequential ;;
      2) ./visualize.sh HPC_OpenMP ;;
      3) ./visualize.sh HPC_MPI ;;
      4) ./visualize.sh HPC_MPI_GRID ;;
      5) ./visualize.sh GPGPU_simple ;;
      6) ./visualize.sh GPGPU_Grid ;;
      7) exit ;;
      *) echo "Choix invalide"  ;;
   esac
done
