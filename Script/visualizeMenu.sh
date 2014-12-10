#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

echo ""

PS3="Programme ? "
select choix in \
   "Sequential" \
   "HPC"  \
   "HPC_OpenMP"  \
   "GPGPU"  \
   "GPGPU_simple"  \
   "Quitter"
do
   case $REPLY in
      1) ./visualize.sh Sequential ;;
      2) ./visualize.sh HPC ;;
      3) ./visualize.sh HPC_OpenMP ;;
      4) ./visualize.sh GPGPU ;;
      5) ./visualize.sh GPGPU_simple ;;
      6) exit ;;
      *) echo "Choix invalide"  ;;
   esac
done
