#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

echo ""

PS3="Programme ? "
select choix in \
   "Sequential" \
   "HPC"  \
   "GPGPU"  \
   "Quitter"
do
   case $REPLY in
      1) ./visualize.sh Sequential ;;
      2) ./visualize.sh HPC ;;
      3) ./visualize.sh GPGPU ;;
      4) exit ;;
      *) echo "Choix invalide"  ;;
   esac
done
