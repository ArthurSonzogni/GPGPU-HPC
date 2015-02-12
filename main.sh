#! /bin/bash

here=$(dirname $(pwd)/$0)
cd $here

echo ""

PS3="Que voulez vous ? "
select choix in \
   "Visualiser Graphiques" \
   "Benchmark algorithms"  \
   "Quitter"
do
   case $REPLY in
      1) ./Script/visualizeMenu.sh ;;
      2) ./Script/benchmark.sh ;;
      3) exit ;;
      *) echo "Choix invalide"  ;;
   esac
done
