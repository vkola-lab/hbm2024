#!/bin/bash -l

# algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx')
FOLDS=(0 1 2 3 4)
for FOLD in ${FOLDS[@]}; do
    qsub -N $1_${FOLD} scripts/run.sh $2 $FOLD
done