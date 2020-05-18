#!/bin/bash

set -e

dest=ANL-rmspow-2Dmodels-B
data=../../../data/blind_edf/files-data-B-fewer.csv
params=param-feat-rmspow-2D.json

# stage the data
python ../anl-stage-edf-scores.py -c $data --dest $dest/staged

# featurize (SLOW)
../anl-preprocess.py -c $dest/staged/csv-staged-data.csv -p $params --dest $dest/features

# pca plots (NOT ACTUALLY PCA, just 2D joint distributions)
flz=$(ls $dest/features/trial*/staged*json)
../plt-pca2d.py -f $flz --dest $dest/plots-pca2D

exit 0
