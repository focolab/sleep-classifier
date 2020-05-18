#!/bin/bash
#
# testing the 1D, two-state GMM (unsupervised) classifiers
# (Two State Model 1D) -> TSM1D
#

set -e

dest=ANL-rmspow-tsm1d 
params=param-feat-rmspow-EMG-EMG.json
data=../../example_data/files-data-A-training_only2.csv

#==============================================================================
# stage the edf and scores
python ../anl-stage-edf-scores.py -c $data --dest $dest/00-stage

# featurize (SLOW)
data=$dest/00-stage/csv-staged-data.csv
../anl-preprocess.py -c $data -p $params --dest $dest/10-features

# PCA, mainly for some visualization of feature space
std=$(ls $dest/10-features/trial*/staged*json)
# ../anl-pca.py -f $std --dest $dest/pca
../plt-pca2d.py -f $std --dest $dest/plots-pca2D

# build models and compute accuracy (confusion matrices)
flz=$(ls $dest/10-features/trial*/staged-trial-data.json)
../anl-two-state-predictions.py -f $flz --dest $dest/predicted-scores

exit 0