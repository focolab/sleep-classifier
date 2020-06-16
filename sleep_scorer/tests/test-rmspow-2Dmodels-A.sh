#!/bin/bash

#
#   2D models using RMS power features for EEG and EMG
#

set -e



dest=ANL-rmspow-2Dmodels-A 
params=param-feat-rmspow-2D.json
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
# ../plt-pca2d.py -f $std -p $dest/pca/*json --dest $dest/plots-pca2D
../plt-pca2d.py -f $std --dest $dest/plots-pca2D

# model training
../anl-trainmodels.py -f $std --dest $dest/30-models

# plot model training results
models=$(ls $dest/30-models/training*/model.p)
../plt-trainmodels.py -f $models --dest $dest/30-models/plt

# make score predictions (for the trainig data) and plot them
../anl-predict.py -f $std -m $models --dest $dest/40-predicted-scores
../plt-scores.py -s $dest/40-predicted-scores/scoreblock-hum-mod.json --dest $dest/plots-predicted-scores
