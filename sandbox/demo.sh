#!/bin/bash
#
#

set -e



# #==============================================================================
# #=========================== STAGING (A) ======================================
# #==============================================================================
# staging the human scoring data

# scorecsv="files-human-scores-A.csv"
# ../code/load-scores.py -c $scorecsv --dest ANL-load-human-scores-A
# scorecsv="files-human-scores-B.csv"
# ../code/load-scores.py -c $scorecsv --dest ANL-load-human-scores-B


# #### NOTE: have to manually edit files-blind.csv and files-400Hz-trn.csv

#==============================================================================
#=========================== TRAINING (A) =====================================
#==============================================================================

# # preprocess training data (A)
# flz="files-data-A-training.csv"
# ../code/anl-preprocess.py -c $flz -p param-staging.json --dest ANL-preprocess-A-train

# # pca of the training ensemble, for later use
# flz=$(ls ANL-preprocess-A-train/trial-*/staged-trial-data.json)
# ../code/anl-pca.py -f $flz --dest ANL-pca-A

# # model training (A)
# flz=$(ls ANL-preprocess-A-train/trial-*/staged-trial-data.json)
# ../code/anl-trainmodels.py -f $flz --dest ANL-models

# # make score predictions for training data (A)
# flz=$(ls ANL-preprocess-A-train/trial-*/staged-trial-data.json)
# mdl=$(ls ANL-models/training*/model.p)
# ../code/anl-predict.py -f $flz -m $mdl --dest ANL-predicted-scores-A

# # MERGE human/model predictions for training data (A)
# hh="ANL-load-human-scores-A/scoreblock-alldata-raw.json"
# mm="ANL-predicted-scores-A/scoreblock-raw.json"
# ../code/anl-merge-pred.py --hh $hh --mm $mm --dest ANL-merged-scores-A


# ------------------------
# TODO:
#   X stack human and model predictions
#   compute consensus scores ()
#   reduce to one score vector per trial/day

    
# # plot model training results (A)
# flz=$(ls ANL-models/training*/model.p)
# ../code/plt-trainmodels.py -f $flz --dest ANL-models/plt

# # pca 2D projections
# flz=$(ls ANL-preprocess-A-train/trial-*/staged-trial-data.json)
# ../code/plt-pca2d.py -f $flz -p ANL-pca-A/*json --dest ANL-pca-A

#==============================================================================
#=========================== PREDICTIONS (B) ==================================
#==============================================================================

# # # preprocess blinded data (B)
# # flz="files-data-B-GT12.csv"
# # ../code/anl-preprocess.py -c $flz -p param-staging.json --dest ANL-preprocess-B-blind

# # make predictions for blinded data (B)
# flz=$(ls ANL-preprocess-B-blind/trial-*/staged-trial-data.json)
# mdl=$(ls ANL-models/training*/model.p)
# ../code/anl-predict.py -f $flz -m $mdl --dest ANL-predicted-scores-B

# # MERGE predictions for blinded data (B)
# hh="ANL-load-human-scores-B/scoreblock-alldata-raw.json"
# mm="ANL-predicted-scores-B/scoreblock-raw.json"
# ../code/anl-merge-pred.py --hh $hh --mm $mm --dest ANL-merged-scores-B

# # plot predictions for blinded data (B) (human and model predictions)
# ../code/plt-scores.py -s ANL-merged-scores-B/scoreblock-raw-merged.json --dest ANL-merged-scores-B/pltv2

# pca 2D projections
flz=$(ls ANL-preprocess-B-blind/trial-*/staged-trial-data.json)
scores="ANL-merged-scores-B/scoreblock-raw-merged.json"
../code/plt-pca2d.py -f $flz -p ANL-pca-A/*json -s $scores --dest ANL-pca-B

# # JS-distance comparison
flz=$(ls ANL-preprocess-B-blind/trial-*/staged-trial-data.json)
flzb=$(ls ANL-preprocess-A-train/trial-*/staged-trial-data.json)
../code/anl-similarity.py -f $flz $flzb -p ANL-pca-A/*json --dest ANL-similarity-B



#### DEPRECATED
# # plot predictions for blinded data (B)
# flz=$(ls ANL-predictions/predictions.json)
# ../code/plt-predict.py -f $flz -c files-blind.csv --dest ANL-predictions/plt


