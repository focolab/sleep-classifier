#!/bin/bash

set -e



# features="../../sandbox/ANL-preprocess-A-train/trial-335/data-features.csv"
# scores="../../sandbox/ANL-preprocess-A-train/trial-335/data-scores.csv"

flz=$(ls ../../sandbox/ANL-preprocess-A-train/trial-3*/staged*json)
../plt-features.py -f $flz

# flz=$(ls ../../sandbox/ANL-preprocess-B*/trial-2*/staged*json)
# ../plt-features.py -f $flz --dest ANL-plt-features-B




