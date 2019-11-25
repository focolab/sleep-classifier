#!/bin/bash

set -e

flz=$(ls ../../sandbox/ANL-preprocess-A-train/trial-3*/staged*json)
#../anl-pca.py -f $flz --dest ANL-pca-A

../plt-pca2d.py -f $flz -p ANL-pca-A/*json



