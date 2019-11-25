#!/bin/bash

set -e

flz=$(ls ../../sandbox/ANL-preprocess-A-train/trial-3*/staged*json)

../anl-similarity.py -f $flz -p ANL-pca-A/*json --dest ANL-test-similarity



