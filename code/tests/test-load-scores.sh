#/bin/bash

set -e

data=../../data

#flz=$(ls $data/Scores_with_multiple_scorers/{335,336,579}scores_*)
flz=$(ls $data/Scores_with_multiple_scorers/*scores_*)
../load-scores.py -f $flz --dest ANL-load-scores-A


flz=$(ls $data/blind_edf/scores*/{404,506,405}*)
../load-scores.py -f $flz --dest ANL-load-scores-B

