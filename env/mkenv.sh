#!/bin/bash
#
#   makes a conda environment   
#   
#====================================================================
tag=$1 || { echo "ERROR: need env name as first arg" ; exit 1; }

# use these packages
[[ $tag ]] || { echo "error, need to pass a nametag as first argument" ; exit 1; }
[[ $tag == 'mouse1' ]] && { pkg="scipy matplotlib pandas pip ipython jupyter seaborn scikit-learn python=3.7"; }
[[ $pkg ]] || { echo "you need to set up the list of options for environment $tag"; exit 1; }

conda create -n $1 $pkg
source activate $1

conda install -c conda-forge jupyterlab
conda install -c plotly plotly 
conda install -c conda-forge pyedflib 

# conda install -c samoturk pymol=1.8.6.0
# conda install -c conda-forge pmw   #== for use with python 3

exit 0
