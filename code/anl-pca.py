#!/usr/bin/env python3
#
#   
#

import os
import argparse
import pdb

import numpy as np
import pandas as pd

import remtools as rt



if __name__ == '__main__':
    """pca for staged trial data features (>=1 trials)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='staged data json files')
    parser.add_argument('--dest', default='ANL-pca', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    print('#=================================================================')
    print('           anl-pca.py')
    print('#=================================================================')


    # staging
    # load trial data
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]
    Xcat = np.vstack([std.sxxb_prep.stack.T for std in allTrialData])
    df_index = allTrialData[0].sxxb_prep.to_dataframe().index.to_frame().reset_index(drop=True)

    # compute pca
    pca = rt.PCA.from_data(Xcat.T, df_index=df_index)
    pca.about()

    # project training data
    prj_data = []
    for std in allTrialData:
        df_prj = pca.project(std.sxxb_prep.stack, num_EV=3)
        df_prj['trial'] = [str(std.trial)]*len(df_prj)
        prj_data.append(df_prj)
    df_prj = pd.concat(prj_data, axis=0).reset_index(drop=True)

    d0 = allTrialData[0].sxxb_prep.stack
    pca.plotSummary(f=os.path.join(args.dest, 'plot-pca-summary.png'), data=d0)


    # dump
    out = os.path.join(args.dest, 'pca-test.json')
    pca.to_json(jf=out)

    csv = os.path.join(args.dest, 'df_pca_prj.csv')
    df_prj.to_csv(csv, float_format='%g')


    # WWRW
    # 1. perform pca on pooled data
    # 2. project points into top PCS (indexing including trial)
    # 3. export PCA (mu/vecs/vals)




