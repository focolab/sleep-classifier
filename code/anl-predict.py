#!/usr/bin/env python3
import os
import argparse
import pdb
import pickle
import itertools

import pandas as pd
import numpy as np

import scoreblock as sb
import remtools as rt

def predict_scores(std=None, model=None):
    """use a trained classifier (model) to predict scores

    Each model can have multiple classifiers (OVO/OVR/LDA/QDA etc..)

    input
    ------
    std : StagedTrialData
        Featurized data
    model : dict
        a rather disorganized dict, created by anl-trainmodels.py

    returns
    ------
    sb_stk : ScoreBlock
        The predicted scores and the corresponding human scores (if they exist)
    """
    nameT = std.trial
    day = std.tagDict.get('day','xx')
    genotype = std.tagDict.get('genotype','xx')
    nameM = model['tagDict']['tag']
    fmd = model['full_model']
    classifier_names = list(fmd['classifiers'].keys())

    # features to predict
    X = std.features.data.T

    # data transforms: standardize and pca
    sc = fmd['sc']
    pca = fmd['pca']
    Xm = X*1.0
    if sc is not None:
        Xm = sc.transform(Xm)
    if pca is not None:
        Xm = pca.transform(Xm)

    data = []
    ndx = []
    for nameC in classifier_names:
        print('predicting:', nameT, nameM, nameC)
        classifier = fmd['classifiers'][nameC]['cls']
        data.append(classifier.predict(Xm))
        d = dict(trial=nameT, M=nameM, classifier=nameC, genotype=genotype, day=day)
        ndx.append(d)

    # make a scoreblock (joining index and data) of predicted model scores
    data = np.asarray(data)
    cols_data = ['ep-%5.5i' % (i+1) for i in range(data.shape[1])]
    df_data = pd.DataFrame(data=data, columns=cols_data)
    df_ndx = pd.DataFrame(data=ndx)
    dff = pd.concat([df_ndx, df_data], axis=1)
    sb_pred = sb.ScoreBlock(df=dff, index_cols=df_ndx.columns.tolist())
    sb_pred.add_const_index_col(name='scoreType', value='model', inplace=True)

    # stack human scores (if they exist) with model scores
    if std.scoreblock is not None:
        others = [std.scoreblock]
    else:
        others = []
    sb_stk = sb_pred.stack(others=others, force_data=True)

    return sb_stk


#==============================================================================
#==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='staged data json files')
    parser.add_argument('-m', nargs='+', type=str, help='(pickled) model files')
    parser.add_argument('--dest', default='ANL-predict', help='output folder')
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    print('#=================================================================')
    print('           anl-predict.py')
    print('#=================================================================')
    print('TODO: models should be split into individual models and indexed')
    print('TODO: models should be split into individual models and indexed')

    # load (featurized) data
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]

    # load models
    allModels = []
    for pf in args.m:
        with open(pf, 'rb') as infile:
            dd = pickle.load(infile)
        allModels.append(dd)

    # do predictions
    todo = itertools.product(allTrialData, allModels)
    scoreblocks = [predict_scores(std=std, model=mdl) for std,mdl in todo]

    # stack it all up and dump
    sb_stk = scoreblocks[0].stack(others=scoreblocks[1:])
    sb_stk.to_json(os.path.join(args.dest, 'scoreblock-hum-mod.json'))
