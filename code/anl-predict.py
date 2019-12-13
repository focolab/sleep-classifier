#!/usr/bin/env python3
#
#   
#   each "model" has a unique training scheme and multiple classifiers
#
#   TMC (Trial Model Classifier) indexing:
#   ------
#   predictions 'y' are a function of indices T,M,C.
#
#   
#   classifiers
#       - LDA: linear discriminant analysis
#       - QDA: quadtratic discriminatn analysis
#       - OVO: one versus one
#       - OVR: one versus rest
#
#
#======================================
import os
import argparse
import json
import warnings
import pdb
import pickle
import itertools

import pandas as pd
import numpy as np

import scoreblock as sb
import remtools as rt
import modeltools as mt


#==============================================================================
#==============================================================================
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

    # load (preprocessed) data
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]


    # load models
    allModels = []
    for i, pf in enumerate(args.m):
        with open(pf,'rb') as infile:
            dd = pickle.load(infile)
        allModels.append(dd)

    print('TODO: models should be split into individual models and indexed')


    # useful information
    trial_names = [td.trial for td in allTrialData]
    model_names = [mm['tagDict']['tag'] for mm in allModels]
    classifier_names = list(allModels[0]['full_model']['classifiers'].keys())

    trials = dict(zip(trial_names, allTrialData))
    models = dict(zip(model_names, allModels))

    print('-------------------------')
    print('trial_names     :', trial_names)
    print('model_names     :', model_names)
    print('classifier_names:', classifier_names)
    print('-------------------------')

    # DATA STORAGE STRATEGIES for prediction data
    # Each TMC combo gives a vector 'y'
    #   nested dictionaries
    #       pred[trial][model][classifier] = y
    #   nested lists
    #       pred[T][M][C] = y
    #   dataframe: 
    #       hstack df_index and df_data, note the index and data columns
    #

    # # nested dictionaries: meehhhh
    # pred = {}
    # for nameT, td in trials.items():
    #     X = td.sxxb_prep.stack.T
    #     resT = {}
    #     for nameM, mdl in models.items():
    #         fmd = mdl['full_model']
    #         resM = {}
    #         for nameC in classifier_names:
    #             y = fmd['classifiers'][nameC]['cls'].predict(X)
    #             print(nameT, nameM, nameC)
    #             resM[nameC] = y
    #         resT[nameM] = resM
    #     pred[nameT] = resT


    # dataframes: df_index + df_pred = df_cat
    cols_index = ['T', 'M', 'C']
    index = itertools.product(trial_names, model_names, classifier_names)
    df_index = pd.DataFrame(data=index, columns=cols_index)

    data = []
    ndx = []
    for nameT, td in trials.items():
        X = td.features.data.T
        for nameM, mdl in models.items():
            fmd = mdl['full_model']

            # data transforms: standardize and pca
            sc = fmd['sc']
            pca = fmd['pca']
            Xm = X*1.0
            if sc is not None:
                Xm = sc.transform(Xm)
            if pca is not None:
                Xm = pca.transform(Xm)

            for nameC in classifier_names:
                classifier = fmd['classifiers'][nameC]['cls']
                y = classifier.predict(Xm)
                print(nameT, nameM, nameC)
                data.append(y)

                d = dict(
                    T=nameT,
                    M=nameM,
                    C=nameC,
                    genotype=td.tagDict.get('genotype','xx'),
                    day=td.tagDict.get('day','xx'),
                )
                ndx.append(d)

    data = np.asarray(data)
    cols_data = ['ep-%5.5i' % (i+1) for i in range(data.shape[1])]
    df_data = pd.DataFrame(data=data, columns=cols_data)


    # combined dataframe of raw predictions
    df_pred = pd.concat([df_index, df_data], axis=1)

    # BETTER version
    df_ndx = pd.DataFrame(data=ndx)
    dff = pd.concat([df_ndx, df_data], axis=1)
    sb_pred = sb.ScoreBlock(df=dff, index_cols=df_ndx.columns.tolist())
    sb_pred.to_json(os.path.join(args.dest, 'scoreblock-raw.json'))

    # make column masks, compute score fractions, stack
    num_epochs = data.shape[1]
    maskAM = slice(0, num_epochs//2)
    maskPM = slice(num_epochs//2, num_epochs)

    # sb_frac_all = sb_pred.count(maskname='24h', frac=True) 
    # sb_frac_am = sb_pred.count(mask=maskAM, maskname='12hAM', frac=True)
    # sb_frac_pm = sb_pred.count(mask=maskPM, maskname='12hPM', frac=True)
    sb_frac_all = sb_pred.mask(maskname='24h').count( frac=True) 
    sb_frac_am = sb_pred.mask(mask=maskAM, maskname='12hAM').count(frac=True)
    sb_frac_pm = sb_pred.mask(mask=maskPM, maskname='12hPM').count(frac=True)

    sb_stack_frac = sb_frac_all.stack(others=[sb_frac_am, sb_frac_pm])
    sb_stack_frac.to_json(os.path.join(args.dest, 'scoreblock-fractions.json'))


    # # combined dataframe of prediction counts
    # def get_label_counts(data=None):
    #     label_names = np.unique(data)
    #     label_counts = []
    #     for row in data:
    #         label_counts.append({x:row.tolist().count(x) for x in label_names})
    #     df_counts = pd.DataFrame(label_counts)
    #     return df_counts

    # label_names = np.unique(data).tolist()
    # num_epochs = data.shape[1]

    # df_pred_counts_all = pd.concat([df_index, get_label_counts(data=data)], axis=1)
    # df_pred_counts_am = pd.concat([df_index, get_label_counts(data=data[:,0:num_epochs//2])], axis=1)
    # df_pred_counts_pm = pd.concat([df_index, get_label_counts(data=data[:,num_epochs//2:])], axis=1)


    # # metadata
    # dd = dict(
    #     _about='sleep state predictions using different models and classifiers',
    #     loc=os.path.abspath(args.dest),
    #     label_names=label_names,
    #     trial_names=trial_names,
    #     model_names=model_names,
    #     classifier_names=classifier_names,
    #     df_pred_csv='df_pred.csv',
    #     df_pred_counts_all_csv='df_pred_counts_all.csv',
    #     df_pred_counts_am_csv='df_pred_counts_am.csv',
    #     df_pred_counts_pm_csv='df_pred_counts_pm.csv',
    #     cols_index=cols_index,
    #     cols_data=cols_data,
    # )

    # # export
    # out = os.path.join(args.dest, 'predictions.json')
    # with open(out, 'w') as jout:
    #     json.dump(dd, jout, indent=2, sort_keys=False)
    #     jout.write('\n')

    # # raw predictions
    # csv = os.path.join(args.dest, 'df_pred.csv')
    # df_pred.to_csv(csv)


    # csv = os.path.join(args.dest, 'df_pred_counts_all.csv')
    # df_pred_counts_all.to_csv(csv)

    # csv = os.path.join(args.dest, 'df_pred_counts_am.csv')
    # df_pred_counts_am.to_csv(csv)

    # csv = os.path.join(args.dest, 'df_pred_counts_pm.csv')
    # df_pred_counts_pm.to_csv(csv)

    # #exit()
