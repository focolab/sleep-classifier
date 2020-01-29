#!/usr/bin/env python3
"""
Testing a 1D two-state (unsupervised) GMM classifier

The motivation for this simple scheme was to see how well the EMG RMS power
could predict Wake/Sleep states, assuming REM is folded into Sleep.

This 1D two-state GMM scheme is applied (independently) to each feature in the
incoming std
"""

import json
import argparse
import os
import pdb

import pandas as pd
import numpy as np

import tsm1d
import remtools as rt
import scoreblock as sb
from tracedumper import TraceDumper


def two_state_prediction(std=None, pdiff=0.95, scoremap_hum={}, scoremap_gmm={}):
    """
    - build a two state (1D) GMM classifier for each feature
    - predict scores
    - map human predictions to two states
    - build scoreblock of human and model scores

    input
    ------

    returns
    ------


    examples:
    scoremap_gmm = {-1:'Switch', 0:'Sleep', 1:'Wake'}
    scoremap_hum = {'Non REM':'Sleep', 'REM':'Sleep'}
    """

    features = std.features
    scoreblock = std.scoreblock
    tagDict = std.tagDict
    X = features.data

    # for each feature, build GMM and predict scores
    ndx, data = [], []
    for i, row in features.df_index.iterrows():

        print(tagDict, row['tag'])
        # print(features.df_index)

        # GMM model scores
        myGMM = tsm1d.TwoStateGMMClassifier.from_data(X[i])
        data.append(myGMM.predict(X[i], pdiff=pdiff))

        # indexing
        dd = {k:v for k,v in tagDict.items()}
        dd.update(row)
        dd['scoreType'] = 'model'
        dd['classifier'] = 'TwoStateGMM'
        dd['pdiff'] = pdiff
        dd['scoreTag'] = dd['tag']
        ndx.append(dd)

    # make a scoreblock
    df_index = pd.DataFrame(data=ndx)
    df_data = pd.DataFrame(data, columns=scoreblock.data_cols)
    df_model_scores = pd.concat([df_index, df_data], axis=1)

    sb_model = sb.ScoreBlock(df=df_model_scores, index_cols=df_index.columns.tolist())
    sb_model = sb_model.applymap(scoremap_gmm)

    sb_human = scoreblock.applymap(scoremap_hum)
    sb_human.df['scoreTag'] = ['hum-%s' % xx for xx in sb_human.df['scorer']]
    sb_human.index_cols += ['scoreTag']        


    sb_out = sb_model.stack(others=[sb_human])

    return sb_out

def two_state_confusion(sb_gt=None, sb_models=None):
    """two state classifier confusion matrices

    compute confusion matrix for sb_gt (ground truth) vs each row sb_models

    input
    ------
    sb_gt : ScoreBlock
        ground truth (only one row)
    sb_models : ScoreBlock
        model predictions (>=1 rows)

    returns
    ------
    confusion : dict
        keyed by 'scoreTag' for each model

    """
    import sklearn
    from sklearn.metrics import confusion_matrix

    if sb_gt is None:
        raise Exception('sb_gt is required')
    if sb_models is None:
        raise Exception('sb_models is required')

    # find unique labels
    labels_models = np.unique(sb_models.data).tolist()
    labels_gt = np.unique(sb_gt.data).tolist()
    labels_all = list(set(labels_models+labels_gt))

    # print(labels_models)
    # print(labels_gt)
    # print(labels_all)

    # for each model, get the confusion matrix
    confusion = {}
    for i, row in sb_models.df_index.iterrows():
        ygt = sb_gt.data.ravel()
        ymd = sb_models.data[i]
        cnf_count = confusion_matrix(ygt, ymd, labels=labels_all)
        dd = dict(cnf=cnf_count.tolist(), labels=labels_all)

        # print(cnf_count)
        confusion[row['scoreTag']] = dd

    # print('----------------')
    # print(confusion)

    return confusion


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='staged data json files')
    parser.add_argument('--dest', default='ANL-1D-2state-GMM-pred', help='output folder')
    args = parser.parse_args()

    pdiff = 0.1
    scoremap_gmm = {-1:'Switch', 0:'Sleep', 1:'Wake'}
    scoremap_hum = {'Non REM':'Sleep', 'REM':'Sleep'}

    os.makedirs(args.dest, exist_ok=True)

    print('#=================================================================')
    print('           anl-two-state-predictions.py')
    print('#=================================================================')

    # load features and scores
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]

    for std in allTrialData:

        #-------------------------------------
        # BOOKKEEPING
        # name tag, output folder
        tagDict = std.tagDict
        trial = tagDict['trial']
        day = tagDict['day']

        trialDayTag = 'trial-%s-day-%s' % (trial, day)

        dest = os.path.join(args.dest, trialDayTag)
        os.makedirs(dest, exist_ok=True)

        #-------------------------------------
        # PREDICTIONS
        sb_out = two_state_prediction(
            std=std,
            pdiff=pdiff,
            scoremap_hum=scoremap_hum,
            scoremap_gmm=scoremap_gmm,
            )

        # EXPORT: scoreblock (TODO: classifiers)
        jf = 'gmm-scoreblock-%s.json' % (trialDayTag)
        sb_out.to_json(os.path.join(dest, jf))

        # optional visualization
        # tracedump
        # td = TraceDumper(std=std, scores=sb_out)
        # td.render_page(page=0)
        # png = 'trace-dump-%s.png' % (trialDayTag)
        # td.export(filename=os.path.join(dest, png))
        # td.resetRC()


        #-------------------------------------
        # ACCURACY        
        # munging: split scores into models and ground truth
        sb_models = sb_out.keeprows(conditions=[('scoreType', 'model')])
        sb_gt = sb_out.keeprows(conditions=[('scoreTag', 'hum-consensus')])

        # munging: mask to drop epochs where gt 'XXX'
        mask = sb_gt.data.ravel() != 'XXX'
        sb_models = sb_models.mask(mask=mask)
        sb_gt = sb_gt.mask(mask=mask)

        # main:
        cc = two_state_confusion(sb_gt=sb_gt, sb_models=sb_models)

        
        for k,v in cc.items():
            cnf = np.asarray(v['cnf'])
            acc = np.sum(np.trace(cnf)/np.sum(cnf.ravel()))
            #print(k, v)
            print('k/ACC:  %s  %5g' % (k, acc))
        #print(cc)



        # EXPORT: cc
        jf = 'gmm-confusion-%s.json' % (trialDayTag)
        with open(os.path.join(dest, jf), 'w') as jout:
            json.dump(cc, jout, indent=2, sort_keys=False)
            jout.write('\n')



        # optional viz
        import plottools as pt
        import matplotlib
        import matplotlib.pyplot as plt

        nmax = 5000

        num_cls = len(cc.items())
        fig = plt.figure(figsize=(3*num_cls, 6), dpi=300)
        ax_top = [plt.subplot(2, num_cls, i+1) for i in range(num_cls)]
        ax_bot = [plt.subplot(2, num_cls, num_cls+i+1) for i in range(num_cls)]

        for i, (k,v) in enumerate(cc.items()):
            cnf_raw = np.asarray(v['cnf'])

            kwa = dict(
                classes=v['labels'],
                title=k,
                cbar=False,
                colorkwa=dict(fraction=0.04),
                cmap=plt.cm.Blues
                )

            pt.plot_confusion_matrix(
                ax=ax_top[i],
                cm=cnf_raw, 
                imkwa={'vmax':nmax},
                normalize=False,
                **kwa
                )
            pt.plot_confusion_matrix(
                ax=ax_bot[i],
                cm=cnf_raw,
                imkwa={'vmax':100*nmax/np.sum(cnf_raw.ravel())},
                normalize=True,
                **kwa
                )

            # gussy it up
            ax_top[i].set_xticklabels([])
            ax_top[i].set_xlabel('')
            ax_bot[i].set_title('')
            
            if i>0:
                ax_top[i].set_ylabel('')
                ax_top[i].set_yticklabels([])
                ax_bot[i].set_ylabel('')
                ax_bot[i].set_yticklabels([])


        # txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        # fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(trialDayTag)
        plt.savefig(os.path.join(dest, 'plot-confusion.png'))


















