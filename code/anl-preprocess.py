#!/usr/bin/env python3
#
#
#   load data (EDF files and scores)
#   build features (spectrograms, filtering, smoothing, striding)
#   
#   TODO: generalize the featurization process
#
#======================================
import os
import argparse 
import json
import pdb

import pandas as pd
import numpy as np

import remtools as rt
import scoreblock as sb


def make_spectrogram_params():
    """defaults"""
    pEEG = dict(lowpass=20,  highpass=2,  logscale=False, normalize=True, medianfilter=9, stride=5)
    pEMG = dict(lowpass=100, highpass=30, logscale=False, normalize=True, medianfilter=9, stride=10)
    dd = dict(spectrogram=dict(EEG=pEEG, EMG=pEMG))
    return dd


def compute_spectrogram_features(edfd=None, params={}):
    """compute spectrogram based features
    
    WWRW: spectrogram specific things are contained here, features used
        elsewhere are not assumed to derive from spectrograms

    input
    ------
    edfd (EDFData)
    params

    output
    ------
    scoreblock

    """

    # defaults
    default_params = make_spectrogram_params()
    pEEG = default_params['spectrogram']['EEG']
    pEMG = default_params['spectrogram']['EMG']

    pEEG.update(params.get('EEG', {}))
    pEMG.update(params.get('EMG', {}))

    # preprocess each spectrogram
    EEG1 = edfd.spectrograms['EEG1'].prep(pEEG).to_df
    EEG2 = edfd.spectrograms['EEG2'].prep(pEEG).to_df
    EMG = edfd.spectrograms['EMG'].prep(pEMG).to_df

    # build a scoreblock
    dd = dict(EEG1=EEG1, EEG2=EEG2, EMG=EMG)
    df = pd.concat(dd).reset_index().rename(columns={'level_0': 'channel'})
    index_cols = ['channel','f[Hz]']
    scb = sb.ScoreBlock(df=df, index_cols=index_cols)

    return scb

def compute_power_features():
    """rms power features"""
    pass


#=========================================================================================
#=============================== main program ============================================
#=========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, help='csv file of score/edf file pairs')
parser.add_argument('-p', type=str, required=True, help='preprocessing parameters param-staging.json')
parser.add_argument('--dest', type=str, default='ANL-preprocess', help='destination folder')
args = parser.parse_args()

os.makedirs(args.dest, exist_ok=True)

#== Staging Parameters
# TODO: what about different featurization schemes?
with open(args.p) as jfopen:
    jdic = json.load(jfopen)

#== load files
load = pd.read_csv(args.c)
print('=====================')
print(load)
print('=====================')
allTrialData = []
for index, row in load.iterrows():
    #== load edf and scores
    edf = rt.EDFData(edf=row['edf'])

    # scores
    try:
        scoreblock = sb.ScoreBlock.from_json(row['scores'])
    except:
        scoreblock = None

    # tags and metadata
    tagDict = dict()
    tagDict['trial'] = row.get('trial', 'trialXX')
    tagDict['genotype'] = row.get('genotype', 'x')
    tagDict['day'] = row.get('day', 1)

    # featurize
    if 'spectrogram' in jdic.keys():
        params = jdic['spectrogram']
        features_scb = compute_spectrogram_features(edfd=edf, params=params)
    else:
        raise Exception('params not recognized')


    fldr = os.path.join(args.dest, 'trial-%s' % (str(tagDict['trial'])))

    #== preprocess spectrograms, stage data for modeling
    std = rt.StagedTrialData(
        loc=fldr, 
        edf=edf,
        scoreblock=scoreblock,
        trial=tagDict['trial'],
        features=features_scb,
        stagingParameters=jdic,
        tagDict=tagDict
        )

    allTrialData.append(std)
    std.to_json()


#=========================================================================================
#== plot cleanup
#rt.plot_spectrogram_cleanup(allTrialData, out=os.path.join(args.dest, 'plot-sxx-cleanup.png'))



'''
#==============================================================================
#== PCA, and  project each distribution BROKEN, and factored out other places
# 
#   dimension reduction requires staged features. Scores are helpful
#


fusedFeat = rt.SxxBundle.fuse([std.sxxb_feat for std in allTrialData])
pca_all = fusedFeat.pca()
pca_all.plotSummary(f=os.path.join(args.dest, 'plot-pca-summary.png'))


#== merge data from multiple trials, for plotting
to_stack = []
for std in allTrialData:
    dfa = std.edf.dfTrialEpoch()
    dfb = pca_all.project(data=std.sxxb_feat.stack)
    dfc = std.sw.dfConsensus

    dfmerge = pd.concat([dfa, dfb, dfc], axis=1)
    print(dfmerge.head())
    to_stack.append(dfmerge)
    
dfstack = pd.concat(to_stack, axis=0)
dfstack = dfstack.loc[:,~dfstack.columns.duplicated()]

print(dfstack.head())

#== PCA plots. 2D and 3D plotly plots
cols = ['PC1','PC2']
figx = pt.scat2d(dfxyz=dfstack, xycols=cols, tagcol='cScoreStr', title='plot', height=800, width=800)
html = os.path.join(args.dest, 'pca-2d-all-scores.html')
pt.fig_2_html(figx, filename=html)

cols = ['PC1','PC2']
figx = pt.scat2d(dfxyz=dfstack, xycols=cols, tagcol='trial', title='plot', height=800, width=800)
html = os.path.join(args.dest, 'pca-2d-all-trials.html')
pt.fig_2_html(figx, filename=html)


cols = ['PC1','PC2','PC3']
figx = pt.scat3d(dfxyz=dfstack, xyzcols=cols, tagcol='cScoreStr', title='plot', height=800, width=1400)
html = os.path.join(args.dest, 'pca-3d-all.html')
pt.fig_2_html(figx, filename=html)


#== single trial analysis
for std in allTrialData:

    trial = std.trial
    fldr = std.loc
    sw = std.sw
    edf = std.edf
    scat = std.sxxb_feat.stack

    pca_projections = pca_all.project(data=scat)
    pca_projections['Epoch#'] = np.arange(1, np.shape(scat)[1]+1)

    #== merge data: PC projections and consensus score
    dfa = sw.dfConsensus.set_index('Epoch#')
    dfb = pca_projections.set_index('Epoch#')
    dfmerge = pd.concat([dfa, dfb], axis=1)
    dfmerge.reset_index(inplace=True)

    print(dfmerge.head(10))
    print(dfmerge.tail(10))

    #== PCA plots. 2D and 3D plotly plots
    cols = ['PC1','PC2']
    figx = pt.scat2d(dfxyz=dfmerge, xycols=cols, tagcol='cScoreStr', title='plot', height=800, width=800)
    html = os.path.join(fldr, 'pca-2d.html')
    pt.fig_2_html(figx, filename=html)

    cols = ['PC1','PC2','PC3']
    figx = pt.scat3d(dfxyz=dfmerge, xyzcols=cols, tagcol='cScoreStr', title='plot', height=800, width=1400)
    html = os.path.join(fldr, 'pca-3d.html')
    pt.fig_2_html(figx, filename=html)


    rt.plot_trial_chunks(edf=edf, sw=sw, dfmerge=dfmerge, chunksize=25, dest=fldr, chunkF=20)





'''


















