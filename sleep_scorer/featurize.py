#!/usr/bin/env python3

#
#   featurization schemes
#
#   powspec:
#       - computes the power spectrum for each epoch (no overlap)
#       - smooth, normalize, bandpass, stride
#
#   rmspow:
#       - bandpass filter and compute block averaged rms power
#       - avoids DFT, numerically safer for short epochs
#       - rms power is log scaled then standardized (zero mean, unit variance)
#

import pdb

import pandas as pd
import numpy as np
from scipy import signal

import sleep_scorer.scoreblock as sb
from sleep_scorer import tsm1d


def Featurizer(object):
    """serializable
    """
    def __init__(self, params={}):
        """pretty much just parameter parsing"""
        pass
    
    def featurize(self, edf=None):
        """returns a scoreblock"""
        pass

    def default_params(self):
        """"""
        pass


def make_powspec_params():
    """defaults"""
    pEEG = dict(lowpass=20,  highpass=2,  logscale=False, normalize=True, medianfilter=9, stride=5)
    pEMG = dict(lowpass=100, highpass=130, logscale=False, normalize=True, medianfilter=9, stride=10)
    dd = dict(spectrogram=dict(EEG=pEEG, EMG=pEMG))
    return dd


def compute_powspec_features(edfd=None, params={}):
    """compute spectrogram based features
    
    WWRW: spectrogram specific things are contained here, features used
        elsewhere are not assumed to derive from spectrograms

    TODO: rename powspec

    input
    ------
    edfd (EDFData)
    params

    output
    ------
    scoreblock

    """

    # defaults
    default_params = make_powspec_params()
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

def make_rmspow_params():
    """defaults"""
    p1 = dict(
        tag='EMG_mfw00',
        channel='EMG',
        epochLength=10,
        medianFilterWidth=0,
        lowcut=100,
        highcut=130
        )
    p2 = dict(
        tag='EMG_mfw90',
        channel='EMG',
        epochLength=10,
        medianFilterWidth=90,
        lowcut=100,
        highcut=130
        )
    dd = dict(rmspower=[p1, p2])
    return dd


def compute_rmspow_features(edfd=None, params=None):
    """compute rms power features

    TODO: rename rmspow
    TODO: extract the inner function
    TODO: for this to work as intented, every featurization needs to use the
          same epochLength.

    - params contains a list of parameter dicts
    - each dict generates one RMSpower featurization
    - this allows for chopping input signals into diffrent bands
    - the output is a scoreblock of stacked feature row vectors (not really
        scores, but its a useful data container)

    input
    ------
    edfd (EDFData)
    params

    output
    ------
    scoreblock
    """


    if params is None:
        params = make_rmspower_params()['rmspower']

    scoreblocks = []
    for pp in params:

        print('--------------')
        print('rmspower featurization:')
        print(pp)

        tag = pp['tag']
        channel = pp['channel']
        epochLength = pp['epochLength']
        mfw = pp['medianFilterWidth']
        lowcut = pp['lowcut']
        highcut = pp['highcut']

        s = edfd.signal_traces[channel]

        # bandpass
        s = s.bandpass(lowcut=lowcut, highcut=highcut)

        # window (array) size
        wsize = int(s.f*epochLength)

        # median filter width (array)
        mfwidth = int(np.floor(mfw/epochLength/2)*2+1)

        # compute rms power, epoch block averaged
        stsq = s.sig**2
        rmspow = np.sqrt(tsm1d.block_avg(stsq, wsize))
        logpow = np.log(rmspow)

        # standardize
        logpow -= np.mean(logpow)
        logpow /= np.std(logpow)

        # time values (block centers)
        tval = (np.arange(len(logpow))+0.5)*epochLength

        # the result
        logpowmf = signal.medfilt(logpow, kernel_size=[mfwidth])

        # build a scoreblock (of features)
        index = dict(
            tag=tag,
            medianFilterWidth=mfw,
            epochLength=epochLength,
            channel=channel
        )

        data_cols = ['epoch-%5.5i' % (i+1) for i in range(len(logpowmf))]
        index_cols = [k for k in index.keys()]
        data = [v for v in index.values()] + logpowmf.tolist()
        df = pd.Series(index=index_cols+data_cols, data=data).to_frame().T
        scb = sb.ScoreBlock(df=df, index_cols=index_cols)

        scoreblocks.append(scb)

    if len(scoreblocks) >1:
        fblk = scoreblocks[0].stack(others=scoreblocks[1:])
    else:
        fblk = scoreblocks[0]

    return fblk



if __name__ == "__main__":

    pass