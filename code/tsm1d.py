
import pdb
import os

import pandas as pd
import numpy as np
from scipy import signal


class TimeSeriesModel1D(object):
    """1D time series model

    built with signal power analysis in mind

    given a time series, compute
        histogram
        two* state GMM classifier (low/switch/high)
        predicted scores


    how to include
    trial/day/signal/etc
    """

    def __init__(self, t, x, metaData={}):
        """
            
        input
        ------
        t (array) : time values [s]
        x (array) : data values

        tagDict : 
        metaData :
        """
        from sklearn import mixture

        # make a Gaussian Mixture Model (sklearn version)
        clf = mixture.GaussianMixture(n_components=2).fit(x.reshape(-1,1))

        # modified GMM classifier with a crossover (xx) region
        gmm2 = TwoStateGMMClassifier(clf=clf)

        self.t = t
        self.x = x
        self.gmm2 = gmm2
        #self.tagDict = tagDict
        self.metaData = metaData
            

    def hx(self, bins=50):
        """
        bins : as in np.histogram
        """
        # x histogram
        hx, be = np.histogram(self.x, bins=bins)
        hx = hx/len(self.x)
        #self.hx = hx
        #self.hx = hx
        #self.hx_be = be
        return hx


    def predict(self, pdiff=0.9):
        """
        use GMM to make predictions
        
        TODO: make this a scoreblock
        """

        scores = self.gmm2.predict(self.x , pdiff=pdiff)
        
        return scores
    
#         data_cols = ['epoch-%6.6i' % (i+1) for i in range(len(logpow))]
#         df_data = pd.DataFrame(data=[scores], columns=data_cols)

#         index_cols = ['classifier', 'medianFilterWidth']
#         index_data = zip(classifiers, mfws)
#         df_index = pd.DataFrame(data=index_data, columns=index_cols)

#         df_index['scoreType'] = ['EMGpowerGMM']*2
#         df_index['epochLength'] = [el]*2
#         df_index['GMM_pdiff'] = [pdiff]*2

#         df = pd.concat([df_index, df_data], axis=1)

#         scoreblock = sb.ScoreBlock(df=df, index_cols=df_index.columns.tolist()) #.applymap(names)

    
        


def block_avg(x, n, strict=True):
    """"""
    #     aa = np.asarray([1,1,1,2,2,2,3,3,3])    
    #     ba = block_avg(aa, 4, strict=False)
    #     print(ba)

    if len(x)%n != 0:
        if not strict:
            print('WARNING: n does not evenly divide len(x)')
        else:
            raise Exception('FAIL: n does not evenly divide len(x), exiting because strict=True')

    num_blocks = int(np.ceil(len(x)/n))
    ans = [np.mean(x[i*n:(i+1)*n]) for i in range(num_blocks)]

    return np.asarray(ans)



class TwoStateGMMClassifier(object):
    """slightly modified gaussian mixture model (1D, two states)
    
    This GMM has two states plus a crossover region
    
    Uses the input classifier (sklearn.mixture.GMM) with two modifications:
    
    1) To correct for spurious assignments at the data limits:
        if x < mu_0, then assign state0
        if x > mu_1, then assign state1
    
    2) Points close to the (inner) decision boundary are assigned the switch state
       if |p0-p1| < pdiff

    """
    def __init__(self, clf=None):
        """
        
        input
        ------
        clf : (sklearn.mixture.GMM) Two state, 1D GMM (wlog mu_0 < mu_1)
        """
        
        self.clf = clf
        self.means = clf.means_.ravel()
        self.covs = clf.covariances_.ravel()
        self.weights = clf.weights_.ravel()

        self.x0 = np.min(self.means)
        self.x1 = np.max(self.means)


    def xinterval(self, pdiff=0.9):
        """find the switch interval numerically"""

        xx = np.linspace(self.x0, self.x1, 1000)
        pred_probs = self.clf.predict_proba(xx.reshape(-1, 1))
        pred_probs_diff = abs(pred_probs[:,0]-pred_probs[:,1])

        ndx = np.where(pred_probs_diff < pdiff)[0]

        return xx[[ndx[0], ndx[-1]]]


    def predict(self, X, pdiff=0.9):
        """predict state

        input
        ------
        X : input data
        pdiff : classify as switch if |p0(x)-p1(x)| < pdiff

        output
        ------
        pred : np.array of predictions in {0,1,-1} (-1 for switching)
        """

        pred_probs = self.clf.predict_proba(X.reshape(-1, 1))
        pred_probs_diff = abs(pred_probs[:,0]-pred_probs[:,1])

        # find low/mid/high regions
        is_low = X < self.x0
        is_hgh = X > self.x1
        is_mid = ~is_low & ~is_hgh

        # find switch interval
        is_switch = is_mid & (pred_probs_diff < pdiff)

        # start with clf predictions, then correct low/high/switch regions
        pred = self.clf.predict(X.reshape(-1, 1))

        # ensure that low X and high X are labeled 0 and 1
        if np.diff(self.means)[0] < 0:
            pred = pred*-1+1

        pred[is_low] = 0
        pred[is_hgh] = 1
        pred[is_switch] = -1

        return pred


def make_1DModel(s=None, epochLength=10, mfw=0, verbose=False):
    """

    - signal processing
    - feature building (rms power)
    - builds a 1D time series model

    NOTE: processing steps here might need to be split apart
            (rms power, block averaging, median filtering, log scaling, mean subtraction)
    TODO: median filter width logic
    TODO: time values at block centers or edges?

    input
    ------
    s : (remtools.SignalTrace)
    epochLength (float) : length [s]
    mfw (float) : median filter width [s], should be an odd multiple of epochLength
    """
    
    # window (array) size
    wsize = int(s.f*epochLength)

    # median filter width (array)
    mfwidth = int(np.floor(mfw/epochLength/2)*2+1)

    # compute rms power, epoch block averaged
    stsq = s.sig**2
    rmspow = np.sqrt(block_avg(stsq, wsize))
    logpow = np.log(rmspow)

    # standardize
    logpow -= np.mean(logpow)
    logpow /= np.std(logpow)

    # time values (block centers)
    tval = (np.arange(len(logpow))+0.5)*epochLength

    # the result
    logpowmf = signal.medfilt(logpow, kernel_size=[mfwidth])

    if verbose:
        print('====================')
        print('mean logpow      :', np.mean(logpow))
        print('epoch length [s] :', epochLength)
        print('points/epoch     :', wsize)
        print('mfw_in [s]       :', mfw)
        print('mfw_used [s]     :', mfwidth*epochLength)
        print('mfw_used (epochs):', mfwidth)


    # tagDict = dict(
    #     epochLength=epochLength,
    #     medianFilterWidth=mfwidth*epochLength,
    # )

    metaData = dict(        
        epochLength=epochLength,
        medianFilterWidth=mfwidth*epochLength,
        bandpass=s.metaData.get('bandpass', {}),
        channel=s.label
    )

    tsmf = TimeSeriesModel1D(
        t=tval, 
        x=logpowmf, 
        #tagDict=tagDict,
        metaData=metaData
        )
    
    return tsmf