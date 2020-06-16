
import pdb
import os

import pandas as pd
import numpy as np
from scipy import signal
from sklearn import mixture
import matplotlib.pyplot as plt

import sleep_scorer.scoreblock as sb


class TimeSeriesModel1D(object):
    """1D time series model

    A 1D timeseries, and accompanying two-state GMM classifier
    """
    def __init__(self, t, x, metaData=None):
        """
        input
        ------
        t (array) : time values [s]
        x (array) : data values
        metaData (dict) : things like trial, day, etc
        """

        # make a Gaussian Mixture Model (sklearn version)
        clf = mixture.GaussianMixture(n_components=2).fit(x.reshape(-1,1))

        # modified GMM classifier with a crossover (xx) region
        gmm2 = TwoStateGMMClassifier(clf=clf)

        self.t = t
        self.x = x
        self.gmm2 = gmm2
        if metaData is None:
            self.metaData = {}
        else:
            self.metaData = metaData
            
    def hx(self, bins=50):
        """
        bins : as in np.histogram
        """
        # x histogram
        hx, be = np.histogram(self.x, bins=bins)
        hx = hx/len(self.x)
        return hx

    def predict(self, pmin=0.95):
        """use GMM to make predictions
        
        returns a ScoreBlock
        """
        scores = self.gmm2.predict(self.x , pmin=pmin)
    
        data_cols = ['epoch-%6.6i' % (i+1) for i in range(len(scores))]
        index = dict(pmin=pmin, scoreType='model', classifier='GMM_1D')
        index_cols = [k for k in index.keys()]
        index_vals = [v for v in index.values()]
        data = index_vals+[s for s in scores]

        # dataframe via series
        df = pd.Series(index=index_cols+data_cols, data=data).to_frame().T

        return sb.ScoreBlock(df=df, index_cols=index_cols)


def block_avg(x, n, strict=True):
    """"""
    if len(x)%n != 0:
        if not strict:
            print('WARNING: n does not evenly divide len(x)')
        else:
            raise Exception('FAIL: n does not evenly divide len(x), exiting because strict=True')

    num_blocks = int(np.ceil(len(x)/n))
    ans = [np.mean(x[i*n:(i+1)*n]) for i in range(num_blocks)]

    return np.asarray(ans)


class TwoStateGMMClassifier(object):
    """modified Gaussian mixture model (1D, two states)
    
    This GMM has two states plus a crossover region
    
    Uses a sklearn.mixture.GMM classifier with two modifications:
    
    1) To correct for spurious assignments at the data limits:
        if x < mu_0, then assign state0
        if x > mu_1, then assign state1
    
    2) Points close to the (inner) decision boundary are assigned the switch state
       if max(p0, p1) < pmin

    TODO: peak scaling transformation (two peaks at -1, 1)
    TODO: sklearn.mixture.GMM could be (de-)serialized by tracking:
            n_components
            precisions_init
            weights_init
            means_init
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

    @classmethod
    def from_data(cls, x):
        clf = mixture.GaussianMixture(n_components=2).fit(x.reshape(-1,1))
        return cls(clf=clf)

    def about(self):
        """"""
        print('--- TwoStateGMMClassifier.about() ---')
        print('means  :', self.means)
        print('weights:', self.weights)

    def xinterval(self, pdiff=None, pmin=0.95):
        """find the switch interval numerically

        The crossover interval between the two Gaussians, where the maximum
        classification probability is below the threshold pmin.

        0.5<pmin<1
        """
        if pdiff is not None:
            print('DEPRECATION WARNING: use pmin not pdiff')
            pmin = pdiff +(1-pdiff)/2.

        xx = np.linspace(self.x0, self.x1, 1000)
        pred_probs = self.clf.predict_proba(xx.reshape(-1, 1))
        pred_probs_max = np.maximum(pred_probs[:,0], pred_probs[:,1])
        ndx = np.where(pred_probs_max < pmin)[0]
        return xx[[ndx[0], ndx[-1]]]


    def predict(self, X, pmin=0.95, legacy_output=False):
        """predict state

        input
        ------
        X : input data ()
        pmin : classify as switch if max(p0,p1) < pmin

        output
        ------
        pred : np.array of predictions in [-1,0,1]

            the output integers are mapped to states via:
            [-1,0,1] -> [low, switch, high]
            however if legacy_output is True then:
            [0,1,-1] -> [low, high, switch]
        """
        pred_probs = self.clf.predict_proba(X.reshape(-1, 1))
        pred_probs_max = np.maximum(pred_probs[:,0], pred_probs[:,1])

        # find low/mid/high regions
        is_low = X < self.x0
        is_hgh = X > self.x1
        is_mid = ~is_low & ~is_hgh

        # find switch cases
        is_switch = is_mid & (pred_probs_max < pmin)

        # start with clf predictions, then correct low/high/switch regions
        pred = self.clf.predict(X.reshape(-1, 1))

        # ensure that low X and high X are labeled 0 and 1
        if np.diff(self.means)[0] < 0:
            pred = pred*-1+1

        if legacy_output:
            pred[is_low] = 0
            pred[is_hgh] = 1
            pred[is_switch] = -1
        else:
            pred[pred==0] = -1
            pred[is_low] = -1
            pred[is_hgh] = 1
            pred[is_switch] = 0

        return pred


def make_1DModel(s=None, epochLength=10, mfw=0, verbose=False):
    """DEPRECATED (use featurize code instead)
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

    print('make_1DModel should be alt constructor for TimeSeriesModel1D')
    raise Exception('deprecated')
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


    metaData = dict(        
        epochLength=epochLength,
        medianFilterWidth=mfwidth*epochLength,
        bandpass=s.metaData.get('bandpass', {}),
        channel=s.label
    )

    tsmf = TimeSeriesModel1D(
        t=tval, 
        x=logpowmf, 
        metaData=metaData
        )
    
    return tsmf




def test_full():
    """
    - synthetic signal
    - built tsm1d
    - score preditions
    """
    os.makedirs('scratch', exist_ok=True)
    np.random.seed(seed=123)

    # generate time series (sin wave)
    x0 = -2
    t = np.arange(81)
    x = 1*np.sin(0.05*t/2*np.pi)+x0

    # build the model
    m = TimeSeriesModel1D(t=t, x=x, metaData={})

    # predict scores
    pmin = 0.99
    scoreblock = m.predict(pmin=pmin)
    scores = scoreblock.data.ravel()

    # GMM limits and peaks
    pb_lim = m.gmm2.xinterval(pmin=pmin)
    gmm_peaks = m.gmm2.means

    # histogram
    bin_edges = np.linspace(x0-1.4, x0+1.4, 21)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    hx = m.hx(bins=bin_edges)

    # plot for sanity checking
    figx = plt.figure(figsize=(12, 4))
    ax = [plt.subplot(1, 4, (1, 3))]
    ax.append(plt.subplot(1, 4, 4, sharey=ax[0]))
    ax[0].plot(t, x, '-o', mfc='grey', label='data')
    ax[0].plot(t, scores, '-o', mfc='grey', label='scores')
    ax[0].axhspan(pb_lim[0], pb_lim[1], color='grey', alpha=0.2)
    ax[0].axhline(gmm_peaks[0], color='red', alpha=0.4)
    ax[0].axhline(gmm_peaks[1], color='red', alpha=0.4)
    ax[0].legend()
    ax[1].plot(hx, bin_centers, '-o', mfc='grey', label='data')
    ax[1].axhspan(pb_lim[0], pb_lim[1], color='grey', alpha=0.2)
    ax[1].axhline(gmm_peaks[0], color='red', alpha=0.4)
    ax[1].axhline(gmm_peaks[1], color='red', alpha=0.4)
    plt.savefig('scratch/plot-ts-model-1D-demo.png')

    # verify that predicted scores are correct (-1 low; 0 mid; 1 high)
    assert scores[4] == 0
    assert scores[5] == 1
    assert scores[47] == -1
    assert scores[48] == -1


def test_TwoStateGMMClassifier_fromdata():
    # generate time series (sin wave)
    t = np.arange(81)
    x = 1*np.sin(0.05*t/2*np.pi)

    myGMM = TwoStateGMMClassifier.from_data(x)
    print(myGMM.xinterval(pmin=0.95))



if __name__ == '__main__':

    test_TwoStateGMMClassifier_fromdata()
    test_full()

