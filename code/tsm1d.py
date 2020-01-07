
import pdb
import os

import pandas as pd
import numpy as np
from scipy import signal

import scoreblock as sb


class TimeSeriesModel1D(object):
    """1D time series model

    TODO: do we need this? used a lot in ipynb

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
        
        #return scores
    
        data_cols = ['epoch-%6.6i' % (i+1) for i in range(len(scores))]


        index = dict(
            pdiff=pdiff,
            scoreType='GMM_1D',
            )

        index_cols = [k for k in index.keys()]
        index_vals = [v for v in index.values()]


        data = index_vals+[s for s in scores]

        # dataframe via series
        df = pd.Series(index=index_cols+data_cols, data=data).to_frame().T

        return sb.ScoreBlock(df=df, index_cols=index_cols)

        # pdb.set_trace()


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


    TODO: peak scaling transformation (two peaks at -1, 1)

    TODO: alt constructor from_data() or train().. get sklearn.mixture.GMM inside here

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

        from sklearn import mixture
        # make a Gaussian Mixture Model (sklearn version)
        clf = mixture.GaussianMixture(n_components=2).fit(x.reshape(-1,1))
        return cls(clf=clf)

    def about(self):
        """"""
        print('--- TwoStateGMMClassifier.about() ---')
        print('means  :', self.means)
        print('weights:', self.weights)


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
        pred : np.array of predictions in [0,1,-1]
            0: low
            1: high
           -1: switching

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

    TODO: featurization code is duplicated in anl-preprocess, deprecate THIS

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
    #import remtools as rt
    import matplotlib.pyplot as plt

    os.makedirs('scratch', exist_ok=True)

    np.random.seed(seed=123)


    # generate time series (sin wave)
    t = np.arange(81)
    x = 1*np.sin(0.05*t/2*np.pi)

    # x = 2/(1+np.exp(0.2*(200-t)))-1
    # x += np.random.randn(len(t))*0.2


    metaData = {}

    # build the model
    m = TimeSeriesModel1D(t=t, x=x, metaData=metaData)

    # predict scores
    pdiff=0.99
    scoreblock = m.predict(pdiff=pdiff)
    scores = scoreblock.data.ravel()


    # predicted switches
    assert scores[4] == -1
    assert scores[5] == 1
    assert scores[47] == -1
    assert scores[48] == 0


    # GMM limits and peaks
    pb_lim = m.gmm2.xinterval(pdiff=pdiff)
    gmm_peaks = m.gmm2.means

    bin_edges = np.linspace(-1.4, 1.4, 21)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    hx = m.hx(bins=bin_edges)


    # print(x)
    # print(np.mean(x))
    # print(gmm_peaks)
    # print(m.gmm2.covs)
    # print(m.gmm2.weights)


    # plot for sanity checking
    figx = plt.figure(figsize=(12, 4))
    ax = [plt.subplot(1, 4, (1, 3))]
    ax.append(plt.subplot(1, 4, 4, sharey=ax[0]))

    ax[0].plot(t, x, '-o', mfc='grey', label='data')
    ax[0].plot(t, scores-2, '-o', mfc='grey', label='scores')
    ax[0].axhspan(pb_lim[0], pb_lim[1], color='grey', alpha=0.2)
    ax[0].axhline(gmm_peaks[0], color='red', alpha=0.4)
    ax[0].axhline(gmm_peaks[1], color='red', alpha=0.4)

    ax[1].plot(hx, bin_centers, '-o', mfc='grey', label='data')
    ax[1].axhspan(pb_lim[0], pb_lim[1], color='grey', alpha=0.2)
    ax[1].axhline(gmm_peaks[0], color='red', alpha=0.4)
    ax[1].axhline(gmm_peaks[1], color='red', alpha=0.4)

    ax[0].legend()

    plt.savefig('scratch/plot-ts-model-1D-demo.png')



def test_TwoStateGMMClassifier_fromdata():


    os.makedirs('scratch', exist_ok=True)
    np.random.seed(seed=123)

    # generate time series (sin wave)
    t = np.arange(81)
    x = 1*np.sin(0.05*t/2*np.pi)
    # x = 2/(1+np.exp(0.2*(200-t)))-1
    # x += np.random.randn(len(t))*0.2

    myGMM = TwoStateGMMClassifier.from_data(x)


    print(myGMM.xinterval(pdiff=0.99))

    # metaData = {}

    # # build the model
    # m = TimeSeriesModel1D(t=t, x=x, metaData=metaData)



if __name__ == '__main__':

    test_TwoStateGMMClassifier_fromdata()
    #test_full()

