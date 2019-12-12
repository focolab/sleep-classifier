#!/usr/bin/env python3
"""
    
    tools for mouse EEG/EMG analysis and sleep state classification

    Classes
        StagingParameters: parameters for data staging, with json in/output
        StagedTrialData: features and scores, with json io for subsequent analysis
        EDFData: imported EDF data, extracted signals and raw spectrograms
        SignalTrace: Signal trace and some metadata
        Spectrogram: Spectrogram with analysis methods
        SxxBundle: a bundle of 3 Spectrograms (TODO: combine w/spectrogram?)
        ScoreWizard: individual and consensus score data
        PCA: principal component analysis, no frills

    Functions
        eigsorted: diagonaize a (cov) matrix
        plot_spectrogram_cleanup: plot the featurization of spectrograms

    DEPRECATED        
        plot_trial_chunks: plots trial information in epoch chunks

sleep-scorer
    TODO:
    EDFData should provide a parameter dictionary
    EDFData lazy/delayed load

"""
import pdb
import re
import os
import json
import datetime
import time

import pandas as pd
import numpy as np
from scipy import signal
import scipy.io.wavfile

import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

import pyedflib
#import plottools as pt

import scoreblock as sb


class StagingParameters(object):
    """"""
    def __init__(self, spectrogram=None, standardization=None):
        print('STAGING PARAMETERS SHOULD BE DEPRECATED')
        self.spectrogram = spectrogram
        #self.standardization = standardization
    
    def to_dict(self):
        abt = 'parmeters for processing EEG and EMG signal data'
        
        dd = dict(_about=abt,
                  spectrogram=self.spectrogram,
                  #standardization=self.standardization.to_dict()
                  )
        return dd

    def to_json(self, out='param-staging.json'):
        dd = self.to_dict()
        with open(out,'w') as jout:
            json.dump(dd, jout, indent=2, sort_keys=False)
            jout.write('\n')

    @classmethod
    def from_json(cls, jsonfile):
        """alternative constructor reading a json file"""

        with open(jsonfile) as jfopen:
            jdic = json.load(jfopen)

        spec = jdic.get('spectrogram', None)
        stnd = jdic.get('standardization', None)
        return cls(spectrogram=spec, standardization=stnd)
    


class StagedTrialData(object):
    """EDF data, Features and Scores for one trial"""
    def __init__(self, 
                 loc=None,
                 edf=None,
                 features=None,
                 scoreblock=None,
                 stagingParameters=None,
                 trial='trialname',
                 tagDict={}):
        """

        three stages of spectrogram processing
            sxxb_raw    raw spectrograms, DFT of experimental signals
            sxxb_prep   preprocessed (median/bandpass/normalized), but not standardized
            
        TODO: edf load option (edfData/edfFile)
        TODO: import/export
        TODO: include pre-processing parameters here
        """

        if loc is None:
            raise ValueError('StagedTrialData needs a home (loc)')

        self.loc = loc      #os.path.abspath(loc)
        os.makedirs(loc, exist_ok=True)

        self.trial = trial
        self.edf = edf
        self.features = features
        self.scoreblock = scoreblock
        self.stagingParameters = stagingParameters

        self.tagDict = tagDict


    def about(self):
        """"""
        print('------ StagedTrialData.about() ------')
        #print('  trial:', self.trial)
        for k,v in self.tagDict.items():
            print('%15s : %s' % (k, str(v)))


    def to_json(self, out='staged-trial-data.json'):
        """"""
        opj = lambda x: os.path.join(self.loc, x)        

        if self.features is not None:
            self.features.to_json(opj('data-features-scoreblock.json'))

        if self.scoreblock is not None:
            self.scoreblock.to_json(opj('data-scoreblock.json'))

        self.stagingParameters.to_json(opj('param-staging.json'))

        #print(os.path.relpath(self.edf.filename, self.loc))
        edfFile = os.path.relpath(self.edf.filename, self.loc)
        fdic = dict(edf=edfFile)


        jdic = dict(_about='staged trial data output',
                    trial=self.trial,
                    tagDict=self.tagDict,
                    edfMetaData=self.edf.metadata,
                    files=fdic)
        
        jfl = opj(out)
        with open(jfl,'w') as jout:
            json.dump(jdic, jout, indent=2, sort_keys=False)
            jout.write('\n')


    @classmethod
    def from_json(cls, jfl, loadEDF=True):
        """"""
        loc = os.path.dirname(jfl)
        opj = lambda x: os.path.join(loc, x)
        
        # these file names are hard coded
        stagingParameters = StagingParameters.from_json(opj('param-staging.json'))

        try:
            scoreblock = sb.ScoreBlock.from_json(opj('data-scoreblock.json'))
        except:
            scoreblock = None

        try:
            features = sb.ScoreBlock.from_json(opj('data-features-scoreblock.json'))
        except:
            features = None


        with open(jfl) as jfopen:
            jdic = json.load(jfopen)

        trial = jdic.get('trial', 'trialname')
        files = jdic.get('files', {})
        tagDict = jdic.get('tagDict', {})
        
        edfFile = files.get('edf')
        
        if loadEDF:
            edf = EDFData(opj(edfFile))
        else:
            edf = None
            
        args = dict(trial=trial, 
                    edf=edf,
                    loc=loc,
                    scoreblock=scoreblock,
                    features=features,
                    tagDict=tagDict,
                    stagingParameters=stagingParameters)
        return cls(**args)



class PCA(object):
    def __init__(self, mu=None, vals=None, vecs=None, df_index=None):
        """vanilla PCA
        
        TODO: more general, affine transformation

        TODO: should have jsonFileName attribute, and a location?
            ** set location on export (and import) **

        N: data dimension
        M: number of observations

        attributes
        ------
        mu (array): mean (N)
        vals (array): eigenvalue (N)
        vecs (array): eigenvectors ()
        df_index(DataFrame): (optional) indexing info describing the dimensions

        methods
        ------


        TODO: subclass the sklearn version?
        """

        if mu is None:
            raise Exception('mu required')
        if vals is None:
            raise Exception('vals required')
        if vecs is None:
            raise Exception('vecs required')

        self.mu = mu
        self.vals = vals
        self.vecs = vecs
        self.df_index = df_index

    @property
    def cov(self):
        """reconstruct covariance matrix"""
        raise Exception('not tested')
        lam = np.diagflat(self.vals)
        vec = self.vecs
        vinv = np.linalg.inv(vec)
        cov = np.dot(vec, np.dot(lam, vinv))
        return cov


    def ellipse(self, w=1, PCs=[1,2], elkwa={}):
        """make a 2D plottable ellipse for a multivariate gaussian

        args
        ------
        w (int): how many standard deviations wide
        PCs (list): Which two PC vectors to use (these are 1-indexed)
        elkwa (dict): extra kwa passed in to Ellipse()
        """

        # zero index
        PCsZ = [x-1 for x in PCs]

        mu = self.mu[PCsZ]
        vals = self.vals[PCsZ]
        vecs = self.vecs.T[PCsZ]

        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))   # flips the order of elements
        width = np.sqrt(vals[0])*w*2
        height = np.sqrt(vals[1])*w*2
        elA = Ellipse(xy=mu, width=width, height=height, angle=theta, **elkwa)
        return elA


    def to_json(self, jf='./pca.json'):

        # json compatible dictionary
        jd = dict(
            mu=self.mu.tolist(),
            vals=self.vals.tolist(),
            vecs=self.vecs.tolist(),
        )

        if self.df_index is not None:
            jd['df_index'] = self.df_index.to_dict()

        # dumppp
        with open(jf, 'w') as jout:
            json.dump(jd, jout, indent=2, sort_keys=False)
            jout.write('\n')

        return

    @classmethod
    def from_json(cls, jf):
        """alternative constructor loads pca from a json file"""
        with open(jf) as jfopen:
            jdic = json.load(jfopen)

        mu = np.asarray(jdic['mu'])
        vals = np.asarray(jdic['vals'])
        vecs = np.asarray(jdic['vecs'])

        try:
            df_index = pd.DataFrame.from_dict(jdic['df_index'])
            df_index.index = df_index.index.astype(int, copy=False)
            df_index.sort_index(inplace=True)
        except:
            df_index = None

        return cls(mu=mu, vals=vals, vecs=vecs, df_index=df_index)


    @property
    def df_pcvar(self):
        """dataframe of PC variance"""
        vals_index = np.arange(len(self.vals))+1
        vals_frac = self.vals/(np.sum(self.vals))
        data = np.asarray([vals_index, self.vals, vals_frac]).T
        cols = ['PC', 'PC_var', 'PC_var_frac']
        df_pcvar = pd.DataFrame(data=data, columns=cols)
        df_pcvar['PC'].astype(int)

        return df_pcvar

    @classmethod
    def from_data(cls, data=None, df_index=None):
        """data (N,M)"""
        cov = np.cov((data))
        vals, vecs = eigsorted(cov)
        mu = np.mean(data, axis=1)
        return cls(mu=mu, vals=vals, vecs=vecs.T, df_index=df_index)

    @classmethod
    def from_mucov(cls, mu=None, cov=None, df_index=None):
        vals, vecs = eigsorted(cov)
        return cls(mu=mu, vals=vals, vecs=vecs.T, df_index=df_index)


    def project(self, data=None, PCs=[1,2,3], num_EV=None):
        """project data onto PCs
        
        args
        ------
        data (np.array): rows are coordinates, columns are observations
        PCs (list of int): PCs on which to project (1-indexed!)

        returns
        ------
        df_prj (pd.DataFrame): projections
        """

        if num_EV is not None:
            print('WARNING: num_EV is DEPRECATED, use PCs (1-indexed) instead ')
            PCs = [1,2,3]

        # shift incoming PCs to be zero-indexed
        PCsZ = [x-1 for x in PCs]

        # project
        data_centered = data*0
        for col in range(np.shape(data)[1]):
            data_centered[:, col] = data[:, col] - self.mu
        ev_prj = np.dot(self.vecs[PCsZ, :], data_centered)

        # dataframe-ize
        cols = ['PC%i' % i for i in PCs]
        df_prj = pd.DataFrame(data=ev_prj.T, columns=cols)

        return df_prj


    def project_histo(self, data=None, PCs=[1,2], bin_edges=None, 
                      numbin=40, numsig=4, tagDict={}):
        """project data onto PCs and histogram it
        
        NOTE: beware dense histograms with too many dimensions/bins
        """
        # make bins
        if bin_edges is None:
            kwa = dict(numbin=numbin, numsig=numsig)
            bin_edges = [self.make_grid_1D(PC=PC, **kwa) for PC in PCs]

        # project
        df_prj = self.project(data=data, PCs=PCs)

        # ND histogram
        hist, _ = np.histogramdd(df_prj.values, bins=bin_edges)


        # PCHisto class
        pch = PCHisto(
            dims=df_prj.columns.tolist(), 
            bin_edges=bin_edges, 
            hist=hist, 
            tagDict=tagDict, 
            pca=self
            )

        return pch



    def make_grid_1D(self, PC=1, numbin=40, numsig=4):
        """generate bins for data projected on one PC

        PC: one indexed
        numbin: how many bins (total)
        numsig: domain extends +/- numsig standard deviations
        """
        pcndx = PC-1
        LX = numsig*np.sqrt(self.vals[pcndx])
        return np.linspace(-LX, LX, numbin+1)

    def plotStackedEigenvectors(self, ax=None):
        """plot eigenvectors"""

        vecs = self.vecs[:3]
        vals = self.vals

        dy = np.max(np.abs(vecs[0]))*np.sqrt(vals[0]/np.sum(vals))
        for i, vec in enumerate(vecs):
            yscale = np.sqrt(vals[i]/np.sum(vals)) 
            xx = np.arange(len(vec))
            yy = vec*yscale
            ax.plot([xx[0], xx[-1]], [-dy*i]*2, '-', color='grey')
            ax.plot(xx, yy-dy*i, label='PC%i (%3.1f pctvar)' % (i+1, 100*vals[i]/np.sum(vals)))

    def plotSummary(self, f='plot-pca-summary.png', data=None):
        """plot PCA Summary (possibly with raw data)"""
        
        figx = plt.figure(figsize=(12, 6), dpi=100)
        ax = [plt.subplot(121), plt.subplot(222), plt.subplot(224)]

        alpha = 0.2
        if data is not None:
            df_prj = self.project(data=data)
            ax[0].scatter(df_prj['PC1'], df_prj['PC2'], alpha=alpha)
            ax[0].set_xlabel('PC1')
            ax[0].set_ylabel('PC2')

        #pdb.set_trace()
        xx = np.arange(len(self.mu))
        ax[1].plot(xx, self.mu)
        ax[1].set_ylabel('avg spectrogram')
        #ax[1].set_yscale('log')

        self.plotStackedEigenvectors(ax=ax[2])
        ax[2].set_title('PCA projection')
        ax[2].legend()
        ax[2].set_title('PCA eigenvectors')
        ax[2].set_xlabel('frequency bin')
        ax[2].set_ylabel('value')

        txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        figx.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)

        plt.tight_layout()
        plt.savefig(f)
        plt.close('all')
        
    def about(self):
        """"""
        print('------ PCA.about() ------')
        print('num dims:', len(self.mu))
        print(self.df_pcvar.head())



class PCHisto(object):
    """histogram of PC projected data


    attributes
    ------
    bin_edges (list of lists): one list of bin edges per dimension
    hist (ND array): bin counts
    dims (list) : names of the dimensions
    tagDict (dict) : metadata tags (trial/genotype etc)
    pca (remtools.PCA) : PCA used for projection



    """
    def __init__(self, dims=None, bin_edges=None, hist=None, tagDict={}, pca=None):

        self.tagDict = tagDict
        self.pca = pca
        self.dims = dims
        self.bin_edges = bin_edges
        self.hist = hist

    def about(self):

        print('------ PCHisto.about() ---------')
        print(self.tagDict)
        print(self.dims)






class SxxBundle(object):
    """Bundle of spectrograms (typically for one trial)
    """
    def __init__(self, EEG1=None, EEG2=None, EMG=None):
        self.EEG1 = EEG1
        self.EEG2 = EEG2
        self.EMG = EMG
        self.spectrograms = dict(EEG1=self.EEG1, EEG2=self.EEG2, EMG=self.EMG)

    @classmethod
    def from_EDFData(cls, edfd):
        """build using raw spectrograms from EDFData"""
        EEG1 = edfd.spectrograms['EEG1']
        EEG2 = edfd.spectrograms['EEG2']
        EMG = edfd.spectrograms['EMG']
        return cls(EEG1=EEG1, EEG2=EEG2, EMG=EMG)

    @property
    def stack(self):
        """stack Sxx data on frequency axis """
        return np.vstack([self.EEG1.Sxx, self.EEG2.Sxx, self.EMG.Sxx])

    @classmethod
    def fuse(cls, bundles):
        """fuse together SxxBundles (join trials on the time axis)"""
        sxxA = [bundle.EEG1 for bundle in bundles]
        sxxB = [bundle.EEG2 for bundle in bundles]
        sxxC = [bundle.EMG for bundle in bundles]
        EEG1 = sxxA[0].concatenate(others=sxxA[1:])
        EEG2 = sxxB[0].concatenate(others=sxxB[1:])
        EMG  = sxxC[0].concatenate(others=sxxC[1:])
        return cls(EEG1=EEG1, EEG2=EEG2, EMG=EMG)

    def pca(self):
        """run PCA on the stacked data"""
        stack = np.vstack([self.EEG1.Sxx, self.EEG2.Sxx, self.EMG.Sxx])
        return PCA(stack)

    def to_scoreblock(self):
        """not scores, but scoreblock is a useful container"""
        import scoreblock as sb
        dd = dict(
            EEG1=self.EEG1.to_df,
            EEG2=self.EEG2.to_df,
            EMG=self.EMG.to_df
            )
        df = pd.concat(dd).reset_index().rename(columns={'level_0': 'channel'})
        index_cols = ['channel','f[Hz]']
        return sb.ScoreBlock(df=df, index_cols=index_cols)


    def to_dataframe(self):
        """one dataframe with a multiindex to track channels"""
        dd = dict(EEG1=self.EEG1.to_df,
                  EEG2=self.EEG2.to_df,
                  EMG=self.EMG.to_df
                  )
        df = pd.concat(dd)
        df.index.set_names(['channel','f[Hz]'], inplace=True)
        return df

    def to_csv(self, csv='sxx_bundle.csv'):
        """make a multi-index dataframe and write it to csv"""
        self.to_dataframe().to_csv(csv, float_format='%g')
        return

    @classmethod
    def from_csv(cls, csv=None):
        """"""
        df = pd.read_csv(csv, index_col=[0,1])

        dfEEG1 = df.loc['EEG1']
        dfEEG2 = df.loc['EEG2']
        dfEMG = df.loc['EMG']
        EEG1 = Spectrogram.from_df(df=dfEEG1, label='EEG1')
        EEG2 = Spectrogram.from_df(df=dfEEG2, label='EEG2')
        EMG = Spectrogram.from_df(df=dfEMG, label='EMG')

        return cls(EEG1=EEG1, EEG2=EEG2, EMG=EMG)

    
    def prep(self, pEEG=None, pEMG=None):
        """preprocessing"""
        #== merge default parameters with input
        pEEGd = dict(lowpass=None, highpass=None, logscale=False, 
                     normalize=False, medianfilter=None, stride=None)
        pEMGd = dict(lowpass=None, highpass=None, logscale=False, 
                     normalize=False, medianfilter=None, stride=None)
        pEEGd.update(pEEG)
        pEMGd.update(pEMG)
        self.params = dict(EEG=pEEGd, EMG=pEMGd)

        EEG1 = self.EEG1.prep(pEEGd)
        EEG2 = self.EEG2.prep(pEEGd)
        EMG = self.EMG.prep(pEMG)

        return SxxBundle(EEG1=EEG1, EEG2=EEG2, EMG=EMG)







class Spectrogram(object):
    """Spectrogram with some signal processing methods

    Attributes:
        f (np.array): frequencies
        t (np.array): times (bin centers)
        Sxx (np.array): spectrogram (row/col are f/t)
        label (str): name of the signal trace

    """
    def __init__(self, Sxx=None, f=None, t=None, label='spectrogram'):
        """spectrogram and related parameters"""
        self.f = f
        self.t = t
        self.Sxx = Sxx
        self.label = label

    @classmethod
    def from_signal(cls, sig=None, fs=1., nperseg=None, noverlap=0, label='spectrogram'):
        """alternative constructor"""
        #== NOTE: output frequency spacing (in f) is 1/L, where L is the interval length
        f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)        
        return cls(f=f, t=t, Sxx=Sxx, label=label)

    @property
    def avg(self):
        return np.mean(self.Sxx, axis=1)

    @property
    def std(self):
        return np.std(self.Sxx, axis=1)

    @property
    def to_df(self):
        """to dataframe"""
        cols = ['time[s]-%g' % f for f in self.t]
        df = pd.DataFrame(data=self.Sxx, columns=cols)
        df.index = self.f
        df.index.name = 'f[Hz]'
        return df

    @classmethod
    def from_df(cls, df=None, label='spectrogram'):
        """from dataframe, row/col contain freq/time"""
        colparse = lambda x: float(x.split('-')[1])
        tt = np.asarray([colparse(col) for col in df.columns])
        ff = df.index.values
        Sxx = df.values
        return cls(f=ff, t=tt, Sxx=Sxx, label=label)

    @property
    def df_avgstd(self):
        """datafame with avg and std, useful for standardizing data"""
        cols = ['f[Hz]', 'avg', 'std']
        data = np.asarray([self.f, self.avg, self.std]).T
        return pd.DataFrame(data=data, columns=cols)


    def prep(self, params=None):
        """apply multiple preprocessing steps in succession

        median filter, bandpass filter, striding, normalization, log scaling
        TODO: make params explicit
        TODO: default parameters should go here (defaults should be False, to do nothing)
        TODO: inplace option?
        """

        sxx = self
        
        if params['medianfilter'] is not None:
            sxx = sxx.median(n=params['medianfilter'])

        sxx = sxx.lowpass(f=params['lowpass']).highpass(f=params['highpass'])
        
        if params['stride'] > 1:
            sxx = sxx.stride(n=params['stride'])
        if params['normalize']:
            sxx = sxx.normalize()
        if params['logscale']:
            sxx = sxx.logscale()

        return sxx

    def stride(self, n=1):
        """stride (frequency axis)"""
        Sxx = self.Sxx[::n, :]
        ff = self.f[::n]
        return Spectrogram(Sxx=Sxx, f=ff, t=self.t, label=self.label)

    def concatenate(self, others):
        """concatenate multiple trials Sxx (time gets borked and reset)
        NOTE: this does zero error checking w/r to epoch length, frequency range, etc..
        """
        Sxx = np.hstack([self.Sxx]+[s.Sxx for s in others])
        t = np.arange(np.shape(Sxx)[1])+1
        return Spectrogram(Sxx=Sxx, f=self.f, t=t, label=self.label)

    def median(self, n=3):
        """median filter (freq axis)"""
        Sxx = signal.medfilt(self.Sxx, kernel_size=[n, 1])
        return Spectrogram(Sxx=Sxx, f=self.f, t=self.t, label=self.label)

    def normalize(self):
        """scale so that the mean spectrum is normalized"""
        Sxx = self.Sxx/np.sum(self.avg)
        
        area = np.mean(Sxx) * (self.f[-1] - self.f[0])
        print('normalization signal/area:', self.label, area)
        return Spectrogram(Sxx=Sxx, f=self.f, t=self.t, label=self.label)

    def logscale(self):
        Sxx = np.log(self.Sxx)
        return Spectrogram(Sxx=Sxx, f=self.f, t=self.t, label=self.label)

    def lowpass(self, f=None):
        """return a new Spectrogram with only low frequencies <=f"""
        if f is None:
            ndx = np.arange(len(self.f))
        else:
            ndx = np.where(self.f <= f)[0]        
        return self.ndxpass(ndx=ndx)

    def highpass(self, f=None):
        """return a new Spectrogram with only high frequencies >=f"""
        if f is None:
            ndx = np.arange(len(self.f))
        else:
            ndx = np.where(self.f >= f)[0]        
        return self.ndxpass(ndx=ndx)

    def ndxpass(self, ndx=None):
        """low/highpass or whatev, return a new spectrogram with a subset of f indices"""
        Sxx = self.Sxx[ndx, :]
        f = self.f[ndx]
        return Spectrogram(Sxx=Sxx, f=f, t=self.t, label=self.label)
    

    def plot(self):
        """could implement plotting here"""
        #sp = spectrograms[label]
        #rr = np.log10(sp.Sxx_range)
        #bin_edges = sp.t-epoch_duration/2.        
        #im = axx[panel].pcolormesh(bin_edges[ia:ib+1], 
                                #sp.f, 
                                #np.log10(sp.Sxx[:,ia:ib+1]), 
                                #cmap='viridis', 
                                #vmin=rr[0], 
                                #vmax=rr[1])
        #cbar = figx.colorbar(im, ax=axx[panel], pad=0.01, fraction=0.02, aspect=10)
        #axx[panel].set_ylabel('%s\n f [Hz]' % (sp.label))
        #axx[panel].grid(True)
        #if i == len(spectrograms)-1:
            #axx[panel].set_xlabel('Time [sec]')
            ##print(axx[panel].get_xlim())
        #else:
            #axx[panel].set_xticks([])
        pass








from scipy.signal import butter, lfilter
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_low(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a

def butter_high(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_filter(data, fs, lowcut=None, highcut=None, order=5):
    """

    low, high or bandpass filter depending on if lowcut and highcut are passed

    adapted from:
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """

    if lowcut is None and highcut is None:
        raise Exception('butter_filter requires lowcut, highcut, or both')

    if lowcut is not None and highcut is not None:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    elif lowcut is None:
        b, a = butter_high(highcut, fs, order=order)
    elif highcut is None:
        b, a = butter_high(highcut, fs, order=order)
        
    y = lfilter(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class SignalTrace(object):
    """signal trace and metadata

    Attributes:
        name (str): name tag
        sig (np.array): the signal
        f (float): frequency [Hz]
        samples_per_epoch (int): is what it says
            NOTE: is epoch information essential here? Seems like an external decision..
    
    TODO: method that returns a range (chunk) of epochs
    """

    def __init__(self, 
                 label='unlabeled_signal', 
                 sig=None, 
                 f=1,
                 samples_per_epoch=None):

        self.label = label
        self.sig = sig
        self.f = f
        self.samples_per_epoch = samples_per_epoch

        num_samples = len(sig)
        duration = num_samples/f
        epoch_duration = samples_per_epoch/f

        self.num_samples = num_samples
        self.duration = duration
        self.epoch_duration = epoch_duration
        self.num_epochs = num_samples/samples_per_epoch

        self.metaData = {}

    def bandpass(self, lowcut=None, highcut=None):
        """bandpass filter the signal"""

        order = 5
        new_sig = butter_filter(self.sig, self.f, lowcut=lowcut, highcut=highcut, order=order)

        new_st = SignalTrace(
            sig=new_sig,
            f=self.f,
            samples_per_epoch=self.samples_per_epoch,
            label=self.label
        )

        bp = dict(bandpass=dict(lowcut=lowcut, highcut=highcut))

        new_st.metaData.update(bp)

        return new_st


    @property
    def tvec(self):
        """time vector"""
        return np.arange(self.num_samples)/self.f


    def report(self):
        """"""
        print('---- SignalTrace.about() ----')
        for k, v in self.__dict__.items():
            print(k,':', v)
        print('-----------------')

    def about(self):
        self.report()


    
class EDFData(object):
    def __init__(self, edf=None):
        """load edf, extract signals, compute spectrograms

        TODO:
        - loading signals should be its own method
        - epoch_length as input argument
        """
        
        f = pyedflib.EdfReader(edf)
        
        n = f.signals_in_file                       # number of signals
        signal_labels = [ss.replace(" ", "") for ss in f.getSignalLabels()]         # signal labels
        duration = f.file_duration                  # total duration [s]
        num_samples = f.getNSamples()               # number of discrete samples
        num_epochs = f.datarecords_in_file          # number of epochs
        
        #== derived
        trial = re.split('\\.', os.path.basename(edf))[0]
        epoch_duration = duration/num_epochs
        samples_per_epoch = num_samples/num_epochs
        freq = num_samples/duration

        
        startdate = "%4i-%2.2i-%2.2i" % (f.getStartdatetime().year,f.getStartdatetime().month,f.getStartdatetime().day)
        starttime = "%i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second)
        print('-----------------------------------------------------------------------')
        print('file               :', edf)
        print('trial              :', trial)
        print("num datarecords    : %i" % f.datarecords_in_file)
        print("num annotations    : %i" % f.annotations_in_file)
        print('number of signals  :', n)
        print('duration [s]       :', duration)
        print("startdatetime      :", startdate, starttime)
        print('epoch duration [s] :', epoch_duration)
        print("samples in file    :",  num_samples)
        print('samples_per_epoch  :', samples_per_epoch)
        print('frequency [Hz]     :', freq)
        print('signal_labels      :', signal_labels)
        print('------------------------------------')

        #=======================================================
        #== load signal traces
        signal_traces = {}
        for i, label in enumerate(signal_labels):
            data = f.readSignal(i)
            print('  loading signal trace:', label)
            signal = SignalTrace(label=label, 
                                 sig=data, 
                                 f=freq[i], 
                                 samples_per_epoch=samples_per_epoch[i])            
            signal_traces[label] = signal

        #=======================================================
        #== compute spectrograms
        spectrograms = {}
        for label in ['EEG1','EEG2','EMG']:
            st = signal_traces[label]
            print('  computing spectrogram:', label)
            spectrograms[label] = Spectrogram.from_signal(sig=st.sig,
                                                          fs=st.f,
                                                          nperseg=int(st.samples_per_epoch), 
                                                          noverlap=0,
                                                          label=label)

        self.startdate = startdate
        self.starttime = starttime
        self.filename = edf
        self.trial = trial
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.epoch_duration = epoch_duration
        self.signal_labels = signal_labels
        self.signal_traces = signal_traces        
        self.spectrograms = spectrograms
        self.metadata = dict(
            about_='edf file metadata',
            filename=edf,
            startdate=startdate,
            starttime=starttime,
            #epoch_duration_s=epoch_duration,
            #num_epochs=num_epochs,
            #num_samples=num_samples,
            EEG1_freq=signal_traces['EEG1'].f,
            EEG2_freq=signal_traces['EEG2'].f,
            EMG_freq=signal_traces['EMG'].f,
        )

        print('  edf load complete: %s' % edf)
        print('-----------------------------------------------------------------------')

    @property
    def freq(self):
        """"""

        ff = [self.signal_traces[name].f for name in ['EEG1', 'EEG2', 'EMG']]
        f = list(set(ff))

        if len(f) > 1:
            raise Exception('channel frequencies do not match :( )')

        return f[0]


    def dump_wavs(self, dest='./.'):
        """convert to wav files"""

        os.makedirs(dest, exist_ok=True)

        for ch in ['EEG1', 'EEG2', 'EMG']:
            signal_trace = self.signal_traces[ch]

            filename = os.path.join(dest, 'trial-%s-%s.wav' % (str(self.trial), ch))
            rate = int(signal_trace.f)
            data = signal_trace.sig

            print('wav export:', filename, '(%i Hz)' % (rate))

            scipy.io.wavfile.write(filename, rate, data)



    # def dfTrialEpoch(self):
    #     """return a dataframe with trial and epochs for appending to others"""
    #     epochs = range(1, self.num_epochs+1)
    #     trial = [self.trial]*self.num_epochs
    #     cols = ['Epoch#', 'trial']
        
    #     df = pd.DataFrame(data=zip(epochs,trial), columns=cols)
    #     return df
    

class ScoreWizard(object):
    """for storing, slicing and dicing human score data (for ONE trial)
    
    """
    
    def __init__(self, data=None):
        """"""

        print('WARNING: ScoreWizard deprecated in favor of scoreblocks')
        self.data = data
        self.scorers = data['scorer'].unique()
        self.trial = data['trial'].unique()[0]
        self.data_by_scorer = {ss:data[data['scorer']==ss] for ss in self.scorers}

        if len(data['trial'].unique()) > 1:
            print('>1 trial in score data',  data['trial'].unique())
            raise Exception()
        
        #== helper dictionaries
        self.scoreNum2Str = dict(set(zip(data['Score#'], data['Score'])))        
        self.scoreNum2Str[0] = 'XXX'
        self.scoreStr2Num = {v:k for k,v in self.scoreNum2Str.items()}

        #== consensus dataframe
        consensus = lambda x: x[0] if len(np.unique(x)) == 1 else 0
        pp = data.pivot_table(index=['Epoch#', 'trial'], columns='scorer', values='Score#')    
        pp['cScoreNum'] = pp.apply(consensus, axis=1)
        pp['cScoreStr'] = pp['cScoreNum'].map(self.scoreNum2Str)
        self.dfScore = pp
        
        #== just the consensus scores
        cols = ['Epoch#', 'trial', 'cScoreNum', 'cScoreStr']
        df_flat = pp.reset_index()[cols]
        df_flat.columns.name = None
        self.dfConsensus = df_flat
        
        self.dfConsensus['trial'] = [str(x) for x in self.dfConsensus['trial']]

    @property
    def scoreblock(self):
        import scoreblock as sb

        scores = []
        ndx = []
        for ss in self.scorers:
            dd = self.data_by_scorer[ss]
            scores.append(dd['Score'].values)
            ndx.append([self.trial, ss])
            #pdb.set_trace()

        scores.append(self.dfConsensus['cScoreStr'])
        ndx.append([self.trial, 'consensus'])

        index_cols = ['Score', 'scorer']
        data_cols = ['epoch-%6.6i' % (i+1) for i in range(len(scores[0]))]
        dfa = pd.DataFrame(data=ndx, columns=index_cols)
        dfb = pd.DataFrame(data=scores, columns=data_cols)

        df = pd.concat([dfa, dfb], axis=1)

        return sb.ScoreBlock(df=df, index_cols=index_cols)


    @classmethod
    def from_csv(cls, csv):
        """build from a data-scores-*csv file"""       
        return cls(data=pd.read_csv(csv))

    def to_csv(self, csv):
        """"""
        self.data.to_csv(csv)
        return


def eigsorted(cov):
    """returned eigenvectors are in COLUMNS"""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]    # flips order of elements
    return vals[order], vecs[:,order]


def split_features_by_scores(features=None, scores=None):
    """split a feature block by scores

    - block rows are features and columns are observations
    - extends to downstream features (PC projections)

    input
    ------
    features (np.array or pd.DataFrame): (M by N)
    scores (categorical array): (N)

    output
    ------
    out (dict): feature blocks, keyed by score

    """

    if isinstance(features, pd.DataFrame):
        featureType = 'df'
        df_ndx = features.index.copy()
    elif isinstance(features, np.ndarray):
        featureType = 'np'
    else:
        raise Exception('features type not recognized')

    out = dict()
    for s in np.unique(scores):
        ndx = np.where(scores==s)[0]
        if featureType == 'np':
            data = features[:, ndx]
        elif featureType == 'df':
            data = pd.DataFrame(data=features.values[:, ndx]).set_index(df_ndx)

        out[s] = data

    return out






    
    




def plot_spectrogram_cleanup(data, out='plot-sxx-cleanup.png'):
    """plot spectrograms as they are transformed into model input features
    
        raw     just the (mean) DFT for each signal
        prep    median filtered, bandpass filtered, normalized (depending on parameters)
        feat    standardized (one transform applied to each trial)

        shaded areas indicate 5th/95th percentiles

    INPUT:
        data (list): list of StagedTrialData objects
        out (str): output file name
    
    """
    
    
    #== plotting raw and pre-processed power spectra
    figx = plt.figure(figsize=(12, 8), dpi=100)
    #== right hand side 
    axb = [plt.subplot(331), plt.subplot(334),  plt.subplot(337)]
    axm = [plt.subplot(332), plt.subplot(335),  plt.subplot(338)]
    axr = [plt.subplot(333), plt.subplot(336),  plt.subplot(339)]

    #== LHS raw spectrogram averages
    for i, signal in enumerate(['EEG1', 'EEG2', 'EMG']):
        for trialdata in data:
            trial = trialdata.trial
            frq = trialdata.edf.spectrograms[signal].f
            avg = trialdata.edf.spectrograms[signal].avg
            std = trialdata.edf.spectrograms[signal].std
            axb[i].plot(frq, avg, label='trial %s' % (str(trial)))

        axb[i].legend()
        axb[i].grid()
        axb[i].set_ylabel('%s avg Sxx' % (signal))
        axb[i].set_xlabel('f [Hz]')
        axb[i].set_yscale('symlog')

    ymin = min(axb[0].get_ylim()[0], axb[1].get_ylim()[0])
    ymax = max(axb[0].get_ylim()[1], axb[1].get_ylim()[1])
    axb[0].set_ylim([ymin, ymax])
    axb[1].set_ylim([ymin, ymax])


    #== middle 'pre-processed' spectrograms
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']*2
    for i, signal in enumerate(['EEG1', 'EEG2', 'EMG']):
        for j, trialdata in enumerate(data):
            trial = trialdata.trial
            frq = trialdata.sxxb_prep.spectrograms[signal].f
            avg = trialdata.sxxb_prep.spectrograms[signal].avg
            std = trialdata.sxxb_prep.spectrograms[signal].std
            pct10 = np.percentile(trialdata.sxxb_prep.spectrograms[signal].Sxx, 5,  axis=1)
            pct90 = np.percentile(trialdata.sxxb_prep.spectrograms[signal].Sxx, 95, axis=1)

            axm[i].fill_between(frq, pct90, pct10, color=cycle[j], alpha=0.2)
            axm[i].plot(frq, pct10, lw=1, color=cycle[j])
            axm[i].plot(frq, pct90, lw=1, color=cycle[j])
            axm[i].plot(frq, avg,   lw=2, color=cycle[j], label='trial %s' % (str(trial)))

        #axm[i].legend()
        axm[i].grid()
        axm[i].set_ylabel('%s pre-processed avg Sxx' % (signal))
        axm[i].set_xlabel('f [Hz]')
        #axm[i].set_yscale('symlog')

    ymin = min(axm[0].get_ylim()[0], axm[1].get_ylim()[0])
    ymax = max(axm[0].get_ylim()[1], axm[1].get_ylim()[1])
    axm[0].set_ylim([ymin, ymax])
    axm[1].set_ylim([ymin, ymax])


    plt.tight_layout()
    plt.savefig(out)
    plt.close('all')
    
    










