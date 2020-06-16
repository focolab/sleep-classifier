"""
autoscoring EMG/EEG mouse recordings

    - project recording data onto 2D feature space (EEG and EMG rms power)
    - plot the full 24h timeseries (EEG and EMG rms power)
    - do conservative auto-scoring (Wake, NonREM and, NOTSURE(incl REM))
    - create sirenia and csv formatted score output
"""
import pdb
import os
import json

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix

import sleep_scorer.plottools as pt
import sleep_scorer.remtools as rt
import sleep_scorer.featurize as fz
import sleep_scorer.tsm1d as tsm1d

sns.set(color_codes=True)
sns.set_style('ticks')

def stage(edf=None, ft=None, dest='xxx', tagDict=None):
    """compute and stage features"""
    if tagDict is None: tagDict = {}

    # load edf
    edfd = rt.EDFData(edf)

    # load featurization params
    with open(ft) as jfopen:
        jdic = json.load(jfopen)

    # featurize
    if 'spectrogram' in jdic.keys():
        params = jdic['spectrogram']
        features_scb = fz.compute_powspec_features(edfd=edfd, params=params)
    elif 'rmspower' in jdic.keys():
        params = jdic['rmspower']
        features_scb = fz.compute_rmspow_features(edfd=edfd, params=params)
    else:
        raise Exception('params not recognized')

    trial = tagDict.get('trial', 'xxx')

    #== stage features
    std = rt.StagedTrialData(
        loc=dest,
        edf=edfd,
        scoreblock=None,
        trial=trial,
        features=features_scb,
        stagingParameters=jdic,
        tagDict=tagDict
        )
    return std

def qp_classifier(mx, my, pmin=0.95):
    """quick and painless sleep state classifier (Wake, Non-REM, NOTSURE)

    inspiration: https://www.youtube.com/watch?v=Wi4N4duxwgk

    input
    ------
    mx : (TimeSeriesModel1D) EMG rms power
    my : (TimeSeriesModel1D) EEG rms power
    pmin : (float) Confidence threshold 0.5<pmin<1. A point is classified as
        NOTSURE if its GMM score probability is < pmin. As a result, a higher
        pmin means more conservative classification and a wider crossover
        region.

    returns
    ------
    stack : (ScoreBlock) 1D GMM scores and the qp consensus scores

    The qp classifier classifies epochs as Wake, Non-REM and NOTSURE (incl REM)
    using uses a consensus score from two 1D GMM classifiers (one for EMG, one
    for EEG), exploiting the fact that both EMG and EEG RMS power are strongly
    bimodal. The idea is to make only the very safe predictions; the most
    slam dunk Wake and Non-REM epochs are labeled as such, and should not
    require human verification, and then everything else (incl REM) is lumped
    into NOTSURE, to be sorted out by manual scoring.

    The inputs, mx and my, are both TimeSeriesModel1D objects, each equipped
    with a two state, 1D GMM classifier that has a crossover 'NOTSURE' region
    in the middle. Using the consensus of both 1D models, only points with the
    same score, (with p>pmin) are classified as Non-REM  or Wake, everything
    else gets scored as NOTSURE.
    """
    EMGMAP = {-1:'Non-REM', 0:'NOTSURE', 1:'Wake'}
    EEGMAP = {-1:'Wake', 0:'NOTSURE', 1:'Non-REM'}
    featureX = mx.metaData.get('tag', 'xtag')
    featureY = my.metaData.get('tag', 'ytag')

    # predictions from the individual, 1D GMMs
    pred_x = mx.predict(pmin=pmin).add_const_index_col(name='feature', value=featureX)
    pred_y = my.predict(pmin=pmin).add_const_index_col(name='feature', value=featureY)
    pred_x = pred_x.applymap(EMGMAP)
    pred_y = pred_y.applymap(EEGMAP)

    # Stack the predictions and take a consensus.
    # Any non-consensus epoch will be classified as NOTSURE
    stack = pred_x.stack(others=[pred_y]).consensus(data_fill='NOTSURE')
    stack.df.iloc[-1]['classifier'] = 'qp_classifier'
    return stack


def get_chunk(x, n=2, N=4):
    """return chunk number 'n' of the vector x w/ N equal sized chunks"""
    chunk = len(x)//N
    return x[n*chunk:(n+1)*chunk]


def plot_qp_vs_human_scores(autoscorer=None, scoreblock=None):
    """plot a comparison of autoscorer and human consensus scores (one trial)

    input
    ------
    autoscorer (AutoScorer): trained AutoScorer (2D EMG/EEG classifier)
    scoreblock (ScoreBlock): should only have one row of scores, presumably
        these are human consensus scores

    output
    ------
    fig : figure
    ax (list): list of subplot axes
    """

    point_kwa = dict(lw=0, marker='o', ms=2, mec='none', color='grey', alpha=1)
    text_kwa = dict(color='k', fontsize=10)
    kwa_cnf = dict(cbar=False, colorkwa=dict(fraction=0.04), cmap=plt.cm.Blues)

    h2d = autoscorer.h2d()
    mx = autoscorer.mx
    my = autoscorer.my
    labels_hum = ['Non REM', 'REM', 'Wake', 'XXX']      # one panel per label
    labels_cnf = ['Non REM', 'REM', 'Wake', 'XXX'] #, 'NOTSURE']

    map_hum = {'Non REM X':'XXX'}
    map_mdl = {'Non-REM':'Non REM', 'NOTSURE':'XXX'}
    # dict_hum = dict()

    # human and model score vectors (w/ some tidying)
    # sc_hum = scoreblock.applymap({'Non REM X':'Non REM'}).data.ravel()
    sc_hum = scoreblock.applymap(map_hum).data.ravel()
    sc_mdl = autoscorer.scores_pred.applymap(map_mdl).data[-1]

    # build dataframe with xy features and (human) consensus scores
    xys = np.vstack((mx.x, my.x, scoreblock.data.ravel())).T
    df_xys = pd.DataFrame(data=xys, columns=['x', 'y', 'scores'])

    # figure setup
    num_panels = len(labels_hum)+2
    fig = plt.figure(figsize=(4*num_panels, 4))
    ax = [plt.subplot(1, num_panels, i+1) for i in range(num_panels)]

    # PLOT joint distribution histogram
    plt_hist_kwa = dict(cmap='bone', levels='auto', ptype='imshow')
    pt.plot_2D_hist(h2d=h2d, ax=ax[0], **plt_hist_kwa, cbar=False)
    plot_gmm_overlay(autoscorer.mx, autoscorer.my, ax=ax[0], pmin=autoscorer.pmin)
    plt_info_text_block(mx=autoscorer.mx, my=autoscorer.my, ax=ax[0], txt_kwa=dict(fontsize=7, color='white'))


    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()

    # PLOT features grouped by score
    for i, label in enumerate(labels_hum):
        axi = ax[i+1]
        dfi = df_xys[df_xys['scores'] == label]
        colx, coly = dfi.columns[:2]
        frac = 100.0*len(dfi)/len(df_xys)
        ntag = 'N=%i  (%2.1f %%)' % (len(dfi), frac)
        axi.plot(dfi[colx], dfi[coly], **point_kwa)
        axi.text(0.02, 0.98, label, ha='left', va='top', transform=axi.transAxes, **text_kwa)
        axi.text(0.98, 0.98, ntag, ha='right', va='top', transform=axi.transAxes, **text_kwa)
        plot_gmm_overlay(autoscorer.mx, autoscorer.my, ax=axi, pmin=autoscorer.pmin)
        if i == 0:
            plt_info_text_block(mx=autoscorer.mx, my=autoscorer.my, ax=axi, 
                        txt_kwa=dict(fontsize=7, color='black'))


        axi.set_xlim(xlim)
        axi.set_ylim(ylim)

    # PLOT confusion matrix
    cnf = confusion_matrix(sc_mdl, sc_hum, labels=labels_cnf)
    pt.plot_confusion_matrix(ax=ax[-1], cm=cnf, normalize=False, classes=labels_cnf, **kwa_cnf)
    ax[-1].set_ylabel('model prediction')
    ax[-1].set_xlabel('human consensus')

    plt.tight_layout()

    return fig, ax

def plot_ts_chunk(mx=None, my=None, ax=None, n=0, N=0, scores=None):
    """plot a chunk of the timeseries and a rug plot for NOTSURE epochs"""

    line_kwa = dict(lw=0.5, color='darkgray')

    # get the chunk from each timeseries
    tc = get_chunk(mx.t, n=n, N=N)
    xc = get_chunk(mx.x, n=n, N=N)
    yc = get_chunk(my.x, n=n, N=N)
    sc = get_chunk(scores, n=n, N=N)

    ndx_u = sc == 'NOTSURE'

    # convert time from seconds to hours, ticks every hour
    tch = tc/3600.
    tick_min = int(np.floor(tch[0]))
    tick_max = int(np.ceil(tch[-1]))
    xticks = range(tick_min, tick_max+1)

    # plot the rms power time series
    dy0 = 2
    dy1 = -2
    ax.axhline(y=dy0, dashes=(4,8), color='grey', lw=0.5, alpha=0.5)
    ax.axhline(y=dy1, dashes=(4,8), color='grey', lw=0.5, alpha=0.5)
    ax.plot(tch, xc+dy0, label=mx.metaData['channel'], **line_kwa)
    ax.plot(tch, yc+dy1, label=my.metaData['channel'], **line_kwa)
    tx_kwa = dict(ha='right', va='center', color='grey', fontsize='x-small')
    ax.text(tch[0], dy0, mx.metaData['channel']+' ', **tx_kwa)
    ax.text(tch[0], dy1, my.metaData['channel']+' ', **tx_kwa)

    # red rug plot for NOTSURE epochs
    ax.scatter(tch[ndx_u], 0*tch[ndx_u], marker='|', color='r', edgecolors='r', lw=0.5, s=8)

    # pimp my plot
    ax.set_ylim([-4.5, 4.5])
    ax.set_xlim([xticks[0], xticks[-1]])

    # ticks and spines
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', which='both', direction='in', width=0.5, 
                color='grey', pad=0, labelsize='xx-small', labelcolor='grey')
    ax.set_yticks([])
    ax.tick_params(axis='y', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_gmm_overlay(mx, my, ax=None, pmin=0.95, zorder=3):
    """plot GMM crossover bands and the Non-REM + Wake peaks"""
    xint = mx.gmm2.xinterval(pmin=pmin)
    yint = my.gmm2.xinterval(pmin=pmin)

    ax.text(0.5, 0.99, 'pmin='+str(pmin), ha='center', va='top', transform=ax.transAxes, color='r')

    ax.axvspan(*xint, color='r', alpha=0.2, zorder=zorder)
    ax.axhspan(*yint, color='r', alpha=0.2, zorder=zorder)
    ax.axvline(xint[0], color='r', alpha=0.5, zorder=zorder)
    ax.axvline(xint[1], color='r', alpha=0.5, zorder=zorder)
    ax.axhline(yint[0], color='r', alpha=0.5, zorder=zorder)
    ax.axhline(yint[1], color='r', alpha=0.5, zorder=zorder)

    ax.plot(mx.gmm2.x0, my.gmm2.x1, 'x', color='k', ms=16, mew=4, zorder=zorder)
    ax.plot(mx.gmm2.x1, my.gmm2.x0, 'x', color='k', ms=16, mew=4, zorder=zorder)
    ax.plot(mx.gmm2.x0, my.gmm2.x1, 'x', color='grey', ms=15, mew=2, zorder=zorder)
    ax.plot(mx.gmm2.x1, my.gmm2.x0, 'x', color='grey', ms=15, mew=2, zorder=zorder)

def plt_info_text_block(mx=None, my=None, ax=None, txt_kwa=None):
    """plot metadata info text block in lower left corner"""
    if txt_kwa is None: txt_kwa = {}
    txt_kwa_def = dict(
        fontsize=5,
        color='black',
        fontfamily='monospace',
        )
    txt_kwa_def.update(txt_kwa)

    t_spam = []
    t_spam.append('f [Hz] : %s \n' % (str(mx.metaData.get('freq',-1))))
    t_spam.append('epc [s]: %-3.0f \n' % (mx.metaData.get('epochLength', -1)))
    for m in [mx, my]:
        # bandpass = m.metaData.get('bandpass', dict(lowcut=-1, highcut=-1))
        lowcut = m.metaData.get('lowcut', -1)
        highcut = m.metaData.get('highcut', -1)

        t_spam.append('channel: %-5s \n' % (m.metaData.get('channel', 'xx')))
        t_spam.append('  median fw [s]: %-3.0f \n' % (m.metaData.get('medianFilterWidth', -1)))
        t_spam.append('  bandpass [Hz]: (%-3.0f, %-3.0f)  \n' % (lowcut, highcut))
    txt = ''.join(t_spam)
    # tx3 = ax.text(xlim[0]+0.1, ylim[0], txt, ha='left', va='bottom', ma='left', **txt_kwa_def)
    tx3 = ax.text(0.02, 0, txt, transform=ax.transAxes, ha='left', va='bottom', ma='left', **txt_kwa_def)

def plot2d(mx=None, my=None, ax=None, pmin=0.95):
    """scatter plot of EMG/EEG rms power    
    
    """
    txt_fontsize = 5
    xlim = [-2.5, 2.5]
    ylim = [-2.5, 2.5]

    ax.plot(mx.x, my.x, '-o', color='grey', lw=0.1, ms=1, alpha=0.6)

    # gussy it up
    ax.text(0.01, 0.99, 'Non-REM', ha='left', va='top', transform=ax.transAxes)
    ax.text(0.99, 0.01, 'Wake',    ha='right', va='bottom', transform=ax.transAxes)
    ax.text(0.99, 0.99, 'NOTSURE\n(+REM)', color='r', ha='right', va='top', transform=ax.transAxes)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(mx.metaData.get('channel', 'xx'))
    ax.set_ylabel(my.metaData.get('channel', 'xx'))

    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    ax.spines['top'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['right'].set_color('grey')

    # metadata info block

    plt_info_text_block(mx=mx, my=my, ax=ax)


def plot_exsum(mx, my, h2d=None, scores=None, df=None):
    """composing the whole figure (executive summary)
    
    input
    ------
    mx : (tsm1d) EMG time series w/ 1D GMM 
    my : (tsm1d) EEG time series w/ 1D GMM
    h2d : (Histo2D)
    scores : (ScoreBlock)
    df : (pd.DataFrame)
    """

    pmin = scores.df_index['pmin'].values[-1]

    consensus_scores = scores.data[2]
    ndx_u = scores.data[2] == 'NOTSURE'
    ndx_n = scores.data[2] == 'Non-REM'
    ndx_w = scores.data[2] == 'Wake'


    fig = plt.figure(figsize=(11, 8.5))
    ax_top = [plt.subplot(8,1,1), plt.subplot(8,1,2), plt.subplot(8,1,3), plt.subplot(8,1,4)]
    ax_bot = [plt.subplot(2,3,4), plt.subplot(2,3,5), plt.subplot(2,3,6)]

    #----------------
    # TOP HALF: time series stack
    Nchunks = 4
    for ip in range(Nchunks):
        plot_ts_chunk(mx=mx, my=my, ax=ax_top[ip], n=ip, N=Nchunks, scores=consensus_scores)

    #----------------
    # BOTTOM HALF: joint distribution, etc..
    # joint distribution scatter
    plot2d(mx=mx, my=my, ax=ax_bot[0], pmin=pmin )
    ax_bot[0].plot(mx.x[ndx_u], my.x[ndx_u], 'ro', ms=0.4, lw=0)
    ax_bot[0].plot(mx.x[ndx_n], my.x[ndx_n], 'bo', ms=0.4, lw=0)
    ax_bot[0].plot(mx.x[ndx_w], my.x[ndx_w], 'go', ms=0.4, lw=0)
    plot_gmm_overlay(mx=mx, my=my, ax=ax_bot[0], pmin=pmin)

    # joint distribution histogram
    plt_hist_kwa = dict(cmap='bone', levels='auto', ptype='imshow')
    pt.plot_2D_hist(h2d=h2d, ax=ax_bot[1], **plt_hist_kwa, cbar=False)
    plot_gmm_overlay(mx, my, ax=ax_bot[1], pmin=pmin)
    ax_bot[1].set_ylabel(None)

    # plot dataframe as a table in an axes
    cell_text = [df.iloc[i] for i in range(len(df))]
    tab = ax_bot[2].table(cellText=cell_text, colLabels=df.columns, loc='center')
    ax_bot[2].axis('off')
    tab.auto_set_column_width(col=list(range(len(df.columns))))
    tab.auto_set_font_size(False)
    tab.set_fontsize('xx-small')

    #----------------
    # GENERAL formatting
    plt.tight_layout(rect=(0,0,1,0.96), h_pad=1)

    return fig, ax_top, ax_bot


class AutoScorer(object):
    """conservative autoscoring using EMG and EEG RMS power

    attributes
    ------
    ft (str): featurization parameters (json file)
    edf (str): edf file
    trial (int): trial (mouse) name, usually 3-4 digit integer
    day (int): for when multiple days were recorded for the same trial (mouse)
    tag (str): nametag that combines trial and day
    pmin (float): threshold for GMM classification (see relevant methods)
    dest (str): folder/path for output
    std (StagedTrialData): computed features, plus some other goodies
    mx (TimeSeriesModel1D): 1D GMM for the x-axis (EMG)
    my (TimeSeriesModel1D): 1D GMM for the y-axis (EEG)
    scores_pred (ScoreBlock): predicted scores
    scores_frac (ScoreBlock): sleep state fractions (derived from scores_pred)

    TODO:
    """
    def __init__(self, data=None, ft=None,pmin=None, dest=None, append_tag=True):
        """
        args
        ------
        data (dict): required keys: edf, trial, day.
        """
        if data is None:
            raise Exception('data required')

        self.ft = ft

        self.edf = data.get('edf')
        self.trial = data.get('trial', -1)
        self.day = data.get('day', -1)
        self.tag = 'trial-%s-day-%i' % (self.trial, self.day)
        self.pmin = None if pmin is None else pmin
        self.dest = 'data-autoscore' if dest is None else dest

        if append_tag:
            self.dest = '%s-%s' % (self.dest, self.tag)
        os.makedirs(self.dest, exist_ok=True)

    def h2d(self):
        """make a Histo2D for the EMG/EEG features

        bin counts are normalized and log-scaled
        """
        X = np.vstack((self.mx.x, self.my.x))
        xedg = np.linspace(-2.5, 2.5, 61)
        yedg = np.linspace(-2.5, 2.5, 61)
        hist, _ = np.histogramdd(X.T, bins=[xedg, yedg])
        xcol = self.mx.metaData.get('channel', 'xx')
        ycol = self.my.metaData.get('channel', 'xx')
        h = rt.Histo2D(
            dims=[xcol, ycol],
            bin_edges=[xedg, yedg],
            hist=hist,
            varX=1,
            varY=1,
            ).normalize().logscale()
        return h

    def about(self):
        """"""
        print('-------- AutoScorer.about() --------')
        print('edf  :', self.edf)
        print('trial:', self.trial)
        print('day  :', self.day)

    def full_run(self, quiet=False):
        """run everything, quietly export the plot to file"""
        self.run()
        _,_,_ = self.plot(quiet=True, save=True)
        self.dump()
        ### print something?
        # print(self.scores_frac.about())
        return self

    def run(self, pmin=None):
        """featurize/train/score"""

        tagDict = dict(trial=self.trial, day=self.day)

        # override
        if pmin is None:
            pmin = self.pmin

        # STAGE n FEATURIZE (make this a method lolo, unpack ft earlier)
        std = stage(edf=self.edf, ft=self.ft, dest=self.dest, tagDict=tagDict)

        # TRAIN: make a 1D, two-state GMM model for x and for y
        mdx = std.stagingParameters["rmspower"][0]
        mdy = std.stagingParameters["rmspower"][1]
        mdx['freq'] = std.edf.freq
        mdy['freq'] = std.edf.freq
        dx = std.features.data[0]
        dy = std.features.data[1]
        tvec = (np.arange(len(dx))+0.5)*mdx.get('epochLength')
        mx = tsm1d.TimeSeriesModel1D(tvec, dx, metaData=mdx)
        my = tsm1d.TimeSeriesModel1D(tvec, dy, metaData=mdy)

        # CLASSIFY: carry out classification
        scores = qp_classifier(mx, my, pmin=pmin)
        scores.add_const_index_col(name='trial', value=self.trial, inplace=True)
        scores.add_const_index_col(name='day', value=self.day, inplace=True)

        # sleep state percentages for 1st(12A) and 2nd(12B) halfs of the day
        N = scores.numdatacols
        state_fracs_A = scores.mask(mask=slice(0, N//2), maskname='12A').count(frac=True)
        state_fracs_B = scores.mask(mask=slice(N//2, N), maskname='12B').count(frac=True)
        stk = state_fracs_A.stack(others=[state_fracs_B])

        # consolidated results
        self.std = std
        self.mx = mx
        self.my = my
        self.scores_pred = scores
        self.scores_frac = stk

    def dump(self, ):
        """dump features and scores"""
        self.std.to_json() #f=os.path.join(self.dest, 'staged-trial-data.json'))
        self.scores_pred.to_json(f=os.path.join(self.dest, 'scoreblock_predicted_scores.json'))
        self.scores_frac.to_json(f=os.path.join(self.dest, 'scoreblock_predicted_score_fractions.json'))

        self.scores_pred.to_sirenia_txt(
            str2num={'Wake':1, 'Non-REM':2, 'NOTSURE':100},
            f=os.path.join(os.path.join(self.dest, 'scores-qp-sirenia.txt')),
            row=2,
            )


    def plot(self, quiet=False, save=False):
        """
        quiet=True will suppress the figure popping up by temporarily switching
        to a non-interactive backend, which is useful for batch processing
        or jupyter notebooks
        """
        # switch to quiet backend?
        if quiet:
            b = plt.get_backend()
            plt.switch_backend('Agg')

        stk = self.scores_frac

        # dataframe with some useful information
        cols = ['pmin', 'trial', 'day', 'mask', 'Wake', 'Non-REM', 'NOTSURE']
        df = stk.df.copy()
        df[stk.data_cols] = (stk.data*1000//1)/10
        df_plt = df[df['feature'] == 'consensus'][cols]

        # plot
        fig, ax_top, ax_bot = plot_exsum(self.mx, self.my, h2d=self.h2d(), scores=self.scores_pred, df=df_plt)
        fig.suptitle('trial %s, day %i' % (self.trial, self.day), fontsize=20)
        if save:
            plt.savefig(os.path.join(self.dest, 'plotx-%s.png' % self.tag), dpi=300)

        # restore other backend
        if quiet:
            plt.switch_backend(b)

        return fig, ax_top, ax_bot


if __name__ == '__main__':

    print('hello world')

