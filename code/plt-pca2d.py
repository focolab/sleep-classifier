#!/usr/bin/env python3

import os
import argparse
import json
import pdb
#import pickle
import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

import scoreblock as sb
import remtools as rt
import plottools as pt


sns.set(color_codes=True)
sns.set_style('ticks')



if __name__ == '__main__':
    """
    plotting feature data projected into PC space


    TODO: load scores
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='staged trial data json files')
    parser.add_argument('-p', default=None, type=str, help='pca json')
    parser.add_argument('-s', default=None, type=str, help='scoreblock.json')
    parser.add_argument('--dest', default='ANL-plt-pca2d', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    print('#=================================================================')
    print('#                        plt-pca-2d.py')
    print('#=================================================================')

    print('plt-pca-2d.py: SHOULD TAKE stds and a scoreblock (optional) as inputs')
    print('plt-pca-2d.py: scoreblock merging/munging should be done beforehand')



    labels = ['REM', 'Non REM', 'Wake', 'XXX']

    pca_hist_kwa = dict(
        PCs=[1, 2],
        numsig=3,
        numbin=60,
        cmap='Greys_r',
        #cmap='rocket',
        log=True,
        normalize=True,
        levels='auto',
    )

    text_kwa = dict(color='white', fontsize=10)
    ellipse_kwa = dict(zorder=1, alpha=1, lw=3, fill=False)
    mu_kwa = dict(marker='o', mec='k')
    legend_line_kwa = dict(mec='none', marker='s', ms=12, lw=0)
    legend_kwa = dict(fontsize=10, framealpha=0, frameon=False)
    point_kwa = dict(lw=0, marker='o', ms=2, mec='none', color='magenta', alpha=1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # LOAD
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]


    #-------------------------------------
    # MUNGING
    # for each trial:
    #   std
    #   h2d
    #   df_xys
    #   levels
    # TODO: sigx and sigy should be packed into h2d?
    #

    data = []
    if args.p is not None:

        # make PCA projections, 2D histograms, xy-score dataframes
        pca = rt.PCA.from_json(args.p)
        for std in allTrialData:
            X = std.features.data
            print('precomputing h2d for', std.tagDict)
            h2d = pca.project_histo(
                data=X,
                PCs=pca_hist_kwa['PCs'],
                numsig=pca_hist_kwa['numsig'],
                numbin=pca_hist_kwa['numbin'],
                )

            tiny = 0.6
            if pca_hist_kwa['normalize']:
                tiny = tiny/np.sum(h2d.hist.ravel())
                h2d = h2d.normalize()
            if pca_hist_kwa['log']:
                h2d = h2d.logscale(tiny=tiny)


            # dataframe of xy data and scores
            df_xys = pca.project(data=X, PCs=pca_hist_kwa['PCs'])

            # split data by scores
            if args.s is not None:
                scores = sb.ScoreBlock.from_json(args.s).keeprows(conditions=[('trial',trial)])
            elif std.scoreblock is not None:
                scores = std.scoreblock

            ndx = scores.df['scorer'] == 'consensus'
            df_xys['scores'] = scores.df[ndx][scores.data_cols].values.ravel()

            # pack it up
            dd = dict(
                std=std,
                h2d=h2d,
                df_xys=df_xys
            )
            data.append(dd)
    else:
        # TODO: compute Histo2D directly from features
        pass


    col1 = 'PC%i' % pca_hist_kwa['PCs'][0]
    col2 = 'PC%i' % pca_hist_kwa['PCs'][1]


    # automatic range limits (inefficient, but convenient for multiple plots)
    if pca_hist_kwa.get('levels', None) in ['auto', None]:
        cmin, cmax = np.inf, -np.inf
        for dd in data:
            h2d = dd['h2d']
            lims = h2d.range
            cmin = min(cmin, lims[0])
            cmax = max(cmax, lims[1])
        if pca_hist_kwa['log'] == True:
            cmin = np.floor(cmin)
            cmax = np.ceil(cmax)
            levels = np.arange(cmin, cmax+1)
        else:
            levels = np.linspace(cmin, cmax, 7)

        pca_hist_kwa['levels'] = levels



    #-------------------------------------
    # MAIN: plot each trial
    for dd in data:

        std = dd['std']
        h2d = dd['h2d']
        df_xys = dd['df_xys']

        trial = std.tagDict.get('trial', 'tt')
        gt = std.tagDict.get('genotype', 'gt')
        tag = 'GT-%s-trial-%s' % (str(gt), str(trial))
        print('plotting: %s' % (tag))


        #-------------------
        # score-wise RAW DATA on top of projected distribution
        nrow = len(labels)
        ncol = 1
        fig = plt.figure(figsize=(6, 4*nrow))
        ax = [plt.subplot(nrow, ncol, i+1) for i in range(nrow*ncol)]

        for i, label in enumerate(labels):

            dfi = df_xys[df_xys['scores'] == label]
            frac = 100.0*len(dfi)/len(df_xys)
            ntag = 'N=%i  (%2.1f %%)' % (len(dfi), frac)
            pt.plot_2D_hist(h2d=h2d, ax=ax[i], ptype='imshow', cmap=pca_hist_kwa['cmap'], levels=levels)

            # gussy it up (PCA SPECIFIC)
            sigX = np.sqrt(pca.vals[pca_hist_kwa['PCs'][0]-1])
            sigY = np.sqrt(pca.vals[pca_hist_kwa['PCs'][1]-1])
            pt.plot_pca_crosshair(ax=ax[i], sigX=sigX, sigY=sigY)

            ax[i].plot(dfi[col1], dfi[col2], **point_kwa)

            ax[i].text(0.02, 0.98, label, ha='left', va='top', transform = ax[i].transAxes, **text_kwa)
            ax[i].text(0.98, 0.98, ntag, ha='right', va='top', transform = ax[i].transAxes, **text_kwa)

            # gussy it up
            if i<nrow-1:
                ax[i].set_xticklabels([])
                ax[i].set_xlabel(None)

        # meta pimping
        ax[0].set_title('genotype/trial %s %s' % (str(gt), str(trial)))

        txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        fig.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(args.dest, 'plt-2D-PCA-histogram-scores-%s.png') % (tag))
        plt.savefig(os.path.join(args.dest, 'plt-2D-PCA-histogram-scores-%s.svg') % (tag))
        plt.close()


        #-------------------
        # score-wise mean/cov blobs on top of projected distributions
        nrow = 2
        ncol = 1
        fig = plt.figure(figsize=(6, 8))
        ax = [plt.subplot(nrow, ncol, i+1) for i in range(nrow*ncol)]

        pt.plot_2D_hist(h2d=h2d, ax=ax[0], ptype='imshow', cmap=pca_hist_kwa['cmap'], levels=levels)
        pt.plot_2D_hist(h2d=h2d, ax=ax[1], ptype='imshow', cmap=pca_hist_kwa['cmap'], levels=levels)

        # gussy it up (PCA SPECIFIC)
        sigX = np.sqrt(pca.vals[pca_hist_kwa['PCs'][0]-1])
        sigY = np.sqrt(pca.vals[pca_hist_kwa['PCs'][1]-1])
        pt.plot_pca_crosshair(ax=ax[0], sigX=sigX, sigY=sigY)
        pt.plot_pca_crosshair(ax=ax[1], sigX=sigX, sigY=sigY)

        # LEGEND
        labels_colors = zip(labels, colors)
        handles = [Line2D([0], [0], color=c, label=l,  **legend_line_kwa) for l,c in labels_colors]
        leg = dict(
            handles=handles, **legend_kwa)

        for i, label in enumerate(labels):
            # use PCA machinery to create 2sigma ellipses
            try:
                dfi = df_xys[df_xys['scores'] == label]
                pcai = rt.PCA.from_data(dfi[[col1, col2]].values.T)

                # plot mean and 2sigma ellipse
                kwa = dict(color=colors[i])
                kwa.update(ellipse_kwa)
                ax[1].add_artist(pcai.ellipse(w=2, PCs=[1, 2], elkwa=kwa))
                ax[1].plot(pcai.mu[0], pcai.mu[1], color=colors[i], **mu_kwa)
            except:
                pass

            txt = r'contours indicate $2\sigma$'
            ax[1].text(0.02, 0.98, txt, ha='left', va='top', transform=ax[1].transAxes, **text_kwa)

        mylegend = ax[1].legend(**leg)
        plt.setp(mylegend.get_texts(), color='w')

        # pimping
        ax[0].set_xticklabels([])
        ax[0].set_xlabel(None)
        ax[0].set_title('genotype/trial %s %s' % (str(gt), str(trial)))

        txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        fig.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(args.dest, 'plt-2D-PCA-class-blobs-%s.png') % (tag))
        plt.savefig(os.path.join(args.dest, 'plt-2D-PCA-class-blobs-%s.svg') % (tag))
        plt.close()




