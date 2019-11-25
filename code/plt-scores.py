#!/usr/bin/env python3

import os
import argparse
import json
import pdb
import pickle
import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
import seaborn as sns

from matplotlib.lines import Line2D

import modeltools as mt
import plottools as pt
import scoreblock as sb


sns.set(color_codes=True)
sns.set_style('ticks')



def make_frac_plot(df=None, labels=None):
    """ sleep state fraction plots

    df columns:
        genotype
        mask: AM/PM/etc
    """

    np.random.seed(12345)

    #== reduce the data (dirty hack)
    dfh = df[df['scorer'] == 'consensus']
    df = df[df['classifier'] == 'OVR']

    #== convert to fractions
    # rowsums = np.sum(data, axis=1)
    # for col in labels:
    #     df[col] /= rowsums

    genotypes = df['genotype'].unique()
    daychunks = df['mask'].unique()


    figx = plt.figure(figsize=(8,3))

    nrow = 1
    ncol = 3
    ax = [plt.subplot(nrow, ncol, i+1) for i in range(nrow*ncol)]

    gt_markers = ['o','o']
    gt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, label in enumerate(labels):
        xtk = []
        xtkl = []

        for j, tod in enumerate(daychunks):
            xtk.append(j)
            xtkl.append(tod)
            
            yavgs = []
            xavgs = []
            for k, gt in enumerate(genotypes):

                dxM = (k-0.5)*0.2
                dxH = (k-0.5)*0.4


                #----------------------------------
                # MODEL PREDICTIONS
                dx = (k-0.5)*0.2
                dfijk = df[(df['genotype']==gt) & (df['mask']==tod)]

                yvals = dfijk[label].values
                yavg = np.mean(yvals)
                ystd = np.std(yvals)
                xavgs.append(j+dx)
                yavgs.append(yavg)
                xvals = [j + dx]*len(yvals)+np.random.randn(len(yvals))*0.02
                
                # pdb.set_trace()

                if j == 0:
                    tag = 'GT%s OVR (n=%i)' % (gt, len(xvals))
                else:
                    tag = None

                kwa = dict(
                    alpha=0.6,
                    marker='o',
                    color=gt_colors[k],
                    edgecolors='none',
                    s=20
                    )

                ax[i].plot([j+dx,j+dx],[yavg-ystd, yavg+ystd], lw=1.5, color='grey', marker=None)
                ax[i].scatter(xvals, yvals, label=tag, **kwa)


                trials = dfijk['trial'].values
                for x,y,t in zip(xvals, yvals, trials):
                    ax[i].text(x, y, t, fontsize=6)

                #----------------------------------
                # HUMAN PREDICTIONS
                dx = (k-0.5)*0.5
                dfijk = dfh[(dfh['genotype']==gt) & (dfh['mask']==tod)]

                yvals_h = dfijk[label].values
                yavg = np.mean(yvals_h)
                ystd = np.std(yvals_h)
                xavgs.append(j+dx)
                yavgs.append(yavg)
                xvals_h = [j + dx]*len(yvals_h)+np.random.randn(len(yvals_h))*0.02

                if j == 0:
                    tag = 'GT%s human (n=%i)' % (gt, len(xvals))
                else:
                    tag = None

                kwa = dict(
                    alpha=0.6,
                    marker='^',
                    color=gt_colors[k],
                    edgecolors='none',
                    s=20
                    )

                ax[i].plot([j+dx,j+dx],[yavg-ystd, yavg+ystd], lw=1.5, color='grey', marker=None, zorder=1)
                ax[i].scatter(xvals_h, yvals_h, label=tag, **kwa, zorder=2)

                # connect human and model values
                for pp in range(len(xvals)):
                    xx = [xvals[pp], xvals_h[pp]]
                    yy = [yvals[pp], yvals_h[pp]]
                    #ax[i].plot(xx, yy, lw=1, color='grey', marker=None, zorder=1, alpha=0.6)

            #ax[i].plot(xavgs, yavgs, lw=2, color='grey', marker=None, zorder=1, alpha=0.6)


        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)            
        ax[i].set_xticks(xtk)
        ax[i].set_xticklabels(xtkl)
        ax[i].set_title(label)
        ax[i].set_ylim([0, ax[i].get_ylim()[1]])
        # if label == 'Wake':
        #     ax[i].set_ylim([0.35, 0.65])

        ax[i].set_xlabel('data interval')
        if i == 0:
            ax[i].legend(fontsize=6)
            ax[i].set_ylabel('fraction of time in sleep state')


    txt = datetime.datetime.now().replace(microsecond=0).isoformat()
    figx.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)

    plt.tight_layout()

    return figx, ax


def plot_tsbands(df=None, aspect=1.0, cmap='viridis', ax=None, zorder=0, cbar=True):
    ''' plot stacks of time series as horizontal colored bands

    teleported from gizmo project

    input
    ---
        df  
        pandas dataframe to be plotted. All columns are shown, so 
        
        aspect
        aspect ratio 
        
    TODO: colorbar..
    TODO: extent
    '''
    if ax is None:
        ax = plt.gca()
    
    xlbl = 'index' if df.index.name is None else df.index.name
    ncol = len(df.columns)
    nrow = len(df.index)
    
    #== scale aspect by data dimensions, giving a square plot, then multiply by aspect
    asp = float(nrow)/float(ncol)*aspect
    
    
    #== the plot
    cax = ax.imshow(df.values.T, aspect=asp, cmap=cmap, origin='lower', zorder=zorder)

    #== set all the ticks (x and y)
    xtk = ax.get_xticks()[1:-1]
    xtk = np.asarray(xtk, dtype=int)
    xtklbl = df.index.values[xtk]
    ax.set_xticks(xtk)
    ax.set_xticklabels(xtklbl)
    ax.set_yticks(range(ncol))
    ax.set_yticklabels(df.columns)
    ax.grid(False)
    ax.set_xlabel(xlbl)

    if cbar:
        cb = plt.colorbar(cax, fraction=0.1, shrink=0.5, orientation='vertical')

    print('#------------------------')
    print('nrow:', nrow)
    print('ncol:', ncol)
    print('shpe:', df.shape)
    print('zmin:', np.nanmin(df.values))
    print('zmax:', np.nanmax(df.values))
    
    return cax, cb


def make_score_leg(d=None, fontsize=8, cmap='rocket', leg_line_kwa={}):
    """make a legend for categorical score data (converted to numbers)
    
    input
    ------
    d (dict): mapping from score to number. Numbers must be in [0,1)

    """

    labels = list(d.keys())
    vals = list(d.values())

    if np.max(vals) >=1 :
        raise Exception('color values must be in [0,1)')

    # colors, labels, numbers
    thiscmap = matplotlib.cm.get_cmap(cmap)
    colors = [thiscmap(v) for v in vals]
    labels_colors = list(zip(labels, colors))

    # for k,v in labels_colors:
    #     print(k, v)

    # custom legend
    handles = [Line2D([0], [0], color=c, label=l, **leg_line_kwa) for l,c in labels_colors]
    leg = dict(handles=handles, fontsize=fontsize)

    return leg


if __name__ == '__main__':
    """
    plotting sleep-state predictions, man vs machine

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help='scoreblock.json')
    parser.add_argument('--dest', default='ANL-plt-scores', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    print('#=================================================================')
    print('#                        plt-scores.py')
    print('#=================================================================')

    # params
    keep_human = dict(
        #conditions=[('scorer', 'ANON')]
        conditions=[('scorer', 'consensus')]
        )
    keep_model = dict(
        conditions=[('classifier', 'OVR'), ('M', '24h_8ch')], 
        comparison='all'
        )

    mapp = {'Non REM X':'Non REM'}

    # load
    sb_pred = sb.ScoreBlock.from_json(args.s)

    # fix bogus scores (e.g. NRX)
    sb_pred = sb_pred.applymap(mapp)

    # keep some rows (human vs OVR scores)
    kr_hum = sb_pred.keeprows(**keep_human)
    kr_mdl = sb_pred.keeprows(**keep_model)
    kr_stack = kr_hum.stack(others=[kr_mdl])

    sb_pred = kr_stack


    # make column masks, compute state fractions, stack
    num_epochs = len(sb_pred.data_cols)
    maskAM = slice(0, num_epochs//2)
    maskPM = slice(num_epochs//2, num_epochs)

    sb_frac_all = sb_pred.mask(maskname='24h').count( frac=True) 
    sb_frac_am = sb_pred.mask(mask=maskAM, maskname='12hAM').count(frac=True)
    sb_frac_pm = sb_pred.mask(mask=maskPM, maskname='12hPM').count(frac=True)

    sb_stack_frac = sb_frac_all.stack(others=[sb_frac_am, sb_frac_pm], data_nan=0)

    #========================= FRACTION PLOTS =================================
    label_names = ['Non REM', 'REM', 'Wake']
    fig, ax = make_frac_plot(df=sb_stack_frac.df, labels=label_names)
    plt.savefig(os.path.join(args.dest, 'plot-sleep_fractions.png'), dpi=300)


    #========================= RASTER PLOTS ===================================
    mapp = {'Non REM':0, 'REM':0.5, 'Wake':0.99, 'Unscored':0.25}
    montage_kwa = dict(
        panelKey='trial',
        labelKeys=['classifier','M'],
        cmap='rocket',
        aspect=200,
        wtf=True
    )
    leg_line_kwa = dict(mec='gray', marker='s', lw=0)
    leg_kwa = dict(cmap='rocket', fontsize=8)


    # rows unsorted and sorted by label (gives blocks)
    sb_num = sb_pred.applymap(mapp)
    data = sb_num.data
    dsrt = np.sort(sb_num.data, axis=1)

    # legend
    leg = make_score_leg(d=mapp, **leg_kwa, leg_line_kwa=leg_line_kwa)

    fig, ax = pt.montage_raster(
        df_index=sb_num.df_index,
        data=dsrt,
        leg=leg,
        **montage_kwa
        )
    plt.savefig(os.path.join(args.dest, 'plot-raster-sortedTS.svg'))
    plt.savefig(os.path.join(args.dest, 'plot-raster-sortedTS.png'))

    fig, ax = pt.montage_raster(
        df_index=sb_num.df_index,
        data=data,
        leg=leg,
        **montage_kwa
        )
    plt.savefig(os.path.join(args.dest, 'plot-raster-sequentialTS.svg'))
    plt.savefig(os.path.join(args.dest, 'plot-raster-sequentialTS.png'))





