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
    dfh = df[df['scorer'] == 'ANON']
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


if __name__ == '__main__':
    """
    plotting classifier sleep-state predictions

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='prediction.json')
    # parser.add_argument('-s', type=str, help='scoreblock.json')
    parser.add_argument('--dest', default='ANL-plt-predictions-v2', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    print('#=================================================================')
    print('#                        plt-predict-v2.py')
    print('#=================================================================')

    raise Exception('deprecated, use plt-scores.py')

    #========================= IMPORTS =======================
    with open(args.f) as jfopen:
        jdic = json.load(jfopen)
    # load the data
    loc = jdic['loc']
    # import the prediction data
    csv = os.path.join(loc, jdic['df_pred_csv'])
    df_pred = pd.read_csv(csv, index_col=0)
    sb_pred = sb.ScoreBlock(df=df_pred, index_cols=jdic['index_cols'])


    # sb_pred = sb.ScoreBlock.from_json(args.s)
    # df_pred = sb_pred.df
    # data_cols = sb_pred.data_cols
    # cols_data = data_cols

    #===================================================
    # WWRW
    # 1. import
    # 2. counts/fractions
    # 3. restack
    # 4. plot
    #===================================================


    # make column masks, compute state fractions, stack
    num_epochs = len(sb_pred.data_cols)
    maskAM = slice(0, num_epochs//2)
    maskPM = slice(num_epochs//2, num_epochs)

    sb_frac_all = sb_pred.count(maskname='24h', frac=True) 
    sb_frac_am = sb_pred.count(mask=maskAM, maskname='12hAM', frac=True)
    sb_frac_pm = sb_pred.count(mask=maskPM, maskname='12hPM', frac=True)

    db_stack = sb_frac_all.stack(others=[sb_frac_am, sb_frac_pm])


    db_stack.about()

    hack_down_data = True
    if hack_down_data:
        # drop Unscored
        df = db_stack.df.copy()

        #pdb.set_trace()
        df.drop(columns='Unscored', inplace=True)

        # add Non REM and Non REM X
        nr = df['Non REM'].values + df['Non REM X'].values
        df['Non REM'] = nr
        df.drop(columns='Non REM X', inplace=True)

        # keep anon human scorer and OVR
        df_anon = df[df['scorer'] == 'ANON']
        df_OVR_04 = df[(df['classifier'] == 'OVR') & (df['M'] == '04h_8ch')]
        df_OVR_24 = df[(df['classifier'] == 'OVR') & (df['M'] == '24h_8ch')]

        df_pred = pd.concat([df_anon, df_OVR_24]).reset_index(drop=True)




    #========================= FRACTION PLOTS =================================
    label_names = ['Non REM', 'REM', 'Wake']
    fig, ax = make_frac_plot(df=df_pred, labels=label_names)
    plt.savefig(os.path.join(args.dest, 'plot-sleep_fractions.png'), dpi=300)

    #pdb.set_trace()


    #========================= RASTER PLOTS ===================================
    # dictionary to convert labels to numbers
    # label, value, rgb

    mapp = {'Non REM':0, 'REM':0.5, 'Wake':1}

    labels = np.unique(df_pred[cols_data]).tolist()
    vals = np.arange(len(labels))/(len(labels)-1.)
    #mapp = dict(zip(labels, vals))

    fmap = lambda x: mapp[x]

    # colors, labels, numbers
    lbl2num = dict(zip(labels, vals))
    thiscmap = matplotlib.cm.get_cmap('rocket')
    colors = [thiscmap(v) for v in vals]
    labels_colors = list(zip(labels, colors))


    # custom legend
    kwa = dict(mec='gray', marker='s', lw=0)
    handles = [Line2D([0], [0], color=c, label=l, **kwa) for l,c in labels_colors]
    leg = dict(handles=handles, fontsize=8)

    # ROW SORT by label (gives blocks)
    dsrt = np.sort(df_pred[cols_data].applymap(fmap).values, axis=1)
    #dsrt = df_pred[cols_data].applymap(fmap).values
    data = df_pred[cols_data].applymap(fmap).values
    
    fig, ax = pt.montage_raster(df_index=df_pred[cols_index], data=dsrt, leg=leg, aspect=200, wtf=True)
    plt.savefig(os.path.join(args.dest, 'plot-raster-sortedTS.svg'))
    plt.savefig(os.path.join(args.dest, 'plot-raster-sortedTS.png'))

    fig, ax = pt.montage_raster(df_index=df_pred[cols_index], data=data, leg=leg, aspect=200, wtf=True)
    plt.savefig(os.path.join(args.dest, 'plot-raster-sequentialTS.svg'))
    plt.savefig(os.path.join(args.dest, 'plot-raster-sequentialTS.png'))





