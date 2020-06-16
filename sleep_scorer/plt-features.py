#!/usr/bin/env python3

"""
TODO:
    - should be focused to plotting the power spectrum? (w light processing)
    - factor plots into functions
"""

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
import seaborn as sns

from matplotlib.lines import Line2D
import matplotlib.patches as patches

#import modeltools as mt
import plottools as pt
import remtools as rt
import scoreblock as sb


sns.set(color_codes=True)
sns.set_style('ticks')




def plot_feature_power(df_feat=None):
    """raster plot of feature power distributions (not state resolved)"""
    xpad = 0

    unique_scores = ['all']

    df_ndx = df_feat.index.to_frame().reset_index(drop=True)
    channels = df_ndx['channel'].unique()
    gt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # make the badass template
    fig, ax, channel_info = pt.plot_features_template(
        df_feat_index=df_ndx, 
        unique_scores=unique_scores,
        xpad=xpad
        )


    # histogram of df_feat
    xdom = np.linspace(0, 0.8, 161)
    data = np.asarray([np.histogram(x, bins=xdom )[0] for x in df_feat.values]).T
    dplt = np.log(data+0.01)

    left = 0-0.5
    right = len(df_ndx)-0.5
    bottom = 0
    top = 0.8

    extent = (left, right, bottom, top)

    ax[0].imshow(dplt, origin='lower', vmax=9, vmin=0, aspect='auto', extent=extent)
    ax[0].set_ylabel('feature value')

    plt.tight_layout(h_pad=0.1)
    return fig, ax



def plot_features(df_feat=None, df_scores=None):
    """plot features for a single trial, grouped by score (sleep state)
    
    
    TODO: use df_feat.index to construct the panels/axes -- sans data
    """

    make_features_integers = True

    xpad = 2

    scores = df_scores.values.ravel()
    unique_scores = np.unique(scores)


    df_ndx = df_feat.index.to_frame().reset_index(drop=True)
    channels = df_ndx['channel'].unique()
    gt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # make the template
    fig, ax, channel_info = pt.plot_features_template(
        df_feat_index=df_ndx, 
        unique_scores=unique_scores,
        xpad=xpad
        )

    # populate the template
    # the main loop enumerates sleep states
    for i, ss in enumerate(unique_scores):

        # isolate class/score specific feature vectors
        cols = [df_feat.columns[j] for j in range(len(scores)) if scores[j]==ss]
        dfi = df_feat[cols] 
        Xi = df_feat[cols].values

        print(i, ss, '(N=%i)' %(len(cols)))

        # one channel at a time
        for dd in channel_info:
            # plot features data, single epochs and the average
            ax[i].plot(dd['xndx'], Xi[dd['ndx']], lw=0.5, alpha=0.02, color=gt_colors[i])
            ax[i].plot(dd['xndx'], np.mean(Xi[dd['ndx']], axis=1), lw=3, color='w')
            ax[i].plot(dd['xndx'], np.mean(Xi[dd['ndx']], axis=1), lw=2, color=gt_colors[i])
            #ax[i].plot(dd['ndx'], dd['ndx']*0, lw=1, color='gray')


        ax[i].set_ylim([-0.05, 0.4])

        # ax[i].set_ylabel(ss)
        # ax[i].spines['top'].set_visible(False)
        # ax[i].spines['right'].set_visible(False)            
        # ax[i].spines['bottom'].set_visible(False)

        # # ticks
        # if i+1<len(ax):
        #     ax[i].set_xticklabels([])
        #     ax[i].set_xticks([])
        # else:
        #     ax[i].set_xticks(xtk)
        #     ax[i].set_xticklabels(xtkl, rotation='vertical')
        #     ax[i].set_xlabel('f [Hz]')




    plt.tight_layout(h_pad=0.1)

    return fig, ax

if __name__ == '__main__':
    """plotting model features"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='staged trial data json files')
    parser.add_argument('--dest', default='ANL-plt-features', help='output folder')
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    print('#=================================================================')
    print('#                        plt-features.py')
    print('#=================================================================')

    scorer = 'consensus'

    # load featurized data
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]

    # PLOTTING
    for std in allTrialData:
        df_feat = std.features.dfmi

        # plot feature vectors for each trial, split by sleep state
        sb_scores = std.scoreblock
        # sb_scores.about()
        df_scores = sb_scores.df.copy().set_index(sb_scores.index_cols)
        dfs = df_scores[df_scores.index.get_level_values('scorer') == scorer]

        fig, ax = plot_features(df_feat=df_feat, df_scores=dfs)
        txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        fig.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)
        ax[0].set_title('trial %s (scorer: %s)' % (str(std.trial), scorer))
        plt.savefig(os.path.join(args.dest, 'plot-features-trial-%s.png' % str(std.trial)), dpi=300)



#==============================================================================
#=========================== graveyard ========================================
#==============================================================================


    # #=========================================

    # # feature blocks, for average power distributions
    # blocks = [
    #     ('EEG1', [2.0, 2.5, 3.0, 3.5], 'EEG1-delta-[2,4)'),
    #     ('EEG2', [2.0, 2.5, 3.0, 3.5], 'EEG2-delta-[2,4)'),
    #     ('EEG1', [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 'EEG1-Theta-[4,7]'),
    #     ('EEG2', [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 'EEG2-Theta-[4,7]'),
    #     ('EMG', [100., 102., 104.], 'EMG-[100,104]'),
    # ]

    # # power bins
    # xdom = np.linspace(0, 0.8, 161)
    # bin_L = xdom[:-1]
    # bin_R = xdom[1:]
    # bin_C = (bin_L+bin_R)/2.0
    # dd = np.asarray([bin_L, bin_R, bin_C]).T
    # df_bins = pd.DataFrame(data=dd, columns=['binL', 'binR', 'binC'])

    # block_pwr_data = []
    # for std in allTrialData:
    #     df_index = std.features.df_index

    #     data = []
    #     # power dist for each block
    #     for b in blocks:
    #         # filter by channel/freq
    #         val1 = df_index['channel'] == b[0]
    #         val2 = np.isin(df_index['f[Hz]'], b[1])

    #         x = std.features.data[val1 & val2].ravel()

    #         # histogram
    #         h = np.histogram(x, bins=xdom)[0]
    #         data.append(h)


    #     cols = ['bin-%3.3i' % i for i in np.arange(len(bin_C))]
    #     dft = pd.DataFrame(data=data, columns=cols)
    #     dft['block'] = [b[2] for b in blocks]
    #     dft['trial'] = [std.trial]*len(dft)

    #     dft.set_index(['trial','block'], inplace=True)

    #     block_pwr_data.append(dft)

    # df_block_power = pd.concat(block_pwr_data, axis=0)


#     for std in allTrialData:
#         #raise Exception('use std.features instead of sxxb_prep')
#         #df_feat = std.sxxb_prep.to_dataframe()
#         df_feat = std.features.dfmi

#         # plot feature POWER for each trial, NOT split by sleep state
#         fig, ax = plot_feature_power(df_feat=df_feat)
#         txt = datetime.datetime.now().replace(microsecond=0).isoformat()
#         fig.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)
#         ax[0].set_title('trial %s' % (str(std.trial)))
# #        fig.suptitle('trial %s' % (str(std.trial)))
#         plt.savefig(os.path.join(args.dest, 'plot-features-power-trial-%s.png' % str(std.trial)), dpi=300)



    # #=========================================================
    # fig = plt.figure(figsize=(12,4))
    # nrow = 1
    # ncol = len(blocks)
    # ax = [plt.subplot(nrow, ncol, i+1) for i in range(nrow*ncol)]


    # for i, b in enumerate(blocks):
    #     rows = df_block_power.index.get_level_values('block') == b[2]

    #     cols = ['bin-%3.3i' % i for i in np.arange(len(bin_C))]

    #     df = df_block_power[rows]
    #     pltdata = df.values.T
        
    #     trial_names = [t.trial for t in allTrialData]
    #     for j, row in enumerate(pltdata.T):
    #         ax[i].plot(row, label=trial_names[j])

    #     ax[i].plot(row*0, ls='--', color='gray', alpha=0.5)
    #     #pdb.set_trace()
    #     # bpd =
    #     # ax
    #     #plt.yscale('symlog')

    #     if i==0:
    #         ax[i].set_ylabel('N(signal power)')

    #     if i>=0:
    #         #ax[i].spines['top'].set_visible(False)
    #         ax[i].set_yticklabels([])

    #     if i == len(blocks)-1:
    #         #ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    #         ax[i].legend(ncol=1, fontsize=8)

    #     ax[i].set_xlim([0,20])
    #     ax[i].set_xlabel('signal power')
    #     ax[i].set_title(b[2])

    # plt.tight_layout()
    # txt = datetime.datetime.now().replace(microsecond=0).isoformat()
    # fig.text(0.99, 0.99, txt, ha='right', va='top', fontsize=12)
    # plt.savefig(os.path.join(args.dest, 'plot-features-blockpower.png'), dpi=300)










    
