#!/usr/bin/env python3
#
#   
#

import os
import argparse
import pdb
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns
from scipy.spatial import distance

import remtools as rt


sns.set(color_codes=True)
sns.set_style('ticks')


def jsd(histos=None):
    """Jensen Shannon distance matrix for list of histograms"""
    n = len(histos)
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            hi = histos[i]
            hj = histos[j]
            jsd = distance.jensenshannon(hi.ravel(), hj.ravel())
            s[i,j] = jsd
            s[j,i] = jsd
    return s



if __name__ == '__main__':
    """similarity comparison for featurized data (>=1 trials)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='staged data json files')
    parser.add_argument('-p', type=str, help='pca json')
    parser.add_argument('--dest', default='ANL-similarity', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    print('#=================================================================')
    print('           anl-similarity.py')
    print('#=================================================================')

    # params
    prj_kwa = dict(numbin=80, PCs=[1,2,3])
    num_levels = 10


    # loading
    allTrialData = [rt.StagedTrialData.from_json(f, loadEDF=False) for f in args.f]
    pca = rt.PCA.from_json(args.p)

    # project features and make histograms
    histos = []
    for std in allTrialData:
        X = std.sxxb_prep.stack        
        td = std.tagDict
        hh = pca.project_histo(data=X, tagDict=td, **prj_kwa)
        histos.append(hh)

    # jensen shannon distance
    s = jsd(histos=[h.hist for h in histos])
    df_index = pd.DataFrame([h.tagDict for h in histos])


    # export


    def hclust(dmat=None, thresh=0.1, method='average'):
        """heirarchical clustering based sort"""
        from scipy.cluster.hierarchy import linkage, fcluster

        size = dmat.shape[0]
        duniq = dmat[np.triu_indices(size, k=1)]
        clustering = fcluster(linkage(duniq, method), t=thresh, criterion='distance')
        c = clustering -1
        ndxsort = np.argsort(c)

        return ndxsort
        
    def matsort(m=None, ndx=None):
        """sort rows and columns of a 2D array by ndx"""
        return m[ndx].T[ndx].T



    # indexing and sorting
    df_index_sorted = df_index.sort_values(by=['genotype', 'trial']) 
    ndx = df_index_sorted.index.values

    #ndx = hclust(dmat=s, thresh=0.25)

    # sort matrix and index
    s = matsort(m=s, ndx=ndx)
    df_index = df_index.iloc[ndx].reset_index()

    # tags
    tt = list(zip(df_index['genotype'], df_index['trial']))
    tags = ['%s-%s' % t for t in tt]



    #-------------------------------------------
    # PLOTS
    cmap = 'viridis'
    colorkwa=dict(fraction=0.04)
    cbar = True
    title = 'Jensen-Shannon distance'
    levels = np.linspace(s.min(), s.max(), num_levels+1)



    cmapx = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, ncolors=cmapx.N, clip=True)


    fig, ax = plt.subplots()
    im = ax.imshow(s, cmap=cmap, norm=norm)
    if cbar:
        ax.figure.colorbar(im, ax=ax, **colorkwa)


    ax.set(xticks=np.arange(s.shape[1]),
           yticks=np.arange(s.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=tags, yticklabels=tags,
           title=title,
           ylabel='genotype-trial',
           xlabel='genotype-trial')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    txt = datetime.datetime.now().replace(microsecond=0).isoformat()
    fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(args.dest, 'plot-jsd.png'))


