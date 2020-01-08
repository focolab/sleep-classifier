#!/usr/bin/env python3
#
#
#
#======================================
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
from matplotlib.colors import ListedColormap
import seaborn as sns

import modeltools as mt
import plottools as pt

from modeltools import ClassifierBundle

sns.set(color_codes=True)
sns.set_style('ticks')



def plot_chop8640(num_total=None, num_keep=None, num_chunks=None,
                  xscl=1./360., xlbl='time[h]', 
                  center=True, out='plot-chop8640.png', kwa={}):
    """plot the binary mask for subsampling 8640 epochs"""

    vec = mt.chop8640(num_total=num_total,
                   num_keep=num_keep, 
                   num_chunks=num_chunks, 
                   center=center)

    vec = np.asarray(vec)

    xx = np.arange(num_total)*xscl

    plt.figure(figsize=(8, 1.2))
    plt.plot(xx, vec, **kwa)
    #plt.fill(vec, color=kwa.get('color'), alpha=0.5)
    
    plt.fill_between(xx, vec*0, vec, facecolor=kwa.get('color'), alpha=0.5)
    
    #facecolor='blue', alpha=0.5
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig(out, dpi=600)

    return




def make_legend(df_legend=None, cmap='rocket'):
    """
    custom legend

    label, value, [color]
    """
    from matplotlib.lines import Line2D

    if 'colors' not in df_legend.columns:
        thiscmap = matplotlib.cm.get_cmap(cmap)
        df_legend['color'] = [thiscmap(v) for v in df_legend['value']]

    labels_colors = df_legend[['label', 'color']].values

    print(df_legend)

    kwa = dict(mec='gray', marker='s', lw=0)
    handles = [Line2D([0], [0], color=c, label=l, **kwa) for l,c in labels_colors]

    leg = dict(handles=handles, fontsize=8)
    return leg





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', type=str, help='pickled model files')
    parser.add_argument('--dest', default='ANL-plt-trainmodels', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    print('#=================================================================')
    print('           plt-trainmodels.py')
    print('#=================================================================')

    #classifier_names = ['LDA', 'QDA', 'OVO', 'OVR']
    classifier_names = ['OVO', 'OVR']
    colors = ['m', 'r', 'g', 'b']

    print('#=================================================================')

    #==========================================================================
    # IMPORT
    trained_model_data = []
    for i, pf in enumerate(args.f):
        print('unpickling: %s' % pf)
        with open(pf,'rb') as infile:
            dd = pickle.load(infile)
        trained_model_data.append(dd)


    # # collect the masks (should be done in anl-trainmodels?)
    # for dd in trained_model_data:
    #     df = dd['df_index'][['trial','Epoch#','isValid', 'isTraining']]
    #     df.set_index(['trial','Epoch#'], inplace=True)
    #     df.columns.name = 'mask'
    #     df = df.stack().unstack('Epoch#')
    #     df.set_index
    #     raise Exception('is this even used?')
    # masks = [dd['df_index'] for dd in trained_model_data]


    # one monster (multiindex) dataframe (of xv_predictions)to rule them all
    dfs = [dd['df_xv_predictions'] for dd in trained_model_data]
    bigdf = pd.concat(dfs, axis=0) #.reset_index(drop=True)
    df_index = bigdf.index.to_frame().reset_index(drop=True)


    #==========================================================================
    # RASTER PLOTS

    labels = ['Non REM', 'REM', 'Wake']
    panelKey = 'trial'
    labelKeys = ['classifier_name', 'hpscheme']
    drop = [('classifier_name', ['LDA','QDA'])]


    # legend
    df_legend = pd.DataFrame(
        data=zip(labels, np.linspace(0,1,len(labels))), 
        columns=['label', 'value']
        )
    leg = make_legend(df_legend=df_legend, cmap='rocket')

    # drop/keep

    df = bigdf.copy()
    for (col, vals) in drop:
        for val in vals:
            #df = df[df.index[col] != val]
            #df = df.query("%s == 't'")
            df = df[df.index.get_level_values(col) != val]
    df_raster = df
    df_index_raster = df.index.to_frame().reset_index(drop=True)

    # make data numerical (again)
    label2num = dict(df_legend[['label', 'value']].values)
    f_label2num = lambda x: label2num.get(x, np.nan)
    data = df_raster.copy().applymap(f_label2num).values
    dsrt = np.sort(data, axis=1)

    # make the raster plots
    kwa = dict(panelKey=panelKey, labelKeys=labelKeys, leg=leg)

    fig, ax = pt.montage_raster(df_index=df_index_raster, data=data, **kwa)
    plt.savefig(os.path.join(args.dest, 'plot-raster-sequentialTS.svg'))
    plt.savefig(os.path.join(args.dest, 'plot-raster-sequentialTS.png'))

    fig, ax = pt.montage_raster(df_index=df_index_raster, data=dsrt, **kwa)
    plt.savefig(os.path.join(args.dest, 'plot-raster-sortedTS.svg'))
    plt.savefig(os.path.join(args.dest, 'plot-raster-sortedTS.png'))


    #==============================================================================
    # accuracy and REM label accuracy
    fig = plt.figure(figsize=(6, 3), dpi=300)
    ax = [plt.subplot(121), plt.subplot(122)]

    dx = np.arange(len(trained_model_data))*0.1
    dx -= np.mean(dx)

    # panel 0: accuracy
    # panel 1: REM label accuracy
    toplot = [
        dict(panel=0, df='df_acc',  ylim=[0.85, 1], title='accuracy'),
        dict(panel=1, df='df_lacc', ylim=[0, 1],    title='REM label accuracy')
        ]

    for pp in toplot:        
        panel = pp['panel']
        df = pp['df']
        title = pp['title']
        ylim = pp['ylim']

        # left out trial values
        for j, scheme in enumerate(trained_model_data):
            df_acc = scheme[df][classifier_names]
            color = colors[j]
            dxj = dx[j]
            for i, col in enumerate(classifier_names):
                ax[panel].plot(dxj+[i]*(len(df_acc)-1), df_acc[col][:-1], 'o', mec='gray', color=color, ms=6, alpha=0.7)

        # averages lines
        for j, scheme in enumerate(trained_model_data):
            df_acc = scheme[df][classifier_names]
            ax[panel].plot(df_acc.loc['Avg'], '-', lw=3, color='w', zorder=2)
            ax[panel].plot(df_acc.loc['Avg'], '-', lw=2, color=colors[j], zorder=3)

        ax[panel].set_title(title)
        ax[panel].set_ylim(ylim)
        ax[panel].grid()

    txt = datetime.datetime.now().replace(microsecond=0).isoformat()
    fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(args.dest, 'plot-accuracy.png'))
    plt.close('all')



    # single trial plots
    for i, dd in enumerate(trained_model_data):
        xv_models = dd['xv_models']
        df_acc = dd['df_acc']
        df_lacc = dd['df_lacc']
        tagDict = dd['tagDict']
        params_train = dd['params_train']
        params_subsampling = dd['params_subsampling']

        # derived
        tag = tagDict['tag']
        fldr = os.path.join(args.dest, 'training-%s' % tag)
        os.makedirs(fldr, exist_ok=True)

        keep_classes = params_train.get('labels')


        print('  plotting: %s' % tag ) #, params_subsampling)


        #======================================================================
        # training blocks (TODO: should import this)
        png = os.path.join(fldr, 'plot-chop8640.png')
        kwa = dict(lw=1, color=colors[i])
        plot_chop8640(**params_subsampling, out=png, kwa=kwa)


        #======================================================================
        #   CONFUSION matrices (summed) for each classifier
        #   requires: xv_models, classifier_names, plot parameters
        fig = plt.figure(figsize=(12, 3), dpi=300)
        ax = [plt.subplot(141), plt.subplot(142), plt.subplot(143), plt.subplot(144)]

        nmax = 0
        cnfsums = {}
        for i,c in enumerate(classifier_names):
            cnfsums[c] = np.sum([m['classifiers'][c]['confusion'] for m in xv_models], axis=0)
            nmax = max(nmax, np.max(cnfsums[c]))

        for i,c in enumerate(classifier_names):
            cnfsum = cnfsums[c]
            pt.plot_confusion_matrix(
                ax=ax[i],
                cm=cnfsum, 
                classes=keep_classes,
                normalize=False,
                title=c,
                imkwa=dict(vmax=nmax),
                cbar=False,
                colorkwa=dict(fraction=0.04),
                cmap=plt.cm.Blues
                )

        txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(fldr, 'plot-confusion.png'))
        plt.close('all')


        #==============================================================================
        #   plot accuracy and REM label accuracy
        #   requires: df_acc, df_lacc, classifier_names
        #   
        fig = plt.figure(figsize=(6, 3), dpi=300)
        ax = [plt.subplot(121), plt.subplot(122)]

        #== panel 0: accuracy
        ax[0].plot(df_acc.loc['Avg'], 'o', mec='k',  color='b', ms=12)
        for i, col in enumerate(classifier_names):
            ax[0].plot([i]*(len(df_acc)-1), df_acc[col][:-1], 'o', mec='k', color='gray', ms=8)

        ax[0].set_title('accuracy')
        ax[0].set_ylim([0.85, 1.])
        ax[0].grid()

        #== panel 1: REM label accuracy
        ax[1].plot(df_lacc.loc['Avg'], 'o', mec='k', color='b', ms=12)    
        for i, col in enumerate(classifier_names):
            ax[1].plot([i]*(len(df_acc)-1), df_lacc[col][:-1], 'o', mec='k', color='gray', ms=8)
        ax[1].set_title('REM label accuracy')
        ax[1].set_ylim([0.0, 1.])
        ax[1].grid()

        txt = datetime.datetime.now().replace(microsecond=0).isoformat()
        fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(fldr, 'plot-accuracy.png'))
        plt.close('all')














    exit()














    
#     cls = 'OVO'
#     for m in models:
#         trial = m['trial']
#         fldr = os.path.join(args.dest, 'trial-%s' % (str(trial)))
#         os.makedirs(fldr, exist_ok=True)

#         #== training 2D plot with dividing borders
#         X = m['X_trn_std']
#         y = m['y_trn'].reset_index(drop=True)

#         Xv = m['X_val_std']
#         yv = m['y_val'].reset_index(drop=True)
#         yp = m[cls]['cls'].predict(Xv)

#         pca = PCA(n_components=2)
#         Xv_pca = pca.fit(X).transform(Xv)


#         #== OVO
#         df = pd.DataFrame(data=Xv_pca, columns=['PC1','PC2'])
#         df['predicted'] = yp
#         df['validation'] = yv['cScoreStr']
#         df['Epoch#'] = yv['Epoch#']

#         #wrong = df[df['validation'] != df['predicted']].copy()
#         #wrong['predicted'] = 'xx'
#         #dfplot = pd.concat([df, wrong])
#         #dfplot.reset_index(drop=True)
        
        
#         #== validation 2D plotly plot (colored by human score?)
#         cols = ['PC1','PC2']
#         figx = pt.scat2d(dfxyz=df, xycols=cols, tagcol='validation', title='plot', height=700, width=700)
#         html = os.path.join(fldr, 'plot-ovo-2D-validation.html')
#         pt.fig_2_html(figx, filename=html)


#         figx = pt.scat2d(dfxyz=df, xycols=cols, tagcol='predicted', title='plot', height=700, width=700)
#         html = os.path.join(fldr, 'plot-ovo-2D-predicted.html')
#         pt.fig_2_html(figx, filename=html)



#     raise Exception()
# #=========================================================================================
# #=========================================================================================
# #=========================================================================================
# #=========================================================================================
# #=========================================================================================

#     for m in models:
#         trial = m['trial']
#         fldr = os.path.join(args.dest, 'trial-%s' % (str(trial)))
#         os.makedirs(fldr, exist_ok=True)

#         #== training 2D plot with dividing borders
#         X = m['X_trn_std']
#         y = m['y_trn'].reset_index(drop=True)
#         lda = m['lda']
#         qda = m['qda']
#         ovr = m['ovr']
#         ovo = m['ovo']
#         pca = m['pca']



#         def get_xyzsurf(x, y, cls):
#             nx = 100
#             ny = 100
#             x_min = min(x)
#             x_max = max(x)
#             y_min = min(y)
#             y_max = max(y)
            
#             xdom = np.linspace(x_min, x_max, nx)
#             ydom = np.linspace(y_min, y_max, ny)
#             xx, yy = np.meshgrid(xdom, ydom)
#             Z = cls.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z[:, 1].reshape(xx.shape)
            
#             return [xdom, ydom, Z]

            
#         xdom = np.linspace(-2, 2, 40)
#         ydom = np.linspace(-2, 2, 40)
#         xx, yy = np.meshgrid(xdom, ydom)
#         #Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
#         Z = xx**2 + np.sin(yy*4)
#         #Z = Z[:, 1].reshape(xx.shape)
#         surfdata = [xdom, ydom, Z]


#         #== LDA
#         X_prj = lda.transform(X)
#         df = pd.DataFrame(data=X_prj, columns=['LD1','LD2'])
#         df = pd.concat([df, y], axis=1)
        
#         proba = lda.predict_proba(X)
#         isoprob = np.asarray([np.abs(np.diff(np.sort(row)[-2:])) for row in proba])
#         ridges = np.where(isoprob<0.05)[0]

#         dfridg = df.iloc[ridges].copy()
#         dfridg['cScoreStr'] = ['border']*len(ridges)
#         dfplot = pd.concat([df, dfridg], axis=0)

#         #== validation 2D plotly plot (colored by human score?)

#         surfdata = get_xyzsurf(df['LD1'], df['LD2'], lda)


#         cols = ['LD1','LD2']
#         figx = pt.scat2d(dfxyz=dfplot, xycols=cols, 
#                         surfdata=surfdata,
#                         tagcol='cScoreStr', title='plot', height=900, width=900)
#         html = os.path.join(fldr, 'plot-lda-2D-training.html')
#         pt.fig_2_html(figx, filename=html)

#         print('CLEAR')

#         #== QDA
#         X_prj = pca.transform(X)
#         df = pd.DataFrame(data=X_prj[:,:2], columns=['PC1','PC2'])
#         df = pd.concat([df, y], axis=1)
        
#         proba = qda.predict_proba(X_prj)
#         isoprob = np.asarray([np.abs(np.diff(np.sort(row)[-2:])) for row in proba])
#         ridges = np.where(isoprob<0.05)[0]

#         dfridg = df.iloc[ridges].copy()
#         dfridg['cScoreStr'] = ['border']*len(ridges)
#         dfplot = pd.concat([df, dfridg], axis=0)

#         #== validation 2D plotly plot (colored by human score?)
#         cols = ['PC1','PC2']
#         figx = pt.scat2d(dfxyz=dfplot, xycols=cols, tagcol='cScoreStr', title='plot', height=900, width=900)
#         html = os.path.join(fldr, 'plot-qda-2D-training.html')
#         pt.fig_2_html(figx, filename=html)



#         #== OVR
#         X_prj = pca.transform(X)
#         df = pd.DataFrame(data=X_prj[:,:2], columns=['PC1','PC2'])
#         df = pd.concat([df, y], axis=1)
        
        
        
#         proba = ovr.predict_proba(X)
#         isoprob = np.asarray([np.abs(np.diff(np.sort(row)[-2:])) for row in proba])
#         ridges = np.where(isoprob<0.05)[0]

#         dfridg = df.iloc[ridges].copy()
#         dfridg['cScoreStr'] = ['border']*len(ridges)
#         dfplot = pd.concat([df, dfridg], axis=0)

#         #== validation 2D plotly plot (colored by human score?)
#         cols = ['PC1','PC2']
#         figx = pt.scat2d(dfxyz=dfplot, xycols=cols, tagcol='cScoreStr', title='plot', height=900, width=900)
#         html = os.path.join(fldr, 'plot-ovr-2D-training.html')
#         pt.fig_2_html(figx, filename=html)




#     raise Exception()

        
#     #== for each trn/val
#         #== build trn/val
#         #== standardize trn
#         #== standardize val
#         #== drop XXX

#         #== LDA
#         #== scoring
        
#     #== model output, comparison
#     #== plotting?




#     #== PCA
#     pca_all = fusedFeat.pca()
#     pca_all.plotSummary(f=os.path.join(args.dest, 'plot-pca-summary.png'))
#     df_PCA = pca_all.project(data=fusedFeat.stack)


#     #== LDA
#     X = fusedFeat.stack.T
#     y = df_scores['cScoreStr'].values
#     lda = LinearDiscriminantAnalysis(n_components=3)
#     X_r2 = lda.fit(X, y).transform(X) 
#     df_LDA = pd.DataFrame(data=X_r2, columns=['LD1', 'LD2', 'LD3'])

#     print('SCORE:', lda.score(X, y))



#     #== MERGE
#     df_cat = pd.concat([df_scores, df_PCA, df_LDA], axis=1)
#     print(df_cat.head())

#     #== output
#     #   projections [trial, epoch, score, PC1-3, LD1-3]
#     #   transformation parameters
#     #   
#     #   


#     dfstack = df_cat


#     #== src (https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_quick_preview.php)

#     def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#         # setup marker generator and color map
#         markers = ('s', 'x', 'o', '^', 'v')
#         colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#         cmap = ListedColormap(colors[:len(np.unique(y))])

#         # plot the decision surface
#         x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#         np.arange(x2_min, x2_max, resolution))
#         #Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#         Z = classifier.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)    

#         Z = Z.reshape(xx1.shape)
#         plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#         plt.xlim(xx1.min(), xx1.max())
#         plt.ylim(xx2.min(), xx2.max())

#         # plot all samples
#     #    X_test, y_test = X[test_idx, :], y[test_idx]
#         for idx, cl in enumerate(np.unique(y)):
#             plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
#                     alpha=0.8, c=cmap(idx),
#                     marker=markers[idx], label=cl)
#         # highlight test samples
#         if test_idx:
#             X_test, y_test = X[test_idx, :], y[test_idx]
#             plt.scatter(X_test[:, 0], X_test[:, 1], c='',
#                     alpha=1.0, linewidth=1, marker='o',
#                     s=55, label='test set')

#     #X_combined_std = np.vstack((X_train_std, X_test_std))
#     #y_combined = np.hstack((y_train, y_test))

#     #plot_decision_regions(X=X, y=y, classifier=lda ) #test_idx=range(105,150))
#     #plt.legend(loc='upper left')
#     #plt.show()








#     #== PCA plots. 2D and 3D plotly plots
#     cols = ['PC1','PC2']
#     figx = pt.scat2d(dfxyz=dfstack, xycols=cols, tagcol='cScoreStr', title='plot', height=800, width=800)
#     html = os.path.join(args.dest, 'plot-pca-2d-all-scores.html')
#     pt.fig_2_html(figx, filename=html)

#     cols = ['PC1','PC2']
#     figx = pt.scat2d(dfxyz=dfstack, xycols=cols, tagcol='trial', title='plot', height=800, width=800)
#     html = os.path.join(args.dest, 'plot-pca-2d-all-trials.html')
#     pt.fig_2_html(figx, filename=html)

#     cols = ['PC1','PC2','PC3']
#     figx = pt.scat3d(dfxyz=dfstack, xyzcols=cols, tagcol='cScoreStr', title='plot', height=800, width=1400)
#     html = os.path.join(args.dest, 'plot-pca-3d-all.html')
#     pt.fig_2_html(figx, filename=html)
        


#     #== PCA plots. 2D and 3D plotly plots
#     cols = ['LD1','LD2']
#     figx = pt.scat2d(dfxyz=dfstack, xycols=cols, tagcol='cScoreStr', title='plot', height=800, width=800)
#     html = os.path.join(args.dest, 'plot-lda-2d-all-scores.html')
#     pt.fig_2_html(figx, filename=html)

#     cols = ['LD1','LD2']
#     figx = pt.scat2d(dfxyz=dfstack, xycols=cols, tagcol='trial', title='plot', height=800, width=800)
#     html = os.path.join(args.dest, 'plot-lda-2d-all-trials.html')
#     pt.fig_2_html(figx, filename=html)

#     cols = ['LD1','LD2','LD3']
#     figx = pt.scat3d(dfxyz=dfstack, xyzcols=cols, tagcol='cScoreStr', title='plot', height=800, width=1400)
#     html = os.path.join(args.dest, 'plot-lda-3d-all.html')
#     pt.fig_2_html(figx, filename=html)





