#!/usr/bin/env python3
import datetime
import pdb

import plotly
import plotly.graph_objs as go
from plotly import tools
import plotly.offline as po


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm

sns.set(color_codes=True)
sns.set_style('ticks')




def plot_confusion_matrix(ax=None,
                          cm=None,
                          classes=None,
                          normalize=False,
                          cbar=True,
                          title=None,
                          cmap=plt.cm.Blues,
                          imkwa={},
                          colorkwa={}):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Normalization is scaled by 100 to give percentages
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        #print("plotting normalized confusion matrix")
        cm = 100*cm.astype('float')/np.sum(cm.ravel())
    else:
        pass
        #print('plotting confusion matrix, without normalization')

    #print(cm)

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, **imkwa)
    if cbar:
        ax.figure.colorbar(im, ax=ax, **colorkwa)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax


def plot_features_template(df_feat_index=None, unique_scores=None, 
                           xpad=2, boxkwa=None,
                           fig=None, ax=None,
                           y0=0,
                           boxheight=0.05):
    """template for plotting state-specific feature vectors

    df_feat_index should have columns 'channel' and 'f[Hz]'
    """

    make_features_integers = True

    if boxkwa is None:
        boxkwa = dict(ec='none', fc='gray', alpha=0.2)


    channel_col = 'channel'
    xlbl = 'f[Hz]'
    df_ndx = df_feat_index

    # set up the x-axis indexing, ticks, and grey boxes for each channel
    # requires: channels, df_ndx, xpad

    channels = df_ndx[channel_col].unique()

    xtk, xtkl, channel_info = [], [], []
    for ic, channel in enumerate(channels):
        # data indices (ndx) and plotting (x-axis) indices
        ndx = np.argwhere(df_ndx[channel_col].values==channel).T[0]
        xndx = ndx+xpad*ic

        # build a rectangle to sit below y=0
        left, width = xndx[0], len(ndx)-1
        bottom, height = y0-boxheight, boxheight
        right = left + width
        top = bottom + height

        # boxcoords are set up for patches.Rectangle()
        boxcoords = [(left, bottom), width, height]

        # pack it up
        dd = dict(name=channel, xndx=xndx, ndx=ndx, boxcoords=boxcoords)
        channel_info.append(dd)

        # ticks and ticklabels
        mid = len(ndx)//2
        ticks = [xndx[0], xndx[mid], xndx[-1]]
        ticklabels = df_ndx['f[Hz]'].values[[ndx[0], ndx[mid], ndx[-1]]]
        if make_features_integers:
            ticklabels = ticklabels.astype(int)
        xtk += ticks
        xtkl += ticklabels.tolist()


    if fig is None:
        fig = plt.figure(figsize=(8,8))

    if ax is None:
        nrow = len(unique_scores)
        ncol = 1
        ax = [plt.subplot(nrow, ncol, i+1) for i in range(nrow*ncol)]


    # the main loop over sleep states (panel rows)
    for i, ss in enumerate(unique_scores):
        # one channel at a time (panel columns)
        for dd in channel_info:

            # grey boxes
            ax[i].add_patch(patches.Rectangle(*dd['boxcoords'], **boxkwa))

            # channel name label
            xctr = np.mean(dd['xndx'])
            bhd2 = boxheight/2
            ax[i].text(xctr, y0-bhd2, dd['name'], va='center', ha='center', fontsize=8)

            # pseudo x-axis (only for bottom-most row)
            if i+1==len(ax):
                ax[i].plot(dd['xndx'], dd['ndx']*0-boxheight+y0, lw=2, color='k', zorder=3)


        ax[i].set_ylabel(ss)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)            
        ax[i].spines['bottom'].set_visible(False)

        # ticks
        if i+1<len(ax):
            ax[i].set_xticklabels([])
            ax[i].set_xticks([])
        else:
            ax[i].set_xticks(xtk)
            ax[i].set_xticklabels(xtkl, rotation='vertical')
            ax[i].set_xlabel(xlbl)

    return fig, ax, channel_info


def plot_2D_hist(ax=None, h2d=None, levels=None,
                 cmap='rocket', ptype='contourf', cbar=True):
    """plot a histogram of data projected onto a 2D basis

    NOTE: this is only the histogram (raw data not plotted here)
    TODO: method of Histo2D?

    arguments:
    ------
    h2d: Histo2D object
    ax: axes on which to plot
    levels: contour levels (also sets the colorbar limits)
    cmap: colormap for contour/imshow
    ptype: (contourf or imshow) which type of plot to plot
    """

    if h2d is None:
        raise Exception('h2d required')
    
    xlbl = h2d.dims[0]
    ylbl = h2d.dims[1]
    xdom = h2d.bin_edges[0]
    ydom = h2d.bin_edges[1]
    xlim = [xdom[0], xdom[-1]]
    ylim = [ydom[0], ydom[-1]]
    extent = (xdom[0], xdom[-1], ydom[0], ydom[-1])


    zplt = h2d.hist
    [zplt_min, zplt_max] = h2d.range

    if isinstance(levels, (list, np.ndarray)):
        pass
    elif levels is None or levels == 'auto':
        zplt_min = np.floor(zplt_min)
        zplt_max = np.ceil(zplt_max)
        levels = np.linspace(zplt_min, zplt_max, 8)

    # make a discretized colormap
    cmapx = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, ncolors=cmapx.N, clip=True)

    # actual plotting
    pltkwa = dict(origin='lower', cmap=cmap, extent=extent)
    if ptype == 'contourf':
        im0 = ax.contourf(zplt.T, levels=levels, **pltkwa)
    elif ptype == 'imshow':
        im0 = ax.imshow(zplt.T, aspect='auto', norm=norm, **pltkwa)

    # colorbar?
    if cbar == True:
        cb = plt.gcf().colorbar(im0, ax=ax, pad=0.01)

        if h2d.isNormalized:
            cb_ylbl = 'p'
        else:
            cb_ylbl = 'N'
        if h2d.isLogScaled:
            cb.ax.set_ylabel('log(%s)' % (cb_ylbl))
        else:
            cb.ax.set_ylabel(cb_ylbl)

    else:
        cb = None

    # gussy it up
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)

    return ax, cb


def plot_PCA_2D_hist(X=None, pca=None, ax=None, 
                     h2d=None,
                     PCs=[1,2],
                     levels=None,
                     numsig=3, numbin=60,
                     cmap='rocket', 
                     log=True,
                     normalize=False,
                     tiny=0.6,
                     ptype='contourf',
                     cbar=True
                     ):
    """plot a histogram of data projected onto a 2D PC basis

    NOTE: this is only the histogram (raw data not plotted here)
    TODO: disentangle 2D histo from PCA specifics. WWRW is general 2D plotter

    arguments:
    ------
    X: the data
    pca: remtools.PCA instance
    h2d: histo2D object (otherwise it is computed here)
    ax: axes on which to plot
    PCs: PC indices (1 indexed)
    justlimits: compute histogram but only return the limits (for multiple plots)
    levels: contour levels (also sets the colorbar limits)

    xdom/ydom: x/y bin edges (disabled)
    numsig: (autogrid) make x/ydom span +/- numsig sigmas (PCA eigenvalues)
    numbin: (autogrid) how many bins within x/ydom
    cmap: colormap for contour/imshow
    log: log scale or not
    normalize: normalize?
    tiny: tiny number added to bin counts (to avoid NaNs)
    ptype: (contourf or imshow) which type of plot to plot    
    """

    raise Exception('deprecated')

    # all in one go
    if h2d is None:
        h2d = pca.project_histo(data=X, PCs=PCs, numsig=numsig, numbin=numbin)
    

    ax, cb = plot_2D_hist(
        ax=ax, 
        h2d=h2d,
        levels=levels,
        cmap=cmap, 
        log=log,
        normalize=normalize,
        tiny=tiny,
        ptype=ptype,
        cbar=cbar,     
    )

    # gussy it up (PCA SPECIFIC)
    sigX = np.sqrt(pca.vals[PCs[0]-1])
    sigY = np.sqrt(pca.vals[PCs[1]-1])

    plot_pca_crosshair(ax=ax, sigX=sigX, sigY=sigY)

    return ax, cb


def plot_pca_crosshair(ax=None, sigX=1, sigY=1):
    """crosshairs, with ticks"""
    gridkwa = dict(color='white', alpha=0.7)

    ticksig = [1, 2]
    for tt in ticksig:
        xx, yy = sigX*tt, sigY*tt
        ax.plot([-xx, -xx], [-yy*0.1, yy*0.1], '-', **gridkwa)
        ax.plot([ xx,  xx], [-yy*0.1, yy*0.1], '-', **gridkwa)
        ax.plot([-xx*0.1, xx*0.1], [-yy, -yy], '-', **gridkwa)
        ax.plot([-xx*0.1, xx*0.1], [ yy,  yy], '-', **gridkwa)
        if tt == max(*ticksig):
            ax.plot([-xx, xx], [0, 0], '-', **gridkwa)
            ax.plot([0, 0], [-yy, yy], '-', **gridkwa)


def montage_raster(df_index=None, data=None, cmap='rocket', aspect=200,
                   panelKey='T', labelKeys=['C','M'], leg=None, wtf=False):
    """
    montage of raster plots

    TODO: color / label legend
    TODO: df_index and data should be panel sorted ahead of time
    """

    #== split data by T
    #aspect = 200
    xlabel = 'epoch#'

    panels = df_index[panelKey].unique()
    num_panels = len(panels)

    h2w = data.shape[0]/data.shape[1]
    figx = plt.figure(figsize=(8,10) )
    ax = [plt.subplot(num_panels, 1, i+1) for i in range(num_panels)]

    if wtf:
        # dirty hack to sort panels
        sums = np.sum(data, axis=1)
        dfsums = df_index.copy()
        dfsums['sum'] = sums

        panelsums = [np.sum(dfsums[dfsums[panelKey]==t]['sum']) for t in panels]
        ndxsrt = np.argsort(panelsums)
        panels_sorted = [panels[n] for n in ndxsrt]

        # print(list(zip(panels, panelsums)))
        # print(panels)
        # print(panels_sorted)

        panels = panels_sorted



    for ip, panel in enumerate(panels):

        # subset of data corresponding to this panel
        dfip = df_index[df_index[panelKey] == panel]

        # sort rows
        dfip = dfip.sort_values(labelKeys, axis=0)

        rld = dfip[[panelKey]+labelKeys].values.astype(str)
        rowlabels = [' '.join(x) for x in rld]

        ndx = dfip.index.values
        num_rows = len(dfip)


        # plot
        ax[ip].imshow(data[ndx], cmap=cmap, aspect=aspect, interpolation='none')


        # fancy ticks, gridlines and labels

        # y gridlines denote groups of the first labelKey
        groupsize = num_rows / len(df_index[labelKeys[0]].unique())

        mjtk = np.arange(-0.5, num_rows, groupsize)
        mntk = np.arange(0, num_rows, 1)

        ax[ip].set_yticks(mjtk, minor=False)
        ax[ip].set_yticks(mntk, minor=True)
        ax[ip].set_yticklabels('', minor=False)
        ax[ip].set_yticklabels(rowlabels, minor=True, family='monospace')
        
        ax[ip].tick_params(axis='x', which='major', labelsize=7, length=4)
        ax[ip].tick_params(axis='y', which='major', labelsize=7, length=0)
        ax[ip].tick_params(axis='y', which='minor', labelsize=7, length=5, pad=6)
        ax[ip].tick_params(axis='y', which='minor', right=True, length=0) 

        ax[ip].grid(True)   #== NOTE: zorder is not honored here

        if leg is not None:
            ax[ip].legend(**leg)

        if ip+1<num_panels:
            ax[ip].set_xticklabels([])
        else:
            ax[ip].set_xlabel(xlabel)


    fig = plt.gcf()
    ax = plt.gca

    txt = datetime.datetime.now().replace(microsecond=0).isoformat()
    fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=12)

    return fig, ax


def scat3d(dfxyz=None, xyzcols=None, tagcol=None, title='plot', height=800, width=800):
    """
    plotly 3D scatter plot
    
    ARGUMENTS:
        dfxyz   dataframe with x,y,z coordinates
        colxyz  column names for x,y,z in the dataframe
      
    TODO: add group tags(for the legend) and neuronID tags(hoverinfo) to the dataframe
    """

    tag = 'tag'
    tfont = dict(family='Courier New, monospace', size=18, color='#3f3f3f')

    lblX = xyzcols[0]
    lblY = xyzcols[1] 
    lblZ = xyzcols[2]

    autosize=False
    opacity=1 #0.99    # opacity=1 required for true 3d rendering! (but opacity=1 breaks hoverinfo)
    lw=2

    #ad_marker=dict(size=4,color='grey',opacity=0.3)
    margin=dict(l=10, r=100, b=10, t=50)

    pltdata = []
    if tagcol is not None:
        for tag in dfxyz[tagcol].unique():
            dfi = dfxyz[dfxyz[tagcol] == tag]
            pltfmt=dict(line=dict(width=lw), opacity=opacity, mode='markers', name=tag)
            pltfmt['legendgroup'] = tag
            pltfmt['hoverinfo'] = 'none'
            pltfmt['marker'] = dict(size=5, line=dict(color='rgb(110, 110, 110)', width=1))
            #pltfmt['showlegend'] = True #if i==0 else False            
            pltdata.append(go.Scatter3d(x=dfi[lblX], 
                                        y=dfi[lblY], 
                                        z=dfi[lblZ], **pltfmt))
    else:
        tag = 'tag'
        pltfmt=dict(line=dict(width=lw), opacity=opacity, mode='markers', name=tag)
        pltfmt['legendgroup'] = tag
        pltfmt['hoverinfo'] = 'none'
        #pltfmt['showlegend'] = True #if i==0 else False
        pltdata = [go.Scatter3d(x=dfxyz[lblX], y=dfxyz[lblY], z=dfxyz[lblZ], **pltfmt)]

    #== camera and layout
    #camera_xy = dict(
        #up=dict(x=0, y=0, z=1),
        #center=dict(x=0, y=0, z=0),
        #eye=dict(x=0.0, y=-0.0, z=2.5)
    #)

    xaxis = dict(title=lblX, titlefont=tfont)
    yaxis = dict(title=lblY, titlefont=tfont)
    zaxis = dict(title=lblZ, titlefont=tfont)

    layout = go.Layout(
        title=title,
        scene = dict(xaxis=xaxis,
                     yaxis=yaxis,
                     zaxis=zaxis,
                     aspectmode='data',
                     #camera=camera_xy,
                     ),
        showlegend=True,
        autosize=autosize,
        width=width,
        height=height,
        margin=margin,
        #updatemenus=list([ dict( x=0.55, y=1, yanchor='top',  buttons=butlst, ) ]),
    )
    fig3d = go.Figure(data=pltdata, layout=layout)

    return fig3d



def scat2d(dfxyz=None, xycols=None, tagcol=None, surfdata=None,
           markers={},
           title='plot', height=800, width=800):
    """
    plotly 2D scatter plot
    
    ARGUMENTS:
        dfxyz   dataframe with x,y,z coordinates
        colxyz  column names for x,y,z in the dataframe
      
    TODO: add group tags(for the legend) and neuronID tags(hoverinfo) to the dataframe
    NOTE: opacity=1 required for true 3d rendering! (but opacity=1 breaks hoverinfo)
    """
    
    gridwidth = 2
    zerolinewidth = 2

    markerFormat = dict(
        size=10,
        opacity=1,
        line=dict(color='rgb(110, 110, 110)', width=1)
        )

    markerFormat.update(markers)

    hovercol = 'Epoch#'
    tfont = dict(family='Courier New, monospace', size=18, color='#3f3f3f')

    lblX = xycols[0]
    lblY = xycols[1]

    autosize=False
    opacity=1 #0.99    
    lw=2
    #ad_marker=dict(size=4,color='grey',opacity=0.3)
    margin=dict(l=100, r=100, b=100, t=100)

    pltdata = []
    #== surface plot?
    if surfdata is not None:
        trace = go.Heatmap(x=surfdata[0],
                           y=surfdata[1],
                           z=surfdata[2])
        pltdata.append(trace)


    if tagcol is not None:
        for tag in dfxyz[tagcol].unique():
            dfi = dfxyz[dfxyz[tagcol] == tag]
            pltfmt=dict(line=dict(width=lw), opacity=opacity, mode='markers', name=tag)
            pltfmt['legendgroup'] = tag
            #pltfmt['hoverinfo'] = 
            if hovercol is not None:
                pltfmt['text'] = dfi[hovercol]
            pltfmt['marker'] = dict(**markerFormat)
            #pltfmt['showlegend'] = True #if i==0 else False            
            pltdata.append(go.Scatter(x=dfi[lblX], 
                                      y=dfi[lblY], 
                                      **pltfmt))
    else:
        tag = 'tag'
        pltfmt=dict(line=dict(width=lw), opacity=opacity, mode='markers', name=tag)
        pltfmt['legendgroup'] = tag
        #pltfmt['hoverinfo'] = 'none'
        if hovercol is not None:
            pltfmt['text'] = dfxyz[hovercol]
        #pltfmt['showlegend'] = True #if i==0 else False
        pltdata.append(go.Scatter(x=dfxyz[lblX], y=dfxyz[lblY], **pltfmt))

    xaxis = dict(title=lblX, titlefont=tfont)
    yaxis = dict(title=lblY, titlefont=tfont)

    layout = go.Layout(
        title=title,
        scene = dict(
            xaxis=xaxis,
            yaxis=yaxis,
            #zaxis=zaxis,
            aspectmode='data',
            #camera=camera_xy,
            ),
        showlegend=True,
        legend=dict(
            x=0,
            y=1),
        autosize=autosize,
        width=width,
        height=height,
        margin=margin,
        hovermode= 'closest'
        #updatemenus=list([ dict( x=0.55, y=1, yanchor='top',  buttons=butlst, ) ]),
    )
    fig2d = go.Figure(data=pltdata, layout=layout)

    pimp = dict(titlefont=tfont, tickfont=tfont, gridwidth=gridwidth, zerolinewidth=zerolinewidth)
    fig2d['layout']['xaxis'].update(title=lblX, **pimp)  
    fig2d['layout']['yaxis'].update(title=lblY, **pimp)

    return fig2d




def fig_2_html(fig, auto_open=False, filename='plot-plotly.html'):
    #== a wrapper for the plotly offline function
    return po.plot(fig, auto_open=auto_open, filename=filename)
