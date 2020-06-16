


import pdb
import re
import os
import json
import datetime
import time

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

import sleep_scorer.plottools as pt
import sleep_scorer.remtools as rt




class EEGPlotter(object):
    """dashboard-like plotting for EEG data
    
    The panel view incorporates:
    - raw signal timeseries
    - features (power spectra)
    - pca (or lowD) feature projections
    - scores (human/model scores or other categoricals like 
        consensus/transition/conflict)
    
    input:
    ------
    std: (StagedTrialData object) (raw edr *and* features)
    pca: (PCA object) pca avg/vecs/vals
    scores : (ScoreBlock)


    methods:
    ------
    plot(fig, ti/tf): high level, just specify target fig and start/end times

    TODO:
    GENERAL
        - general time series plotting (e.g. EMG power)
        - general 2d/histogram plots (e.g. EMG power)
    SPEED HAX
        - fuse time series
        - keep axes, clear data
        - keep histograms
        - scrolling w/pseudocache: 
            - plot some range beyond axes limits, 
            - adjust axes limits for small steps
            - replot big chunks less often/ asneeded
    PDF OVERLAY
        - comet
        - conflicts
        - switch epochs
    """

    def __init__(self, std=None, pca=None, scores=None, params={}):

        
        self.std = std
        self.pca = pca

        self.scores = scores


        self.df_prj = self.pca.project(std.features.data)

        self.params = self.default_params()
        self.params.update(params)

        # FEATURES
        self.X = self.std.features.data

        # RAW features (normalize, stride)
        # # raw features, look like shit w/current formatting
        # self.df_feat = rt.SxxBundle.from_EDFData(self.std.edf).to_dataframe()

        # stash (time consuming) computed values here for re-use
        self.stash = {}

        # initial viewstate
        self.viewEpoch = 100
        self.viewWidth = 20


        # render the first frame
        self.make_fig()
        self.render()


    @property
    def viewrange(self):
        ea = self.viewEpoch - self.viewWidth
        eb = self.viewEpoch + self.viewWidth
        return [ea, eb]

    def default_params(self):
        """make default params"""
        params = dict(
            name='gallahad',
            quest='grail'
        )
        return params


    def about(self):
        """helpful information at a glance"""
        print('------ EEGPlotter.about() ------')
        print('params:', self.params)
        self.scoreblock.about()
        self.pca.about()
        self.std.about()


    def make_fig(self):
        """create the figure w/event handling"""


        aa = '707070'
        #aa = 'a0a0a0'
        b0 = 'b0b0b0'
        gg = 'c0c0c0'

        pp = {
            'lines.linewidth':2,
            'axes.facecolor':'k',
            'axes.edgecolor': gg,
            'axes.labelcolor': gg,
            'figure.facecolor':'k',
            'figure.edgecolor':'k',
            'savefig.facecolor':'k',
            'savefig.edgecolor':'k',
            'xtick.color' : gg,
            'ytick.color' : gg,
            'grid.color'  : aa,
            'text.color' : gg
        }

        matplotlib.rcParams.update(pp)

        self.fig = plt.figure(figsize=(16, 8), dpi=80)

        self.fig.canvas.mpl_connect('key_press_event', self.kupdate)
        self.fig.canvas.mpl_connect('button_press_event', self.mupdate)
        self.fig.canvas.mpl_connect('scroll_event', self.mupdate)

        # self.fig.set_facecolor('k')

    def kupdate(self, event):
        """keypress updates"""
        # step sizes
        s = [1, 5, 10]

        # print(event.key)

        if event.key == 'left':
            self.lstep(inc=s[0])
        # if event.key == 'shift+left':
        #     self.lstep(inc=s[1])
        if event.key == 'ctrl+left':
            self.lstep(inc=s[2])

        if event.key == 'right':
            self.rstep(inc=s[0])
        # if event.key == 'shift+right':
        #     self.lstep(inc=s[1])
        if event.key == 'ctrl+right':
            self.rstep(inc=s[2])

        if event.key == 'up':
            self.viewWidth = max(self.viewWidth-1, 3)
            self.render()
        if event.key == 'ctrl+up':
            self.viewWidth = max(self.viewWidth-2, 3)
            self.render()

        if event.key == 'down':
            self.viewWidth += 1
            self.render()
        if event.key == 'ctrl+down':
            self.viewWidth += 2
            self.render()

    def mupdate(self, event):
        """update when mouse buttons pushed or wheel spun"""
        # step sizes
        s = [1, 5, 10]


        # STEP LEFT (backward in time)
        if event.button == 1:
            if event.key is None:
                self.lstep(inc=s[0])
            elif event.key == 'shift':
                self.lstep(inc=s[1])
            elif event.key == 'control':
                self.lstep(inc=s[2])

        # STEP RIGHT (forward in time)
        if event.button == 3:
            if event.key is None:
                self.rstep(inc=s[0])
            elif event.key == 'shift':
                self.rstep(inc=s[1])
            elif event.key == 'control':
                self.rstep(inc=s[2])

        # zoom out
        if event.button == 'down':
            self.viewWidth += 1
            self.render()

        # zoom in
        if event.button == 'up':
            self.viewWidth = max(self.viewWidth-1, 3)
            self.render()


    def rstep(self, inc=1):
        """step right, next epoch"""
        self.viewEpoch += inc
        self.render()

    def lstep(self, inc=1):
        """step left, prev epoch"""
        self.viewEpoch -= inc
        self.render()


    def render(self, viewEpoch=None, viewWidth=None):
        """render the figure
        
        render?! I hardly know 'er!
        """


        t00 = time.time()

        sig_labels_plot = ['EEG1', 'EEG2', 'EMG']

        if viewEpoch is not None:
            self.viewEpoch = viewEpoch

        if viewWidth is not None:
            self.viewWidth = viewWidth

        [ia, ib] = self.viewrange
        ie = self.viewEpoch

        chunksize = ib-ia

        edf = self.std.edf
        dfmerge = self.df_prj
        num_epochs = edf.num_epochs
        epoch_duration = edf.epoch_duration
        spectrograms = edf.spectrograms
        signal_traces = edf.signal_traces

        t05 = time.time()

        figx = self.fig

        t10 = time.time()

        #== plot AXES

        # if self.fig.axes == []:
        #     self.ax = [
        #         plt.subplot2grid((4,7),(0,0), rowspan=1, colspan=4), 
        #         plt.subplot2grid((4,7),(1,0), rowspan=1, colspan=4), 
        #         plt.subplot2grid((4,7),(2,0), rowspan=2, colspan=2), 
        #         plt.subplot2grid((4,7),(2,2), rowspan=2, colspan=2), 
        #         plt.subplot2grid((4,7),(0,4), rowspan=4, colspan=2)
        #         ]
        # else:
        #     #pdb.set_trace()
        #     for axi in self.fig.axes:
        #         axi.clear()


        self.ax = [
            plt.subplot2grid((4,7),(0,0), rowspan=1, colspan=4), 
            plt.subplot2grid((4,7),(1,0), rowspan=1, colspan=4), 
            plt.subplot2grid((4,7),(2,0), rowspan=2, colspan=2), 
            plt.subplot2grid((4,7),(2,2), rowspan=2, colspan=2), 
            plt.subplot2grid((4,7),(0,4), rowspan=4, colspan=2)
            ]


        axx = self.ax[0:4]
        axb = [self.ax[-1]]

        t15 = time.time()

        # print('  --')
        # print('  t assign: %4.2f' % (t05-t00))
        # print('  t fig   : %4.2f' % (t10-t05))
        # print('  t ax    : %4.2f' % (t15-t10))


        #======================================================================
        #======================================================================
        #======== LHS (signals/pca)
        #======================================================================
        #======================================================================

        t20 = time.time()
        #==================================================
        #== panel 0, RAW signal time series
        raw_stride = 5
        dy_raw = -300

        tr000 = time.time()


        xxx, yyy, lbl = [], [], []
        for i, label in enumerate(sig_labels_plot):
        
            st = signal_traces[label]
            ndxi = int(st.samples_per_epoch*ia)
            ndxf = int(st.samples_per_epoch*ib)
            ti = ndxi/st.f
            tf = ndxf/st.f
            xx = np.linspace(ti, tf, int(st.samples_per_epoch)*chunksize)

            yy = st.sig[ndxi:ndxf]+dy_raw*i
            xxx.append(xx[::raw_stride])
            yyy.append(yy[::raw_stride])
            lbl.append(label)
        
        tr001 = time.time()
        
        tr002 = time.time()
        xxx = np.asarray(xxx).T
        yyy = np.asarray(yyy).T
        lobj = axx[0].plot(xxx, yyy, lw=1)


        # BOX BOX
        ndxe = int(st.samples_per_epoch*ie)
        te = ndxe/st.f
        xbox = [te, te, te-10, te-10, te]
        ybox = [-900, 300, 300, -900, -900]
        axx[0].plot(xbox, ybox, 'c-', ms=0, lw=2)


        tr003 = time.time()


        axx[0].set_ylim([-400+2*dy_raw, 400])
        axx[0].set_xlim([xx[0], xx[-1]])
        #axx[0].set_xticks(np.linspace(ti, tf, chunksize+1))
        #axx[0].set_xticklabels([])
        axx[0].grid(True)
        axx[0].set_ylabel('raw signals')
        axx[0].set_xlabel('t [s]')

        axx[0].spines['top'].set_visible(False)
        axx[0].spines['right'].set_visible(False)
        axx[0].spines['bottom'].set_visible(False)
        leg = axx[0].legend(lobj, lbl, loc='upper right') #, ncol=len(lbl))
        leg.get_frame().set_edgecolor('none')


        tr004 = time.time()
        # print('raw 1 : %3.0f' % ((tr001-tr000)*1000))
        # print('raw 2 : %3.0f' % ((tr002-tr001)*1000))
        # print('raw 3 : %3.0f' % ((tr003-tr002)*1000))
        # print('raw 4 : %3.0f' % ((tr004-tr003)*1000))


        # PCA histos and projections
        t40 = time.time()

        #==================================================
        #== panel 1, PC time series
        pcvec_cols = ['PC1', 'PC2', 'PC3']
        for i, col in enumerate(pcvec_cols):
            dy = -1
            xx = np.arange(ia+1, ib+1)
            yy = dfmerge[col][ia:ib] + dy*i
            axx[1].plot(xx, yy, '-o', ms=4, label=col)

        # BOX BOX
        xbox = [ie-0.5, ie-0.5, ie+0.5, ie+0.5, ie-0.5]
        ybox = [-2.8, 0.8, 0.8, -2.8, -2.8]
        axx[1].plot(xbox, ybox, 'c-', ms=0, lw=2)


        axx[1].set_xlim([ia+0.5, ib+0.5])
        axx[1].set_ylim(-2.9, 0.9)
        axx[1].set_xlabel('Epoch')
        axx[1].set_ylabel('PC projections')
        axx[1].grid(True)
        axx[1].spines['top'].set_visible(False)
        axx[1].spines['right'].set_visible(False)    
        axx[1].spines['bottom'].set_visible(False)
        
        leg = axx[1].legend(loc='upper right') #, ncol=len(pcvec_cols))
        leg.get_frame().set_edgecolor('none')




        #==================================================
        #== panel 2 and 3: PC 2D projections
        line_kwa = dict(
            color='magenta',
            marker='o',
            lw=2,
            ms=4,
        )

        # line_kwa_inner = dict(
        #     color='blue',
        #     marker='o',
        #     lw=0,
        #     ms=3,
        # )

        pca_hist_kwa = dict(
            numsig=3,
            numbin=60,
        )

        plt_hist_kwa = dict(
            cmap='Greys_r',
            levels='auto',
            ptype='imshow',
        )

        tail_length = 7


        t50 = time.time()

        # pre-compute histograms and stash them
        if 'h2d_32' not in self.stash.keys():
            h2d = self.pca.project_histo(self.X, PCs=[3,2], **pca_hist_kwa).normalize().logscale()
            self.stash['h2d_32'] = h2d

        if 'h2d_12' not in self.stash.keys():
            h2d = self.pca.project_histo(self.X, PCs=[1,2], **pca_hist_kwa).normalize().logscale()
            self.stash['h2d_12'] = h2d

        h2d_32 = self.stash['h2d_32']
        h2d_12 = self.stash['h2d_12']

        t55 = time.time()

        #pt.plot_PCA_2D_hist(X=self.X, h2d=h2d_32, pca=self.pca, PCs=[3,2], ax=axx[2], **pca_hist_kwa, cbar=False)
        pt.plot_2D_hist(h2d=h2d_32, ax=axx[2], **plt_hist_kwa, cbar=False)
        pt.plot_pca_crosshair(ax=axx[2], sigX=h2d_32.varX, sigY=h2d_32.varY)

        # axx[2].plot(dfmerge['PC3'][ia:ib], dfmerge['PC2'][ia:ib], **line_kwa)
        # axx[2].plot(dfmerge['PC3'][ia:ib], dfmerge['PC2'][ia:ib], **line_kwa_inner)
        axx[2].plot(dfmerge['PC3'][ie-tail_length:ie], dfmerge['PC2'][ie-tail_length:ie], **line_kwa)
        axx[2].plot(dfmerge['PC3'][ie-1], dfmerge['PC2'][ie-1], 'co', mfc='w', mew=4, ms=12)


        axx[2].set_xlabel('PC3')
        axx[2].set_ylabel('PC2')
        axx[2].spines['top'].set_visible(False)
        axx[2].spines['right'].set_visible(False)
        axx[2].spines['bottom'].set_visible(False)
        axx[2].spines['left'].set_visible(False)

        #pt.plot_PCA_2D_hist(X=self.X, h2d=h2d_12, pca=self.pca, PCs=[1,2], ax=axx[3], **pca_hist_kwa, cbar=True)
        pt.plot_2D_hist(h2d=h2d_12, ax=axx[3], **plt_hist_kwa, cbar=False)
        pt.plot_pca_crosshair(ax=axx[3], sigX=h2d_12.varX, sigY=h2d_12.varY)
        # axx[3].plot(dfmerge['PC1'][ia:ib], dfmerge['PC2'][ia:ib], **line_kwa)
        # axx[3].plot(dfmerge['PC1'][ia:ib], dfmerge['PC2'][ia:ib], **line_kwa_inner)
        axx[3].plot(dfmerge['PC1'][ie-tail_length:ie], dfmerge['PC2'][ie-tail_length:ie], **line_kwa)
        axx[3].plot(dfmerge['PC1'][ie-1], dfmerge['PC2'][ie-1], 'co', mfc='w', mew=4, ms=12)

        axx[3].set_xlabel('PC1')
        axx[3].set_ylabel('')
        axx[3].set_yticklabels([])
        axx[3].spines['top'].set_visible(False)
        axx[3].spines['right'].set_visible(False)
        axx[3].spines['bottom'].set_visible(False)
        axx[3].spines['left'].set_visible(False)

        # overlay scores on 2D histograms
        # BROKEN (should use scoreblock not scorewizard)
        try:
            #print('ia,ib:', ia,ib)
            dfj = dfmerge[ia+1:ib]

            wk = dfj[dfj['cScoreNum'] == sw.scoreStr2Num['Wake']]
            rm = dfj[dfj['cScoreNum'] == sw.scoreStr2Num['REM']]
            nr = dfj[dfj['cScoreNum'] == sw.scoreStr2Num['Non REM']]
            xx = dfj[dfj['cScoreNum'] == 0]

            kwa = dict(lw=0, marker='o', mec='k', ms=7)

            axx[2].plot(wk['PC2'], wk['PC3'], **kwa, color='blue',  label='Wake')
            axx[2].plot(rm['PC2'], rm['PC3'], **kwa, color='red',   label='REM')
            axx[2].plot(nr['PC2'], nr['PC3'], **kwa, color='green', label='Non REM')
            axx[3].plot(wk['PC1'], wk['PC2'], **kwa, color='blue',  label='Wake')
            axx[3].plot(rm['PC1'], rm['PC2'], **kwa, color='red',   label='REM')
            axx[3].plot(nr['PC1'], nr['PC2'], **kwa, color='green', label='Non REM')
        except:
            pass



        #======================================================================
        #======================================================================
        #============================= RHS (features)
        #======================================================================
        #======================================================================


        t60 = time.time()

        df_feat_index = self.std.features.df_index

        unique_scores = ['all']
        gt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax, channel_info = pt.plot_features_template(
            y0=ia,
            df_feat_index=df_feat_index,
            unique_scores=unique_scores,
            xpad=3,
            boxheight=1,
            fig=figx,
            ax=axb
            )

        t65 = time.time()

        # FEATURES
        for i, chi in enumerate(channel_info):
            # NOTE plot calls are rate limiting (10-15ms each)
            # TODO: flip order of channels/epochs, concatenate channels w/padding
            taa = time.time()

            feature_scale = 4
            xxx, yyy, ccc = [], [], []
            xmax = 0
            for it in range(ia, ib):
                cc = gt_colors[it % len(gt_colors)]
                xx = chi['xndx']
                yy = self.std.features.data[chi['ndx'], it]*feature_scale+it+1
                ccc.append(cc)
                xxx.append(xx)
                yyy.append(yy)
                xmax = max(xmax, max(xx))

        
            #axb[0].set_prop_cycle('color', ccc)
            tbb = time.time()
            axb[0].plot(np.asarray(xxx).T, np.asarray(yyy).T, color='gray')
            
            # re-plot the current viewEpoch features in cyan
            yve = self.X[chi['ndx'], ie-1]*feature_scale+ie
            axb[0].plot(xx , yve, 'c-', lw=3)

            tcc = time.time()

            # print('ftplt 1   : %3.0f' % ((tbb-taa)*1000))
            # print('ftplt 2   : %3.0f' % ((tcc-tbb)*1000))


        t70 = time.time()
        # SCORES
        # TODO: make this a single call to scatter (won't work, scatter cannot use >1 marker)
        # TODO: must call plot or scatter 1x per label -- speed test comparison

        scoreTypes = self.scores.df['scoreType'].unique()
        scoreTags = self.scores.df['scoreTag'].unique()

        for j, scoreTag in enumerate(scoreTags):

            data = self.scores.keeprows(conditions=[('scoreTag', scoreTag)])
            # print(scoreTag)
            # print(data.data)

            dx = xmax+5+6*j
            dfj = data.data.ravel()[ia:ib]
            yvals = np.arange(ia, ib)+1
            # print(yvals)
            # print(dfj)

            wk = yvals[dfj == 'Wake']
            rm = yvals[dfj == 'REM']
            nr = yvals[dfj == 'Non REM']
            ttt01 = time.time()
            
            axb[0].plot(0*wk+dx, wk, lw=0, marker=r'$\mathtt{W}$', color='blue',  label='Wake', ms=12)
            axb[0].plot(0*rm+dx, rm, lw=0, marker=r'$\mathtt{R}$', color='red',   label='REM',  ms=12)
            axb[0].plot(0*nr+dx, nr, lw=0, marker=r'$\mathtt{N}$', color='green', label='Non REM', ms=12)

            axb[0].text(dx, ia, scoreTag, rotation=90, ha='center', va='top', fontfamily='monospace')

            ttt02 = time.time()

            # # markers
            # mdic = {}
            # mdic['Wake'] = r'$\mathtt{W}$'
            # mdic['REM'] =  r'$\mathtt{R}$'
            # mdic['Non REM'] = r'$\mathtt{N}$'
            # cdic = {}
            # cdic['Wake'] = 'blue'
            # cdic['REM'] =  'red'
            # cdic['Non REM'] = 'green'
            # yy = dfj['Epoch#'].values
            # xx = yy*0+dx
            # colors = [cdic.get(x, 'gray') for x in dfj['Score'].values]
            # markers = [mdic.get(x, 'x') for x in dfj['Score'].values]
            # axb[0].scatter(xx, yy, c=colors, marker=markers)


        axb[0].set_ylim([ia-1, ib+1])



        t80 = time.time()



        #======================================================================
        #======================================================================
        #======= ANNOTATIONS (render time, timestamp)

        try:
            for an in self.annotations:
                an.remove()
        except:
            # only should happen once
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

        t99 = time.time()


        #txt = 'time to render: %3.0f ms' % ((t99-t00)*1000)
        #tx0 = figx.text(0.99, 0.99, txt, ha='right', va='top', fontsize=24)

        txt = datetime.datetime.now().replace(microsecond=0).isoformat().replace('T','\n')
        tx1 = figx.text(0.01, 0.99, txt, ha='left', va='top', fontsize=20)

        txt = 'trial: %i Epoch: %i' % (self.std.trial , self.viewEpoch)
        tx2 = figx.text(0.5, 0.99, txt, ha='center', va='top', fontsize=42)


        t_spam = []
        t_spam.append('TIMING [ms]\n')
        t_spam.append('ax setup  : %3.0f\n' % ((t20-t00)*1000))
        t_spam.append('signal_ts : %3.0f\n' % ((t40-t20)*1000))
        t_spam.append('PCA_ts    : %3.0f\n' % ((t50-t40)*1000))
        t_spam.append('PCA_2dprj : %3.0f\n' % ((t55-t50)*1000))
        t_spam.append('PCA_2dplt : %3.0f\n' % ((t60-t55)*1000))
        t_spam.append('features 1: %3.0f\n' % ((t65-t60)*1000))
        t_spam.append('features 2: %3.0f\n' % ((t70-t65)*1000))
        t_spam.append('features 3: %3.0f\n' % ((t80-t70)*1000))
        t_spam.append('tight ly  : %3.0f\n' % ((t99-t80)*1000))
        t_spam.append('TOTAL     : %3.0f\n' % ((t99-t00)*1000))

        txt = ''.join(t_spam)
        tx3 = figx.text(0.99, 0.01, txt, ha='right', va='bottom', ma='left', fontsize=12, fontfamily='monospace')


        self.annotations = [tx1, tx2, tx3]

        figx.show()

