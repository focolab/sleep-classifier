

import os
import pdb
import datetime

import numpy as np
import pandas as pd


import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt


# #import plottools as pt
# import remtools as rt
# import scoreblock as sb
# import eegplotter as ep





class TraceDumper(object):
    """visualize a 24h recording and accompanying scores
    
    This will fit 1 hour of data per page using 6 rows/page and 60 epochs/row
    
    
    """
    def __init__(self, std=None, scores=None):
        """
        
        input
        ------
        std (StagedTrialData) : data class
        scores (ScoreBlock) : block of scores (human,model etc)

        """
        
        import itertools

        self.std = std
        self.scores = scores
        

        # hard codey stuff
        self.epoch_length = 10          # 10s
        self.epochs_per_panel = 60     # 180 = 30 min
        self.panels_per_page = 6
        signal_names = ['EEG1', 'EEG2', 'EMG']

        
        # stage data traces, precompute strides etc ranges
        self.signals = {}
        for name in signal_names:
            trace = std.edf.signal_traces[name]

            epoch_stride = int(trace.samples_per_epoch)
            panel_stride = int(trace.samples_per_epoch*self.epochs_per_panel)
            page_stride = int(panel_stride*self.panels_per_page)

            num_pages = int(np.ceil(len(trace.sig)/page_stride))

            #print('-------------------------')
            #print('trace       :', name)
            #print('len trace   :', len(trace.sig))
            #print('panel_stride:', panel_stride)
            #print('page_stride :', page_stride)
            #print('num_pages   :', num_pages)

            pp = itertools.product(range(num_pages), range(self.panels_per_page))
            df = pd.DataFrame(data=pp, columns=['page', 'panel'])
            df['ia'] = df['page']*page_stride + df['panel']*panel_stride
            df['ib'] = df['ia'] + panel_stride

            sig = trace.sig/np.std(trace.sig)
            
            
            dd = dict(
                name=name,
                df_index=df,
                tvec=trace.tvec,
                sig=sig,
                epoch_stride=epoch_stride,
                panel_stride=panel_stride,
                page_stride=page_stride,
                num_pages=num_pages
            )

            self.signals[name] = dd
        
        
        #print(self.std.edf.metadata)
        self.num_pages = np.unique([x['num_pages'] for x in self.signals.values()])
        if len(self.num_pages) > 1:
            raise Exception('num_pages conflict')
        else:
            self.num_pages = self.num_pages[0]
        
        
        # panel/epoch indexing stuff
        # epoch time series
        pp = itertools.product(range(num_pages), range(self.panels_per_page))
        df = pd.DataFrame(data=pp, columns=['page', 'panel'])
        
        df['ta'] = (df['page']*self.panels_per_page + df['panel'])*self.epochs_per_panel*self.epoch_length
        df['tb'] = df['ta']+self.epochs_per_panel*self.epoch_length
        
        #df['ea'] = df['ta']/self.epoch_length+1
        #df['eb'] = df['ea']+self.epochs_per_panel
        #print(df.head(20))
        
        self.df_panel_index = df
        


        self.make_fig()
        

    def make_fig(self):
        """create the figure"""

        # nicer default colors
        aa = '707070'
        #aa = 'a0a0a0'
        b0 = 'b0b0b0'
        gg = 'c0c0c0'
        
        # black background
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
        
        # white bg w/gray lines/text
        # good for printing
        pp = {
            'lines.linewidth':2,
            'axes.edgecolor': aa,
            'axes.labelcolor': aa,
            'xtick.color' : aa,
            'ytick.color' : aa,
            'grid.color'  : aa,
            'text.color' : aa
        }

        matplotlib.rcParams.update(pp)
        
        self.fig = plt.figure(figsize=(24, 16), dpi=60)


    def export(self, filename='trace-dump.png'):
        """"""
        self.fig.savefig(filename)
        


    def export_full_pdf(self, f=None, pages=None, dpi=150):
        """export all pages to pdf file 

        f (str) : filename
        pages (list, range) : pages to export
        dpi (int) : dpi for rasterization (only effects timeseries)


        Rasterization schemes (implemented in render_page())
        ------
        # fully rasterize the plot
        ax = fig.add_subplot(111, rasterized=True)
        
        # rasterize anything below some zorder
        gca().set_rasterization_zorder(1)
        plot(randn(100),randn(100,500),"k",alpha=0.03,zorder=0)
        savefig("test.pdf",dpi=90)

        """

        from matplotlib.backends.backend_pdf import PdfPages
        
        if f is None:
            trial = int(self.std.tagDict.get('trial', 0))
            day = (self.std.tagDict.get('day', 0))
            f = 'trace-dump-trial-%4.4i-day-%i-dpi-%i.pdf' % (trial, day, dpi)       

        if pages is None:
            pages = range(self.num_pages)

        print('pdf ouput to: %s' % f)
        with PdfPages(f) as pdf:
            for page in pages:
                print('  rendering page', page)                
                self.render_page(page=page)
                pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page

        print('pdf ouput complete')

        
        
        
    def render_page(self, page=0):
        """hot mess"""
        


        scoremap = {
            'Wake':0.9999,
            'REM':0.5,
            'Non REM':0.0,
            'Non REM X':np.nan,
            'REM X':np.nan,
            'Unscored':np.nan,
            'Wake X':np.nan,
            'XXX':np.nan
        }

        epoch_tick_stride = 6
        ts_stride = 1
        
        dy_pos = +7
        dy_neg = -2


        plot_signal_kwa = dict(lw=1, color='grey', alpha=0.7, zorder=-1)
        
        # nuke it all and start from scratch (obviously slow, but hey)
        ppp = self.panels_per_page
        self.fig.clf()
        ax = [plt.subplot(ppp, 1, i+1) for i in range(ppp)]


        # SIGNALS plotted w/ zorder -1 are rasterized on pdf/svg export
        for axi in ax:
            axi.set_rasterization_zorder(0)


        # one panel at a time
        for panel in range(self.panels_per_page):
            # time series plotted w positive dy shifts
            # scores plotted w negative dy shifts


            # PANEL STUFF
            dfp = self.df_panel_index.copy()
            dfp =  dfp[(dfp['page'] == page) & (dfp['panel'] == panel)]
            
            # time limits
            [ta, tb] = dfp[['ta','tb']].values.ravel()

            # epoch limits
            ea = int(ta/self.epoch_length)
            eb = int(ea+self.epochs_per_panel)
            
            #----------------------------------
            # RAW SIGNALS time series
            for iy, name in enumerate(['EMG', 'EEG1', 'EEG2']):

                # positive y offset
                dyi = (iy+1)*dy_pos
                ylim_max = (iy+1+0.5)*dy_pos

                # plot the slice of timeseries
                sig = self.signals[name]
                df = sig['df_index']
                df = df[(df['page'] == page) & (df['panel'] == panel)]
                [ia, ib] = df[['ia', 'ib']].values.ravel()
                sl = slice(ia, ib, ts_stride)
                ax[panel].plot(sig['tvec'][sl], sig['sig'][sl]+dyi, **plot_signal_kwa)

                # label
                ax[panel].text(tb+1, dyi, name, ha='left', va='center')


            #----------------------------------
            # SCORE LABELS (categorical)
            scoreTypes = self.scores.df['scoreType'].unique()
            scoreTags = self.scores.df['scoreTag'].unique()

            for j, scoreTag in enumerate(scoreTags):

                # negative y offset
                dyj = dy_neg*(0.5+j)
                ylim_min = dy_neg*(1+j)

                # label
                ax[panel].text(tb+1, dyj, scoreTag, ha='left', va='center')
                
                # reduce clutter: only plot letter labels for one row of scores
                if j == len(scoreTags)-1:
                    data = self.scores.keeprows(conditions=[('scoreTag', scoreTag)])

                    dfj = data.data.ravel()[ea:eb]
                    tvals = np.linspace(ea, eb-1, self.epochs_per_panel)*self.epoch_length
                    tvals += self.epoch_length/2
                    
                    wk = tvals[dfj == 'Wake']
                    rm = tvals[dfj == 'REM']
                    nr = tvals[dfj == 'Non REM']
                    
                    kwa = dict(ms=5, alpha=1, lw=0, mec='gray', mew=0)

                    ax[panel].plot(wk, 0*wk+dyj, marker=r'$\mathtt{W}$', color='blue',  label='Wake', **kwa)
                    ax[panel].plot(rm, 0*rm+dyj, marker=r'$\mathtt{R}$', color='red',   label='REM',  **kwa)
                    ax[panel].plot(nr, 0*nr+dyj, marker=r'$\mathtt{N}$', color='green', label='Non REM', **kwa)


            #----------------------------------
            # IMSHOW panel for scores
            # make scores numerical (WOW that is some sexy method chaining ^^)
            scnum = self.scores.mask(mask=slice(ea, eb)).applymap(scoremap)
            
            extent = (ta, tb, 0, ylim_min)
            # 'tab10' also a cmap option
            ax[panel].imshow(scnum.data, origin='lower', cmap='plasma', aspect='auto', extent=extent, vmin=0, vmax=1)
            

            #----------------------------------
            # FORMATTING FANCYNESS
            numticks = int(self.epochs_per_panel/epoch_tick_stride)+1
            xtk = np.linspace(ta, tb, numticks)

            ax[panel].set_xticks(xtk)
            ax[panel].set_yticks([])
            ax[panel].spines['top'].set_visible(False)
            ax[panel].spines['bottom'].set_visible(False)
            ax[panel].spines['left'].set_visible(False)
            ax[panel].spines['right'].set_visible(False)
            ax[panel].grid()
            ax[panel].set_xlim([ta, tb])
            ax[panel].set_ylim([ylim_min, ylim_max]) 
            ax[panel].axhline(y=0, lw=1, color='grey')

    
        # global tidying, annotation
        plt.tight_layout(h_pad=4)
        plt.subplots_adjust(top=0.95)

        txt = datetime.datetime.now().replace(microsecond=0).isoformat().replace('T','\n')
        tx1 = self.fig.text(0.01, 0.99, txt, ha='left', va='top', fontsize=20)

        tdp = (self.std.trial, self.std.tagDict['day'], page+1, self.num_pages)
        txt = 'trial: %i  day: %i  page: %i/%i' % tdp
        tx2 = self.fig.text(0.5, 0.99, txt, ha='center', va='top', fontsize=42)

        self.fig.show()
