#!/usr/bin/env python3

#
#   intended to supplant load-scores.py
#

import pdb
import os

import pandas as pd
import numpy as np
import scoreblock as sb

class SireniaScoreLoader(object):
    """load and consolidate sirenia score files (txt) into scoreblocks"""

    def __init__(self, df=None, autoload=True):
        """
        required df columns: 'trial', 'genotype', 'day', 'scorer', 'file'
        """

        self.df_flz = df

        if autoload:
            self.load()
        else:
            self.data = []
            self.sb_stack = None
            self.sb_count = None


    @classmethod
    def from_csv(cls, csv, autoload=True):
        """build from a csv file of sirenia (txt) files and metadata
        
        required columns
            trial
            genotype
            day
            scorer
            file
        """
        # For the case where the csv has 'humanScores' and 'edf' files stacked,
        # filter out the scores
        df = pd.read_csv(csv, index_col=0)
        df = df[df['filetype'] == 'humanScores']

        # convert files to absolute paths
        loc = os.path.dirname(os.path.abspath(csv))
        fabs = [os.path.join(loc, f) for f in df['file']]
        df['file'] = fabs

        return cls(df=df, autoload=autoload)

    def load(self, consensus=True):
        """load score files, group by trial/day, make scoreblocks, consensus"""

        df = self.df_flz

        # load scores for each trial/day/scorer
        ydata = []
        # start_times = []
        for i, row in df.iterrows():
            #print(row['trial'], row['genotype'], row['day'], row['scorer'])
            dfi = pd.read_csv(row['file'])
            dfi.columns = [col.replace(" ","") for col in list(dfi.columns)]
            ydata.append(dfi['Score'].values)
            # start_times.append(dfi['StartTime'].values[0])

        # combine all score vectors into a single stacked dataframe
        ydata = np.asarray(ydata)
        index_cols = ['trial', 'genotype', 'day', 'scorer']
        data_cols = ['Epoch-%5.5i' % (i+1) for i in range(ydata.shape[1])]
        df_data = pd.DataFrame(ydata, columns=data_cols)
        df_index = df[index_cols].reset_index(drop=True)
        df_scores = pd.concat([df_index, df_data], axis=1)

        # dump scoreblocks (with consensus) for each trial/day combo
        df_uniq = df_index[['trial','day']].drop_duplicates()
        td_uniq = df_uniq.values
        data = []
        for trial, day in td_uniq:
            df_td = df_scores[(df_index['trial']==trial) & (df_index['day']==day)] #.copy()
            df_td.reset_index(drop=True, inplace=True)

            sb_td = sb.ScoreBlock(df=df_td, index_cols=index_cols)
            sb_td.add_const_index_col(name='scoreType', value='human', inplace=True)

            if consensus:
                sb_cc = sb_td.consensus()
                if sb_td.numrows == 1:
                    # in the case of just one scorer, the 2nd row is a mock consensus
                    # NOTE: use iloc assignment here to avoid warnings about slice/copy/assignments
                    sb_cc.df.iloc[1]['scorer'] = 'consensus'
            else:
                sb_cc = sb_td

            data.append(sb_cc)
            # jf = 'scoreblock-trial-%s-day-%i.json' % (str(trial), day)
            # sb_cc.to_json(os.path.join(args.dest, jf))


        # make a combined scoreblock and dump it
        sb_stack = data[0].stack(others=data[1:])

        # score fractions
        sb_count = sb_stack.count(frac=True)
        # sb_count.to_json(os.path.join(args.dest, 'scoreblock-alldata-frac.json'))

        self.data = data
        self.sb_stack = sb_stack
        self.sb_count = sb_count


    def dump_trial_blocks(self):
        """dump single trial scoreblocks"""
        pass

    def dump_cat_block(self, jf):
        """dump concatenated block"""
        self.sb_stack.to_json(jf)




def stage_edf_and_scores(csv=None, dest='anl-staged'):
    """stage edf and (sirenia) scores

    - consolidates scores and dumps a scoreblock for each edf
    - bundles edfs and corresponding scoreblocks for downstream analysis

    input
    ------ 
    csv: str
        csv file of edf and (sirenia) score files
        Required cols: 
            'trial'     : int or str
            'day'       : int
            'scorer'    : str
            'genotype'  : str
            'filetype'  : 'edf' or 'humanScores'
            'file'      : relative path to the file

    dest : str
        output folder

    output
    ------
    df_out : dataframe
        edf files and corresponding scoreblocks (if scores exist)
 
    """

    df = pd.read_csv(csv, index_col=0)

    # convert filepaths from relative (relative to csv) to absolute
    dfloc = os.path.dirname(os.path.abspath(csv))
    df['file'] = [os.path.join(dfloc, x) for x in df['file']] 

    os.makedirs(dest, exist_ok=True)

    # split into edf files and score files
    df_edf = df[df['filetype'] == 'edf']
    df_scores = df[df['filetype'] == 'humanScores']

    # loop over edf files
    scoreblock_json_files = []
    for _, row in df_edf.iterrows():
        trial = row['trial']
        day = row['day']
        tag = 'trial-%s-day-%s' % (str(trial), str(day))
        print('staging:', tag)

        df_td = df_scores[(df_scores['trial']==trial) & (df_scores['day']==day)].copy()

        # if there are score files for the same trial and day
        if len(df_td) > 0:
            scoreloaderA = SireniaScoreLoader(df=df_td)
            jf = os.path.join(os.path.abspath(dest), 'scoreblock-%s.json' % (tag))
            scoreloaderA.dump_cat_block(jf)
        else:
            jf = ''

        scoreblock_json_files.append(jf)

    # build a master dataframe of edf/scoreblock files
    keepcols = ['trial', 'day', 'genotype']
    df_out = df_edf[keepcols].reset_index(drop=True)
    df_out['scores'] = scoreblock_json_files
    df_out['edf'] = df_edf['file'].values

    # dump it
    csv_out = os.path.join(os.path.abspath(dest), 'csv-staged-data.csv')
    df_out.to_csv(csv_out)

    return df_out


#==============================================================================
#==============================================================================


def test_loader():
    """load a csv (of txt files), compute trial scoreblocks, dump"""

    os.makedirs('scratch', exist_ok=True)

    # find the example data
    dn = os.path.dirname(__file__)
    csv = '../example_data/files-data-A-training.csv'
    f = os.path.join(dn, csv)

    # build score loader
    ssl = SireniaScoreLoader.from_csv(f)

    # dump scoreblocks
    jf = os.path.join('scratch', 'scoreloader-test-alldata.json')
    ssl.dump_cat_block(jf)


    print('test_loader success')

if __name__ == "__main__":

    test_loader()


