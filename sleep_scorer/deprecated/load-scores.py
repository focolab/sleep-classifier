#!/usr/bin/env python3
#
#   - import a csv table of score files (and possibly edf files)
#   - strip out spaces in column names
#   - consolidate into trial datablocks (with consensus)
#   TODO: use relative paths in csv?
#======================================
import pdb
import os
import argparse
import pandas as pd
import numpy as np
import scoreblock as sb

raise Exception('deprecated, replaced by scoreloader.py')
#==============================================================================
pp = argparse.ArgumentParser()
pp.add_argument('-c', required=True, default=None, type=str,  help='csv table of score files')
pp.add_argument('--dest', type=str, default='ANL-load-scores', help='output folder')
args = pp.parse_args()

os.makedirs(args.dest, exist_ok=True)

# import table of score files
df = pd.read_csv(args.c, index_col=0)

# for the case for a csv with 'humanScores' and 'edf' files stacked together
if 'filetype' in df.columns:
    df = df[df['filetype'] == 'humanScores']

# load scores for each trial/day/scorer
ydata = []
for i, row in df.iterrows():
    print(row['trial'], row['genotype'], row['day'], row['scorer'])
    dfi = pd.read_csv(row['file'])
    dfi.columns = [col.replace(" ","") for col in list(dfi.columns)]
    ydata.append(dfi['Score'].values)

# combine all score vectors into a single stacked dataframe
ydata = np.asarray(ydata)
index_cols = ['trial', 'genotype', 'day', 'scorer']
data_cols = ['Epoch-%5.5i' % (i+1) for i in range(ydata.shape[1])]
df_data = pd.DataFrame(ydata, columns=data_cols)
df_index = df[index_cols]
df_scores = pd.concat([df_index, df_data], axis=1)


# dump scoreblocks (with consensus) for each trial/day combo
td_uniq = df_index[['trial','day']].drop_duplicates().values
data = []
for trial, day in td_uniq:
    df_td = df_scores[(df_index['trial']==trial) & (df_index['day']==day)].copy()
    df_td.reset_index(drop=True, inplace=True)

    sb_td = sb.ScoreBlock(df=df_td, index_cols=index_cols)
    sb_td.add_const_index_col(name='scoreType', value='human', inplace=True)

    sb_cc = sb_td.consensus()
    data.append(sb_cc)

    jf = 'scoreblock-trial-%s-day-%i.json' % (str(trial), day)
    sb_cc.to_json(os.path.join(args.dest, jf))


# make a combined scoreblock and dump it
sb_stack = data[0].stack(others=data[1:])
sb_stack.to_json(os.path.join(args.dest, 'scoreblock-alldata-raw.json'))

# score fractions
sb_count = sb_stack.count(frac=True)
sb_count.to_json(os.path.join(args.dest, 'scoreblock-alldata-frac.json'))
