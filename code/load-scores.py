#!/usr/bin/env python3
#
#   - import human scoring datafiles
#   - strip out spaces in column names
#   - consolidate into trial dataframes (with trial/scorer info)
#   - tabulate consensus count (all humans agree)
#
#======================================
import pdb
import os
import argparse
import re

import pandas as pd
import numpy as np

import scoreblock as sb

# def parse_fnm(x):
#     """
#     file naming formats:
#         374scores_NG.txt    -> [374, 'NG']
#         374_scores.txt      -> [374, 'ANON']
#     """
#     xx = re.split('scores_|\\.|_scores\\.', x)
#     print(xx)
#     if xx[1] == 'txt':
#         return [xx[0], 'ANON']
#     else:
#         return xx[0:2]
# def get_score_consensus(df=None):
#     """tabulating human scoring consensus"""    
#     consensus = lambda x: x[0] if len(np.unique(x)) == 1 else 0
#     pp = df.pivot_table(index=['Epoch#', 'trial'], columns='scorer', values='Score#')
#     pp['consensus'] = pp.apply(consensus, axis=1)
#     qq = pp['consensus'].value_counts().sort_index().to_frame()
#     qq['fractional'] = qq/np.sum(qq.values)
#     qq.index.name = 'Score#'
#     qq.reset_index(inplace=True)
#     return qq

def consensus(X, val='XXX'):
    """element-wise consensus among lists in X"""
    c = [y[0] if len(np.unique(y)) == 1 else val for y in zip(*X)]
    return c

#=========================================================================================
pp = argparse.ArgumentParser()
pp.add_argument('-f', nargs='+', type=str, help='input files')
pp.add_argument('-c', default=None, type=str,  help='csv table of score files')
pp.add_argument('--dest', type=str, default='ANL-load-scores', help='output folder')
args = pp.parse_args()

os.makedirs(args.dest, exist_ok=True)

if args.c is not None:
    print('CSV FILE:', args.c)

    # import table of score files
    df = pd.read_csv(args.c, index_col=0)

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
    #df_scores = df_scores.astype(dict('trial'=int, 'day'=int)) 


    # consensus scores for each unique trial/day combination
    td_uniq = list(set([tuple(x) for x in df_index[['trial', 'day']].values]))
    data = []
    for trial, day in td_uniq:
        df_td = df_scores[(df_index['trial']==trial) & (df_index['day']==day)].copy()
        copyrow = df_td.iloc[0].copy()
        copyrow[data_cols] = consensus(df_td[data_cols].values)
        copyrow['scorer'] = 'consensus'
        data.append(copyrow)

    # stack consensus series
    df_consensus = pd.concat(data, axis=1).T

    # stack the human scores and consensus, sort, reset the index
    df_stack = pd.concat([df_scores, df_consensus], axis=0)
    df_stack.sort_values(by=['trial', 'day'], inplace=True)
    df_stack.reset_index(drop=True, inplace=True)

    # make a scoreblock and dump it
    sb_stack = sb.ScoreBlock(df=df_stack, index_cols=index_cols)
    sb_stack.to_json(os.path.join(args.dest, 'scoreblock-alldata-raw.json'))

    # score fractions
    sb_count = sb_stack.count(frac=True)
    sb_count.to_json(os.path.join(args.dest, 'scoreblock-alldata-frac.json'))

    # also dump scoreblocks for each trial/day combo
    for trial, day in td_uniq:
        df_td = df_stack[(df_stack['trial']==trial) & (df_stack['day']==day)]
        sb_td = sb.ScoreBlock(df=df_td, index_cols=index_cols)
        jf = 'scoreblock-trial-%s-day-%i.json' % (str(trial), day)
        sb_td.to_json(os.path.join(args.dest, jf))


#==============================================================================
# deprecate everything below here
#==============================================================================
#==============================================================================
for i in range(10):
    print('load-scores.py: DEPRECATE THIS CRAP OMFG')

# data = []
# summary_data = []
# #== load scoring files into DataFrames and append with trial and scorer, 
# for ff in args.f:
#     base = os.path.basename(ff)
#     trial, scorer = parse_fnm(base)

#     df = pd.read_csv(ff)
#     df.columns = [col.replace(" ","") for col in list(df.columns)]
    
#     df['trial'] = [trial]*len(df)
#     df['scorer'] = [scorer]*len(df)
#     data.append(df)

#     dd = dict(trial=trial,
#               scorer=scorer,
#               num_epochs=len(df),
#               score_file_base=base)
#     summary_data.append(dd)
#     print('load trial/scorer/file:', [trial, scorer, ff])

# #== summarize the input (meta)data
# df_summary = pd.DataFrame(data=summary_data)
# df_summary.to_csv(os.path.join(args.dest,'metadata-input.csv'))

# #== concatenate everything
# df_raw = pd.concat(data)
# df_raw.reset_index(inplace=True, drop=True)
# #df_raw.to_csv(os.path.join(args.dest,'data-all.csv'))

# #== hash to convert 'Score#' to 'Score'
# aa = df_raw['Score#'].tolist()
# bb = df_raw['Score'].tolist()
# scoreNum2Name = dict(set(list(zip(aa,bb))) )
# scoreNum2Name[0] = 'XXX'

# #== dump raw scores
# for trial in df_raw['trial'].unique():
#     csv = os.path.join(args.dest, 'data-scores-trial-%s.csv' % (trial))
#     df_trial = df_raw[df_raw['trial'] == trial].reset_index(drop=True)
#     df_trial.to_csv(csv)


# #=========================================
# # output dataframes with score vectors (including consensus) as rows
# stack = []
# for trial in df_summary['trial'].unique():
#     dft = df_raw[df_raw['trial'] == trial]
#     scorers = dft['scorer'].unique()
#     scores = [dft[dft['scorer']==sc]['Score'].values.tolist() for sc in scorers]

#     num_epochs = len(scores[0])

#     def consensus(X, val='XXX'):
#         """element-wise consensus among lists in X"""
#         c = [y[0] if len(np.unique(y)) == 1 else val for y in zip(*X)]
#         return c

#     scores.append(consensus(scores))

#     cols = ['epoch-%5.5i' % (e+1) for e in range(num_epochs)]
#     df_scores = pd.DataFrame(data=scores, columns=cols)
#     df_scores['trial'] = [trial]*len(scores)
#     df_scores['scorer'] = scorers.tolist()+['consensus']
#     df_scores = df_scores[['trial','scorer']+cols]
#     stack.append(df_scores)

#     # csv = os.path.join(args.dest, 'df_scores-trial-%s-scoreblock.csv' % str(trial))
#     # df_scores.to_csv(csv)

# df_stack = pd.concat(stack, axis=0).reset_index(drop=True)
# csv = os.path.join(args.dest, 'df_scores-alltrials-scoreblock.csv')
# df_stack.to_csv(csv)


# # # THIS THIS THIS
# # sb_scores = sb.ScoreBlock(df=df_stack, index_cols=['trial', 'scorer'])
# # sb_scores.to_json(os.path.join(args.dest, 'scoreblock-humans.json'))

# #=========================================
# #== output score consensus info for each trial
# all_df = []
# for trial in df_summary['trial'].unique():
#     ndx = df_raw['trial'] == trial    
#     qq = get_score_consensus(df_raw[ndx])
#     qq['Score'] = [scoreNum2Name[x] for x in qq['Score#']]
#     qq['Trial'] = [trial]*len(qq)
#     print('----------------------')
#     print('consensus for trial:',trial)
#     print(qq.head(10))
#     # qq.to_csv(os.path.join(args.dest,'consensus-%s.csv' % trial))
#     all_df.append(qq)

# #== overall consensus fractions
# df_consensus = get_score_consensus(df_raw)
# df_consensus['Score'] = [scoreNum2Name[x] for x in df_consensus['Score#']]
# df_consensus.to_csv(os.path.join(args.dest, 'consensus-full.csv'))


# #== df of consensus fractions for all trials
# trials = df_summary['trial'].unique()
# data = [ {x:y for [x,y] in df[['Score','fractional']].values} for df in  all_df]
# df_all = pd.DataFrame(data=data).fillna(0)
# df_all.index = trials
# df_all.index.name = 'trial'
# df_all.to_csv(os.path.join(args.dest, 'fractional_consensus.csv'), float_format='%g')


# print(df_all)


