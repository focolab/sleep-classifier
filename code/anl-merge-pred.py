#!/usr/bin/env python3

import os
import argparse
import pdb
import datetime

import scoreblock as sb


if __name__ == '__main__':
    """
    combine score predictions, staging for subsequent plotting or whatev
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--hh', type=str, help='human score scoreblock')
    parser.add_argument('--mm', type=str, help='model score scoreblock')
    parser.add_argument('--dest', default='ANL-merge-scores', help='output folder')
    args = parser.parse_args()

    print('#=================================================================')
    print('#                        anl-merge-pred.py')
    print('#=================================================================')
    print('human scoreblock: %s' % (args.hh))
    print('model scoreblock: %s' % (args.mm))


    rename_index_cols = {'T':'trial', 'C':'classifier'}
    os.makedirs(args.dest, exist_ok=True)

    # import
    sb_mod = sb.ScoreBlock.from_json(args.mm)
    sb_hum = sb.ScoreBlock.from_json(args.hh)

    # stack
    sb_stack = sb_mod.stack(
        others=[sb_hum],
        force_data=True,
        rename_index_cols=rename_index_cols
        )

    # get score fractions
    sb_stack_fractions = sb_stack.count(frac=True)

    # annotate with ancestry
    ancestry = dict(
        human_scores=os.path.abspath(args.hh),        
        model_scores=os.path.abspath(args.mm)
        )

    sb_stack.ancestry = ancestry
    sb_stack_fractions.ancestry = ancestry

    # dump em out
    sb_stack.to_json(os.path.join(args.dest, 'scoreblock-raw-merged.json'))
    sb_stack_fractions.to_json(os.path.join(args.dest, 'scoreblock-frac-merged.json'))




    # #========================= IMPORTS =======================
    # if args.c is not None:
    #     df_gt = pd.read_csv(args.c, index_col=0)
    #     trial_to_genotype = dict(zip(df_gt['trial'], df_gt['genotype']))

    # with open(args.f) as jfopen:
    #     jdic = json.load(jfopen)


    # # import model and human prediction data
    # csv = os.path.join(jdic['loc'], jdic['df_pred_csv'])
    # df_pred = pd.read_csv(csv, index_col=0)
    # df_human = pd.read_csv(args.s, index_col=0)



    # # scoreblock for model scoring
    # sb_pred = sb.ScoreBlock(df=df_pred, index_cols=jdic['cols_index'])

    # # scoreblock for human scoring
    # index_cols_human = ['trial', 'scorer']
    # sb_human = sb.ScoreBlock(df=df_human, index_cols=index_cols_human)


    # # stack the model and human scoring
    # sb_stack = sb_human.stack(
    #     others=[sb_pred],
    #     force_data=True,
    #     rename_index_cols=rename_index_cols
    #     )
    
    # # add a genotype column
    # df_stack = sb_stack.df
    # gt = [trial_to_genotype[t] for t in df_stack['trial']]
    # df_stack['genotype'] = gt
    # sb_stack.index_cols.append('genotype')





    # print('#==================================')
    # sb_stack.about()
    # sb_stack_counts.about()
    # sb_stack_fractions.about()

    #pdb.set_trace()

    # # export
    # dd = dict(
    #     _about="merged state predictions (model/human)",
    #     loc=os.path.abspath(args.dest),
    #     df_pred_csv='df_pred.csv',
    #     df_pred_counts_csv='df_pred_counts.csv',
    #     df_pred_fractions_csv='df_pred_fractions.csv',
    #     index_cols=sb_stack.index_cols,
    # )

    # # export
    # out = os.path.join(args.dest, 'predictions.json')
    # with open(out, 'w') as jout:
    #     json.dump(dd, jout, indent=2, sort_keys=False)
    #     jout.write('\n')

    # # export scoreblocks
    # sb_stack.df.to_csv(          os.path.join(args.dest, dd['df_pred_csv']))
    # sb_stack_counts.df.to_csv(   os.path.join(args.dest, dd['df_pred_counts_csv']))
    # sb_stack_fractions.df.to_csv(os.path.join(args.dest, dd['df_pred_fractions_csv']))




    # pdb.set_trace()

    #===================================================
    # WWRW
    # 1. import model and human scores
    # 2. stack
    # 3. add columns (genotype)
    # 4. count
    # 5. restack
    # 6. plot
    #===================================================








