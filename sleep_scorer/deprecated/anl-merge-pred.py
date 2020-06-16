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

    raise Exception('deprecated, human scores are automatically appended to predicted scores')

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







