#
#   stage edf files and scoreblocks (built from individual score files)
#

import argparse
import scoreloader as scl

pp = argparse.ArgumentParser()
pp.add_argument('-c', required=True, default=None, type=str,  help='csv table of score files and edf files')
pp.add_argument('--dest', type=str, default='ANL-stage-edf-scores', help='output folder')
args = pp.parse_args()

scl.stage_edf_and_scores(csv=args.c, dest=args.dest)
