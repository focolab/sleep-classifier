#!/usr/bin/env python3


import pdb
import os
import json


import numpy as np
import pandas as pd


def ppd(d):
    for k, v in d.items():
        print('%15s:' % k, v)

def consensus(X, val='XXX'):
    """element-wise consensus among lists in X"""
    c = [y[0] if len(np.unique(y)) == 1 else val for y in zip(*X)]
    return c

class ScoreBlock(object):
    """stack of y vectors (human/model) as a DataFrame w/ extra features

    - The dataframe comprises multiple y vectors, stacked as row vectors,
        and >=1 index columns that adjoin the ts data stack.
    - This allows for organized storage across trials, models, human scorers
        and so on
    - Not a pandas multi-index, but uses attributes index_cols and data_cols
    - The dump/load methods provide lossless storage (whereas loading a pandas
        dataframe from csv requires user input to recreate the multi index,
        or pickling). Calling dump creates a json file (with metadata) and a
        csv (the dataframe). The json file is used for load().

    attributes
    ------
    df : pandas.DataFrame
        the dataframe
    data_cols : list
        names of the data columns, i.e. [epochs-0001, epoch-0002, ...]
    index_cols : list
        names of index columns, things like trial, classifier, scorer, etc
    tagDict: dict
        dictionary of tags and metadata, (must be jsonizable)
    ancestry: dict
        PRELIMINARY some way of tracking input data

    methods
    ------
    about():
        print an executive summary of this scoreblock
    stack():
        stack blocks on top of each other
    count():
        count occurances of (categorical) states in each (data) row
    to_json():
        dump dataframe to csv and everything else to json
    from_json():
        load from a previously dumped json
    applymap():
        apply a map (dict) to categorical data, (good for re-naming/merging
        states)
    keeprows():
        keep a subset of rows, based on conditions
    
    consensus():
        determine (binary) consensus for each column of df (index and data)


    """
    def __init__(self, loc=None, df=None, index_cols=None, data_cols=None, tagDict={},
                 ancestry={}):
        """ """
        self.loc = loc
        self.df = df
        self.tagDict = tagDict

        self.ancestry = ancestry

        if index_cols is None:
            raise Exception('index_cols required')
        else:
            self.index_cols = index_cols

        if data_cols is None:
            self.data_cols = [c for c in df.columns if c not in self.index_cols]
        else:
            self.data_cols = data_cols


    @property
    def data(self):
        return self.df[self.data_cols].values

    @property
    def df_index(self):
        return self.df[self.index_cols].copy()

    @property
    def uniqueScores(self):
        return np.unique(self.data).tolist()

    @property
    def numrows(self):
        return self.df.shape[0]

    def about(self):
        """executive summary"""
        opj = os.path.join
        print('-------- ScoreBlock.about() --------')
        print('loc       :', self.loc)
        print('index_cols:', self.index_cols)
        if len(self.data_cols)>6:
            print('data_cols :', self.data_cols[:3], '...', self.data_cols[-3:])
        else:
            print('data_cols :', self.data_cols)
        print('tagDict   :')
        for k,v in self.tagDict.items():
            print('%15s:' % k, v)
        print('ancestry  :')
        for k,v in self.ancestry.items():
            print('%15s:' % k, v)
        print('df        :')
        print(self.df.head())


    def add_const_index_col(self, name='newcolumn', value=None):
        """add an index column with a constant value"""
        df_out = self.df.copy()
        df_out[name] = [value]*len(df_out)

        index_cols = self.index_cols+[name]
        df_out = df_out[index_cols+self.data_cols]

        out = ScoreBlock(
            df=df_out,
            index_cols=index_cols,
            tagDict=self.tagDict
            )
        return out

    def copy_index_cols(self):
        """"""
        return [c for c in self.index_cols]

    def applymap(self, m):
        """apply a dictionary map 'm' to DATA values"""

        fmap = lambda x: m.get(x, x)
        df_out = self.df.copy()
        df_out[self.data_cols] = df_out[self.data_cols].applymap(fmap)

        out = ScoreBlock(
            df=df_out,
            index_cols=self.copy_index_cols(),
            tagDict=self.tagDict
            )
        return out


    def stack(self, others=[], force_data=False, rename_index_cols=None,
              verbose=False, data_nan=None,
              ):
        """stack multiple scoreblocks and return a new block

        force_data: ignore mismatching data column names and force an aligned 
            data stack (requires the same number of data columns) 
        rename_index_cols: to fix mismatching index columns, prior to stacking

        TODO: possible to just convert to multi-index and concat?
        """
        
        sblocks = [self] + others

        if verbose:
            print('#== stacking ScoreBlocks:')
            for i, sblock in enumerate(sblocks):
                print('%3i:' % i, 'index_cols=', sblock.index_cols,' data_cols[:4]=',sblock.data_cols[:4])

        # find out if data columns have different lengths
        ragged_data = False
        if len(list(set([len(sblock.data_cols) for sblock in sblocks]))) > 1:
            #raise Exception('data_cols have non-matching lengths')
            print('WARNING: data_cols have non-matching lengths')
            ragged_data = True
            if force_data:
                raise Exception('force_data will not work for ragged data')

        # warn if data columns have different names
        dcols = list(zip(*[sblock.data_cols for sblock in sblocks]))
        counts = list(set([len(set(tuple(x))) for x in dcols]))
        if len(counts)>1 or 1 not in counts:
            print('WARNING: stacking data_cols with non-matching names')

        # index stack (with column renaming)
        dfs = [sb.df[sb.index_cols].copy() for sb in sblocks]
        if rename_index_cols is not None:
            dfs = [df.rename(columns=rename_index_cols) for df in dfs]
        df_index = pd.concat(dfs, sort=False).reset_index(drop=True)

        # data stack
        if force_data == True:
            data = np.vstack([sb.df[sb.data_cols] for sb in sblocks])
            df_data = pd.DataFrame(data=data, columns=self.data_cols).reset_index(drop=True)
        else:
            tostack = [sb.df[sb.data_cols] for sb in sblocks]
            df_data = pd.concat(tostack, sort=True).reset_index(drop=True)

        if data_nan is not None:
            df_data = df_data.fillna(data_nan)

        # combine index and data
        df_out = pd.concat([df_index, df_data], axis=1).reset_index(drop=True)

        tagDict = dict(_about='stacked scoreblocks')

        # build a new ScoreBlock
        out = ScoreBlock(
            df=df_out,
            index_cols=df_index.columns.tolist(),
            tagDict=tagDict
            )

        return out


    def keeprows(self, conditions=[], comparison='all'):
        """keep rows from a the dataframe, subject to conditions

        declarative method for row selection based off the index dataframe

        TODO: make this take heirarchical args and 

        keeprows conditions
    
        ('colname', value, comparison)  
                ('classifier', 'OVO', 'eq')
                ('classifier', ['OVO','OVR'], 'in')


        ('colname', func)               ('classifier', lambda x: x in ['OVO', 'OVR'])
    
    
    


        input
        ------
        conditions (list): list of (col, val) tuples where col is a column
            name and val is the value that must match
        comparison (str): logical comparison for multiple conditions (all/any)
            all = intersection
            any = union

        output
        ------
        df_new (pd.DataFrame): dataframe with fewer rows

        examples
        ------
        [('trial', 335), ('day', 1), 'all']


        """

        # dictionary of pairwise comparison functions
        dd = {
            'eq': lambda a,b: a==b,
            'ne': lambda a,b: a!=b,
            'lt': lambda a,b: a<b,
            'gt': lambda a,b: a>b,
            'le': lambda a,b: a<=b,
            'ge': lambda a,b: a>=b,
            'in': lambda a,b: a in b
            #'isnan': lambda a
        }


        if comparison not in ['all', 'any']:
            raise Exception('comparison must be all or any')

        if conditions is None:
            raise Exception('need to pass conditions (at least one)')

        df = self.df.copy()

        vex = [df[col] == val for col, val in conditions]
        aa = list(zip(*vex))

        if comparison == 'all':
            kk = [all(a) for a in aa]
        if comparison == 'any':
            kk = [any(a) for a in aa]

        df_new = df[kk].reset_index(drop=True)

        out = ScoreBlock(
            df=df_new, 
            index_cols=self.copy_index_cols(),
            tagDict=self.tagDict
            )

        return out




    def consensus(self, out='append', index_fill='consensus', data_fill='XXX'):
        """compute consensus for each column of df (index and data)
        
        If a column has one unique value, it is taken as the consensus.
        Otherwise, index_fill or data_fill are used. By doing this for the
        index, all matching indices are carried forward.

        out (str) : 'append' or 'Series'
            append   : a new ScoreBlock with the consensus appended to df
            Series   : a pandas series (index and data)
            data_only: bare list of data consensus values

        index_fill (??): fill value for non-consensus indices
        data_fill (??): fill value for non-consensus data

        """

        if out not in ['Series', 'append', 'data_only']:
            raise Exception('out should be in [Series, append]')


        # consensus for the index and for the data
        c_index = consensus(self.df_index.values, val=index_fill)
        c_data = consensus(self.data, val=data_fill)
        ccc = c_index+c_data


        if out == 'Series':
            ser = pd.Series(dict(zip(self.df.columns, ccc)))
            cc = ser
        if out == 'data_only':
            cc = c_data
        elif out == 'append':
            dfc = self.df.copy()
            dfc.loc[len(dfc)] = ccc
            cc = ScoreBlock(
                df=dfc, 
                index_cols=self.copy_index_cols(),
                tagDict=self.tagDict,
                ancestry=self.ancestry,                
                )

        return cc


    def mask(self, mask=None, maskname=None, maskcolname='mask'):
        """select subset of columns and optionally mark the change in the index"""

        if mask is None:
            mask = slice(None)

        data = self.data[:, mask]
        data_cols = self.data_cols[mask]
        df_data = pd.DataFrame(data=data, columns=data_cols)

        df_index = self.df_index.copy()

        # update index with mask column?
        if maskname is not None:
            df_index[maskcolname] = [maskname]*len(self.df_index)
            index_cols = self.index_cols+[maskcolname]
        else:
            index_cols = self.index_cols

        df_new = pd.concat([df_index, df_data], axis=1)

        out = ScoreBlock(
            df=df_new,
            index_cols=index_cols,
            tagDict=self.tagDict
            )

        return out


    # # # def countXX(self, mask=None, maskname=None, maskcolname='mask', frac=False):
    # # #     """count occurances of states for each data row
        
    # # #     TODO: make mask a separate method

    # # #     - use mask to select column subsets (i.e. light and dark intervals)
    # # #     - NOTE: score_names (unique) are sorted by np.unique
    # # #     """

    # # #     def get_score_counts(data=None):
    # # #         score_names = np.unique(data)
    # # #         score_counts = []
    # # #         for row in data:
    # # #             score_counts.append({x:row.tolist().count(x) for x in score_names})
    # # #         df_counts = pd.DataFrame(score_counts)
    # # #         return df_counts

    # # #     if mask is not None:
    # # #         self.mask(mask=mask, maskname=maskname, maskcolname=maskcolname)


    # # #     # i can has mask?
    # # #     if mask is None:
    # # #         df_counts = get_score_counts(data=self.data)
    # # #     else:
    # # #         df_counts = get_score_counts(data=self.data[:, mask])


    # # #     # convert to fractions?
    # # #     if frac == True:
    # # #         rowsums = np.sum(df_counts.values, axis=1)
    # # #         for col in df_counts.columns:
    # # #             df_counts[col] /= rowsums


    # # #     dfc = pd.concat([self.df_index, df_counts], axis=1)


    # # #     index_cols = self.df_index.columns.tolist()
    # # #     data_cols = df_counts.columns.tolist()

    # # #     if maskname is not None:
    # # #         dfc[maskcolname] = [maskname]*len(df_counts)
    # # #         index_cols += [maskcolname]

    # # #     dfc = dfc[index_cols+data_cols]
    # # #     # build a new ScoreBlock
    # # #     out = ScoreBlock(
    # # #         df=dfc,
    # # #         index_cols=index_cols,
    # # #         tagDict=self.tagDict
    # # #         )

    # # #     return out


    def count(self, frac=False):
        """count occurances of states for each data row
        - NOTE: score_names (unique) are sorted by np.unique
        """

        def get_score_counts(data=None):
            names = np.unique(data)
            score_counts = [{x:row.tolist().count(x) for x in names} for row in data]
            df_counts = pd.DataFrame(score_counts)
            return df_counts

        df_counts = get_score_counts(data=self.data)

        # convert to fractions?
        if frac == True:
            rowsums = np.sum(df_counts.values, axis=1)
            for col in df_counts.columns:
                df_counts[col] /= rowsums

        dfc = pd.concat([self.df_index, df_counts], axis=1)

        # build a new ScoreBlock
        out = ScoreBlock(
            df=dfc,
            index_cols=self.copy_index_cols(),
            tagDict=self.tagDict
            )

        return out

    def to_json(self, f='scoreblock.json'):
        """dump to disk (json)
        
        - dataframe is written to csv (automatic naming)
        - errything else to json
        """

        loc = os.path.dirname(os.path.abspath(f))
        os.makedirs(loc, exist_ok=True)

        # df to csv file
        c = f.replace('.json', '.csv')
        self.df.to_csv(c, float_format='%g')

        jdic = dict(
            loc=loc,
            df_csv=os.path.relpath(c, loc),
            tagDict=self.tagDict,
            ancestry=self.ancestry,
            index_cols=self.index_cols,
        )

        with open(f, 'w') as jout:
            json.dump(jdic, jout, indent=2, sort_keys=False)
            jout.write('\n')


    @classmethod
    def from_json(cls, jsn, allow_move=True):
        """load from disk (json)"""

        with open(jsn, 'r') as jfopen:
            jdic = json.load(jfopen)

        # was this moved?
        loc_actual = os.path.dirname(os.path.abspath(jsn))
        loc_stored = jdic.get('loc', None)

        if loc_actual != loc_stored:
            if not allow_move:
                print('ERROR: scoreblock json has moved')
                print(' from: %s' % loc_stored)
                print('   to: %s' % loc_actual)
                raise Exception('scoreblock json has moved :(')
            else:
                print('WARNING: scoreblock json has moved')
                print('   from: %s' % loc_stored)
                print('     to: %s' % loc_actual)

        # load metadata
        pp = {}
        pp['loc'] = loc_actual
        pp['tagDict'] = jdic.get('tagDict', {})
        pp['ancestry'] = jdic.get('ancestry', {})
        pp['index_cols'] = jdic.get('index_cols')

        # load the dataframe
        df_csv = os.path.join(loc_actual, jdic['df_csv'])
        df = pd.read_csv(df_csv, index_col=0)

        return cls(df=df, **pp)



#==============================================================================
#===================== TESTING ================================================
#==============================================================================


def fibo(nrow=3, ncol=4):
    """some numbers in an array"""
    N = ncol   # cols
    M = nrow   # rows
    fibo = np.asarray([1]*N*M)
    for i in range(2, N*M):
        fibo[i] = fibo[i-1] + fibo[i-2]
    data = fibo[:M*N].reshape(N, M).T
    return data

def states(nrow=3, ncol=4, states=None):
    """make an array with classification states (discrete, strings)"""
    if states is None:
        states = ['duck', 'stoat', 'herring']

    return np.random.choice(states, size=(nrow,ncol))


def test_scoreblock_stack():
    """test scoreblock stacking"""

    tagDict = dict(name='gallahad', quest='grail', color='blue')

    N = 6
    M = 4
    data = fibo(nrow=M, ncol=N)

    # ScoreBlock 1
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb1 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)

    # ScoreBlock 2 (has a different index column name)
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'LETTER']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb2 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)

    # ScoreBlock 3 (has a different data_cols names relative to 1)
    data_cols = ['dcXX-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb3 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)

    # ScoreBlock 4 (has one fewer data_cols than 1)
    data = fibo(nrow=M, ncol=N-1)
    data_cols = ['dc-%4.4i' % n for n in range(N-1)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb4 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)


    # stack and rename index columns
    stk = sb1.stack(others=[sb2], force_data=True, rename_index_cols={'LETTER':'letter'})
    assert stk.df['letter'].tolist() == ['a','b','c','d']*2, 'should be [a,b,c,d,a,b,c,d]'

    # stack and force_data
    stk = sb1.stack(others=[sb3], force_data=True)
    assert stk.data_cols == sb1.data_cols,  'should be %s' % (str(stk.data_cols))

    # stack ragged data (should pad last data col with nan)
    stk = sb1.stack(others=[sb4], force_data=False)
    assert np.isnan(stk.df['dc-0005'].values[-1]), 'should be nan'
    assert np.isnan(stk.df['dc-0005'].values[-4]), 'should be nan'


    # stack ragged data and replace nan
    stk = sb1.stack(others=[sb4], force_data=False, data_nan=-1)
    assert stk.df['dc-0005'].values[-1] == -1, 'should be nan'
    assert stk.df['dc-0005'].values[-4] == -1, 'should be nan'


def test_scoreblock_consensus():
    """testing consensus"""

    tagDict = dict(name='gallahad', quest='grail', color='blue')
    N = 5
    data = [
        ['duck',  'herring', 'duck', 'stoat',   'herring'],
        ['duck',  'stoat',   'duck', 'stoat',   'herring'],
        ['stoat', 'duck',    'duck', 'herring', 'herring'],
        ['stoat', 'duck',    'duck', 'duck',    'herring'],
    ]

    # ScoreBlock 1
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'a', 'b','b'], ['camelot', 'camelot', 'camelot', 'camelot'])
    index_cols = ['number', 'letter', 'place']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb1 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols)

    cc = sb1.consensus(index_fill='ConSEnSuS', data_fill='LOL')

    assert cc.df_index.iloc[-1].tolist() == ['ConSEnSuS', 'ConSEnSuS', 'camelot']
    assert cc.data[-1].tolist() == ['LOL', 'LOL', 'duck', 'LOL', 'herring']


def test_scoreblock_applymap():
    """test applymap (map data values via dictionary)"""

    mapp = dict(duck='DUCK')

    tagDict = dict(name='gallahad', quest='grail', color='blue')
    # some categorical data (N=6 columns)
    N = 4
    data = [
        ['duck', 'herring', 'stoat', 'stoat'],
        ['duck', 'stoat', 'duck', 'stoat'],
        ['stoat', 'duck', 'duck', 'herring'],
    ]

    # ScoreBlock 1
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2], ['a', 'b', 'c'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb1 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols)

    sb2 = sb1.applymap(mapp)
    assert sb2.data_cols[0] == 'dc-0000', 'should be dc-0000'
    assert sb2.data_cols[-1] == 'dc-0003', 'should be dc-0005'
    assert sb2.tagDict['color'] == 'blue', 'should be blue'
    assert sb2.index_cols[0] == 'number', 'should be number'
    assert sb2.df['dc-0001'].tolist() == ['herring', 'stoat', 'DUCK']
    assert sb2.df['dc-0002'].tolist() == ['stoat', 'DUCK', 'DUCK']



def test_scoreblock_json():
    """test scoreblock json export/import"""

    loc = 'scratch'
    os.makedirs(loc, exist_ok=True)

    tagDict = dict(name='gallahad', quest='grail', color='blue')

    # some categorical data (N=6 columns)
    N = 6
    data = [
        ['duck', 'herring', 'stoat', 'stoat', 'stoat', 'herring'],
        ['duck', 'stoat', 'duck', 'stoat', 'stoat', 'herring'],
        ['stoat', 'duck', 'duck', 'herring', 'stoat', 'herring'],
        ['herring', 'herring', 'stoat', 'stoat', 'stoat', 'herring']
    ]

    # ScoreBlock 1
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb1 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)

    # dump
    jf = os.path.join(loc, 'test-scoreblock.json')
    sb1.to_json(f=jf)

    # reload
    sb2 = ScoreBlock.from_json(jf)

    # compare
    assert sb2.data_cols[0] == 'dc-0000', 'should be dc-0000'
    assert sb2.data_cols[-1] == 'dc-0005', 'should be dc-0005'
    assert sb2.tagDict['color'] == 'blue', 'should be blue'
    assert sb2.index_cols[0] == 'number', 'should be number'
    assert sb2.df['dc-0001'].tolist() == ['herring', 'stoat', 'duck', 'herring']



def test_scoreblock_count():
    """test scoreblock counting (and masking)"""

    tagDict = dict(name='gallahad', quest='grail', color='blue')

    # some categorical data (N=6 columns)
    N = 6
    data = [
        ['duck', 'herring', 'stoat', 'stoat', 'stoat', 'herring'],
        ['duck', 'stoat', 'duck', 'stoat', 'stoat', 'herring'],
        ['stoat', 'duck', 'duck', 'herring', 'stoat', 'herring'],
        ['herring', 'herring', 'stoat', 'stoat', 'stoat', 'herring']
    ]

    # ScoreBlock 1
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb1 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)

    # make masks and do counts
    maskAM = slice(0, N//2)
    maskPM = slice(N//2, N)

    sb_counts_all = sb1.mask(maskname='24h').count(frac=True)
    sb_counts_am = sb1.mask(mask=maskAM, maskname='12hAM').count(frac=True)
    sb_counts_pm = sb1.mask(mask=maskPM, maskname='12hPM').count(frac=True)


    assert sb_counts_all.data_cols == ['duck', 'herring', 'stoat']
    assert sb_counts_am.data_cols == ['duck', 'herring', 'stoat']
    assert sb_counts_pm.data_cols == ['herring', 'stoat']

    # stack the counts results
    stk = sb_counts_all.stack(others=[sb_counts_am, sb_counts_pm])
    assert np.isnan(stk.df['duck'].values[-1]), 'should be nan'

    # stack and fix NaN values
    stk = sb_counts_all.stack(others=[sb_counts_am, sb_counts_pm], data_nan=-2)
    assert stk.df['duck'].values[-1] == -2, 'should be -2'


if __name__ == '__main__':

    test_scoreblock_count()
    test_scoreblock_consensus()
    #test_scoreblock_keeprows()
    test_scoreblock_applymap()
    test_scoreblock_json()
    test_scoreblock_stack()




