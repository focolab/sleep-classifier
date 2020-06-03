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
    applymap():
        apply a map (dict) to categorical data, (good for re-naming/merging
        states)
    keeprows():
        keep a subset of rows, based on conditions
    to_json():
        dump dataframe to csv and everything else to json
    from_json():
        load from a previously dumped json
    
    consensus():
        determine (binary) consensus for each column of df (index and data)

    TODO: confusion matrix method

    """
    def __init__(self, loc=None, df=None, index_cols=None, data_cols=None, tagDict={},
                 ancestry={}):
        """ """
        self.loc = loc
        self.df = df
        self.tagDict = tagDict

        self.ancestry = ancestry

        if isinstance(self.df, pd.Series):
            self.df = self.df.to_frame().T

        if index_cols is None:
            raise Exception('index_cols required')
        else:
            self.index_cols = index_cols

        if data_cols is None:
            self.data_cols = [c for c in self.df.columns if c not in self.index_cols]
        else:
            self.data_cols = data_cols


    @property
    def data(self):
        return self.df[self.data_cols].values

    @property
    def df_index(self):
        return self.df[self.index_cols].copy()

    @property
    def dfmi(self):
        """return a mutli-index dataframe"""
        return self.df.set_index(self.index_cols)

    @property
    def uniqueScores(self):
        return np.unique(self.data).tolist()

    @property
    def numrows(self):
        return self.df.shape[0]

    @property
    def numdatacols(self):
        return self.df.shape[1]

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


    def add_const_index_col(self, name='newcolumn', value=None, inplace=False):
        """add an index column with a constant value"""
        df_out = self.df.copy()
        df_out[name] = [value]*len(df_out)

        index_cols = self.index_cols+[name]
        df_out = df_out[index_cols+self.data_cols]

        if inplace:
            self.df = df_out
            self.index_cols = index_cols
        else:
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


    def keeprows_iloc(self, iloc=None):
        """"""
        out = ScoreBlock(
            df=self.df.iloc[iloc],
            index_cols=self.df_index.columns.tolist(),
            tagDict=self.tagDict
            )
        return out

    def keeprows(self, conditions=[], comparison='all'):
        """keep rows from a the dataframe, subject to conditions

        declarative method for row selection based off the index dataframe

        a tangled up mess, but it works for a few cases

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
            ser = pd.Series(dict(zip(self.df.columns.values, ccc)))
            dfc = pd.concat([self.df, ser.to_frame().T]).reset_index(drop=True)
            cc = ScoreBlock(
                df=dfc,
                index_cols=self.copy_index_cols(),
                tagDict=self.tagDict,
                ancestry=self.ancestry,                
                )

        return cc


    def to_sirenia_txt(self, f='scores.txt', row=0, str2num=None):
        """dump a row of scores to Sirenia formatted csv (.txt)"""
        df = self.to_sirenia_df(row=row, str2num=str2num)
        df.to_csv(f)

    def to_sirenia_df(self, row=0, str2num=None):
        """convert a row of scores to Sirenia formatted DataFrame

        Epoch #,Start Time,End Time,Score #, Score
        1,10/15/2018 09:00:00,10/15/2018 09:00:10,1,Wake
        2,10/15/2018 09:00:10,10/15/2018 09:00:20,1,Wake
        8623,10/16/2018 08:57:00,10/16/2018 08:57:10,2,Non REM
        8627,10/16/2018 08:57:40,10/16/2018 08:57:50,1,Wake
        """
        if str2num is None:
            str2num = {}
        num2str = {v:k for k,v in str2num.items()}

        from datetime import datetime, timedelta
        startdate = "2019-01-02"
        starttime = "09:00:00"
        epoch_len = 10
        scores = self.data[row]
        num_epochs = len(scores)

        # date time strings
        dt = datetime.fromisoformat('%sT%s' % (startdate, starttime))
        epoch_indices = range(1,len(scores)+1)
        times = [dt + timedelta(seconds=epoch_len*i) for i in range(num_epochs+1)]
        ta = [xx.strftime('%d/%m/%Y %X') for xx in times[:-1]]
        tb = [xx.strftime('%d/%m/%Y %X') for xx in times[1:]]

        # we need scores in string and number (integer) format
        if isinstance(scores[0], str):
            scores_strs = scores
            scores_nums = [str2num[x] for x in scores]
        elif isinstance(scores[0], int):
            scores_nums = scores
            scores_strs = [num2str[x] for x in scores]
        else:
            raise Exception('wtf')

        data = zip(epoch_indices, ta, tb, scores_nums, scores_strs)
        cols=['Epoch #', 'Start Time', 'End Time', 'Score #', 'Score']
        df = pd.DataFrame(data=data, columns=cols).set_index('Epoch #', drop=True)

        return df


    def mask(self, mask=None, maskname=None, maskcolname='mask'):
        """select subset of columns and optionally mark the change in the index"""

        if mask is None:
            mask = slice(None)

        data = self.data[:, mask]
        data_cols = np.asarray(self.data_cols)[mask].tolist()
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



    def count(self, frac=False):
        """count occurances of states for each data row
        (mainly for categorical data)
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

def demo_block():
    """Scoreblock for testing, with categorical data (N=6 columns)"""
    tagDict = dict(name='gallahad', quest='grail', color='blue')
    N = 6
    data = [
        ['duck',    'herring', 'stoat', 'stoat',   'stoat', 'herring'],
        ['duck',    'stoat',   'duck',  'stoat',   'stoat', 'herring'],
        ['stoat',   'duck',    'duck',  'herring', 'stoat', 'herring'],
        ['herring', 'herring', 'stoat', 'stoat',   'stoat', 'herring']
    ]
    # ScoreBlock
    data_cols = ['dc-%4.4i' % n for n in range(N)]
    ndx = zip([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
    index_cols = ['number', 'letter']
    df1 = pd.DataFrame(data=ndx, columns=index_cols)
    df2 = pd.DataFrame(data=data, columns=data_cols)
    df = pd.concat([df1, df2], axis=1)
    sb1 = ScoreBlock(df=df, tagDict=tagDict, index_cols=index_cols, data_cols=data_cols)
    return sb1

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
    sb1 = demo_block()
    sb1.df['letter'] = ['a', 'a', 'a', 'a']     # manually make these all the same
    cc = sb1.consensus(index_fill='ConSEnSuS', data_fill='LOL')
    assert cc.df_index.iloc[-1].tolist() == ['ConSEnSuS', 'a']
    assert cc.data[-1].tolist() == ['LOL', 'LOL', 'LOL', 'LOL', 'stoat', 'herring']

def test_scoreblock_applymap():
    """test applymap (map data values via dictionary)"""
    sb1 = demo_block()
    sb2 = sb1.applymap(dict(duck='DUCK'))
    assert sb2.tagDict['color'] == 'blue', 'should be blue'
    assert sb2.index_cols[0] == 'number', 'should be number'
    assert sb2.df['dc-0001'].tolist() == ['herring', 'stoat', 'DUCK', 'herring']
    assert sb2.df['dc-0002'].tolist() == ['stoat', 'DUCK', 'DUCK', 'stoat']

def test_scoreblock_json():
    """test scoreblock json export/import"""
    loc = 'scratch'
    os.makedirs(loc, exist_ok=True)
    sb1 = demo_block()
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
    N = 6
    sb1 = demo_block()

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

def test_bool_mask():
    """test boolean mask"""
    sb1 = demo_block()

    # make mask and apply it
    mask00 = [True, True, True, False, True, False]
    sb_mask00 = sb1.mask(mask=mask00, maskname='mask00')

    assert sb_mask00.data_cols == ['dc-0000','dc-0001','dc-0002','dc-0004']

def test_sirenia():
    """test to_sirenia_df and to_sirenia_txt"""
    loc = 'scratch'
    os.makedirs(loc, exist_ok=True)
    sb1 = demo_block()

    # export score row 0 to sireia ready DataFrame
    str2num = dict(duck=1, herring=2, stoat=3)
    df = sb1.to_sirenia_df(str2num=str2num, row=0)
    assert df['Score'][:3].values.tolist() == ['duck', 'herring', 'stoat']
    assert df['Score #'].values.tolist() == [1, 2, 3, 3, 3, 2]

    # and dump to file
    sb1.to_sirenia_txt(
        f=os.path.join(loc,'scores_sirenia.txt'),
        str2num=str2num,
        row=0
        )


if __name__ == '__main__':
    test_sirenia()
    test_bool_mask()
    test_scoreblock_count()
    test_scoreblock_consensus()
    #test_scoreblock_keeprows()
    test_scoreblock_applymap()
    test_scoreblock_json()
    test_scoreblock_stack()

