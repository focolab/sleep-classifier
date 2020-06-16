#!/usr/bin/env python3
"""
    
    model tools for mouse EEG/EMG analysis and sleep state classification

    Classes

    Functions
        chop8640: chop 8640 epochs (24h) into chunks for training (bool mask)

"""

# import os
# import argparse
# import json
# #import warnings
# import pdb
# import pickle
# import datetime

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use("tkAgg")
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import seaborn as sns


def chop8640(num_total=8640, num_keep=8640, num_chunks=10, center=True):
    """chop 8640 10s epochs (24h) into chunks
    
    num_total: total number of points (epochs)
    num_keep: how many to keep
    num_chunks: split num_keep into this many chunks
    
    NOTE: 360 epochs/hour, (using 10s epochs)
    NOTE: 1440 minutes/day
    """

    if num_keep%num_chunks != 0:
        print('num_total : ', num_total)
        print('num_keep  : ', num_keep)
        print('num_chunks: ', num_chunks)
        raise Exception('num_keep/num_chunks must be an integer')
    if num_total%num_chunks != 0:
        print('num_total : ', num_total)
        print('num_keep  : ', num_keep)
        print('num_chunks: ', num_chunks)
        raise Exception('num_total/num_chunks must be an integer')


    num_drop = num_total - num_keep
    
    # build a template list that gets repated num_chunks times
    u_keep = num_keep/num_chunks
    u_drop = num_drop/num_chunks
    unit = [True]*int(u_keep) + [False]*int(u_drop)
    vec = np.asarray(unit*num_chunks)
    
    if center is True:
        vec = np.roll(vec, int(u_drop/2) )

    return vec.tolist()


class ClassifierBundle(object):
    """a bundle of trained classifiers

    *classifiers trained on the same data*
    
    used to store trained models and to make new predictions
    """

    def __init__(self, sc=None, pca=None, classifiers={}, training_params={}):
        """"""
        self.sc = sc
        self.pca = pca
        self.classifiers = classifiers
        self.training_params = training_params

    @property
    def classifier_names(self):
        return list(self.classifiers.keys())

    def about(self):
        """"""
        print('-------- ClassifierBundle --------')
        print('classifier_names:', self.classifier_names)
        print('training_params :')
        ppd(self.training_params)
        return

    def predict(self, X=None, ):
        """given X, predict y

        and, if appropriate, standardize (sc) and pca transform X

        returns a dataframe and data_cols (data columns of df)
        """

        # apply sc
        # apply pca
        # loop over classifiers

        Xm = X*1.0
        if self.sc is not None:
            Xm = self.sc.transform(Xm)
        if self.pca is not None:
            Xm = self.pca.transform(Xm)

        data = []
        for nameC in self.classifier_names:
            classifier = self.classifiers[nameC]['cls']
            y = classifier.predict(Xm)
            #print(nameT, nameM, nameC)
            data.append(y)
        data = np.asarray(data)
        cols_data = ['dcol-%5.5i' % i for i in range(data.shape[1])]
        df_data = pd.DataFrame(data=data, columns=cols_data)

        df_data['classifier_name'] = self.classifier_names

        df_data = df_data[['classifier_name']+cols_data]
        return df_data, cols_data

