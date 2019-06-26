# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:42:54 2019

@author: fatma
"""
from random import *
import numpy as np
import pandas as pd

def subsample(n,ratio):
    """
    Creates a random subsample indices from the dataset with replacement
    """
    sample = list()
    n_sample = round(n * ratio)
    while len(sample) < n_sample:
        sample.append(randrange(n))
    return(np.asarray(sample))
 
def bagging_predict(models, X):
    """
    Makes a prediction with a list of models
    """
    y_pred_res = models[0].predict(X)
    for model in models[1:]:
        y_pred = model.predict(X)
        y_pred_res = pd.concat([y_pred_res,y_pred], axis=1)
    
    y_bag = list()
    for i in range(len(X)):
        y_bag.append(np.argmax(y_pred_res.iloc[i].value_counts()))
    return(pd.Series(y_bag))
 
 
def bagging(X_train, y_train, X_test, sample_size, n_models, model):
    """
    Bootstrap Aggregation Algorithm
    """
    models = list()
    for i in range(n_models):
        sample_ix = subsample(len(X_train), sample_size)
        sample_x = X_train.iloc[sample_ix]
        sample_y = y_train.iloc[sample_ix]
        model.fit(sample_x, sample_y)
        models.append(model)
    predictions = bagging_predict(models,X_test)
    return(predictions)
