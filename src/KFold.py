# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 21:45:29 2019

@author: axel
"""
import numpy as np
import pandas as pd
import random

class KFold:
    def __init__(self,n_splits,shuffle=True,random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self,X): 
        if(self.shuffle) :
            random.seed(self.random_state)
            liste_indice = np.array_split( random.sample( range(X.shape[0] ) , X.shape[0])   , self.n_splits)
        else :
            liste_indice = np.array_split(  range(X.shape[0]  , X.shape[0])   , self.n_splits)
        return( [ (  np.concatenate(  liste_indice[:i]+liste_indice[i+1:] ) , liste_indice[i] )  for i in range(self.n_splits) ]  )
        
           