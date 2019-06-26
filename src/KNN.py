# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:03:42 2019

@author: fatma
"""
import numpy as np
import pandas as pd
from kernel import kernel
from tqdm import tqdm
from KFold import KFold
from itertools import product


class kknn:
    
    def __init__(self, kernel = 'rbf', sigma = 1, kn = 3, k = 3, d = 1, e = 11, beta = 0.1):
        self.X_train = None 
        self.y_train = None
        self.kernel = kernel #string specifying kernel used
        self.sigma = sigma #parameter for rbf kernel
        self.kn = kn #knn
        self.k = k #kspectrum
        self.d = d #affine gap
        self.e = e #affine gap
        self.beta = beta #beta for LA
    
    def fit(self, X_train, y_train):
        self.X_train = X_train.as_matrix()
        self.y_train = y_train.as_matrix()
        
    def predict(self, X_test):
        X_test = X_test.as_matrix()
        n = np.shape(X_test)[0]
        y_pred = np.zeros((n,1))
        
        for i in tqdm(range(n)):
            x = X_test[i,]
            #dictionary of distances between x and all x_train
            x_dist = dict() 
            #building dict
            for j in range(np.shape(self.X_train)[0]):
                #computing distance^2
                x_dist[j] = kernel(self.kernel, x, x, self.sigma, self.k, self.d, self.e, self.beta) \
                            - 2 * kernel(self.kernel, self.X_train [i,], x, self.sigma, self.k, self.d, self.e, self.beta) \
                            + kernel(self.kernel, self.X_train [i,], self.X_train [i,], self.sigma, self.k, self.d, self.e, self.beta)
                
            #sorting dict by value and getting k nearest neighbors 
            x_k = sorted((value, key) for (key,value) in x_dist.items())[:self.kn] #returns a list of length k
            #getting indices of k nearest neighbors
            x_k_ind = [z[1] for z in x_k]
            #getting labels of k nearest neighbors
            y_k = self.y_train[x_k_ind,]
            #predicting label of x(i)
            y_pred[i,] = np.argmax(pd.Series(y_k).value_counts())
        
        y_pred_list = map(lambda x: x[0], y_pred)
        
        return(pd.Series(y_pred_list))
    
    def grid_search(self,params,X,y,cv=3):
        """
        params:dictionary with key: name of parameter, values type is list (even if it's one element)
        names should be kernel, ld, sigma, k
        returns best score with corresponding parameters kernel, ld, sigma, k
        """
        params_df=pd.DataFrame([row for row in product(*params.values())], 
                           columns=params.keys())
        
        def cv_(param):
            params_names=list(params_df.columns.values)
            try:
                self.kernel=param[params_names.index('kernel')] #looks for index of kernel in df
            except ValueError:
                pass
            
            try:
                self.sigma=param[params_names.index('sigma')]
            except ValueError:
                pass
            
            try:
                self.kn=param[params_names.index('kn')]
            except ValueError:
                pass
            
            try:
                self.d=param[params_names.index('d')]
            except ValueError:
                pass
            
            try:
                self.e=param[params_names.index('e')]
            except ValueError:
                pass
            
            try:
                self.beta=param[params_names.index('beta')]
            except ValueError:
                pass
            
            X_reset=X.reset_index(drop=True)
            y_reset=y.reset_index(drop=True) #needed to compute accuracy
            
            kf = KFold(n_splits=cv, shuffle=True, random_state=42) 
            score=[]
            for train_index, val_index in kf.split(X_reset):
                X_train, X_val = X_reset.iloc[train_index], X_reset.iloc[val_index] 
                y_train, y_val = y_reset.iloc[train_index], y_reset.iloc[val_index]
                self.fit(X_train,y_train)
                y_pred=self.predict(X_val)
                score.append(np.mean(y_pred==y_val.reset_index(drop=True)))
            return(np.mean(score),np.var(score))
                     
        scores = params_df.apply(cv_,axis=1)
        params_df['score_mean'] = scores.apply(lambda x: x[0])
        params_df['score_var'] = scores.apply(lambda x: x[1])
                 
        best_score_arg=np.argmax(params_df['score_mean'])
        results={}
        results['top_score']=np.max(params_df['score_mean'])
        
        try:
            results['kernel']=params_df['kernel'][best_score_arg]
        except KeyError:
            pass
        
        try:
            results['sigma']=params_df['sigma'][best_score_arg]
        except KeyError:
            pass
        
        try:
            results['k']=params_df['k'][best_score_arg]
        except KeyError:
            pass
        
        try:
            results['d']=params_df['d'][best_score_arg]
        except KeyError:
            pass
        
        try:
            results['e']=params_df['e'][best_score_arg]
        except KeyError:
            pass
        
        try:
            results['beta']=params_df['beta'][best_score_arg]
        except KeyError:
            pass
        
        return(results,params_df)