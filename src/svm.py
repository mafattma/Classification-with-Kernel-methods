# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:45:29 2019

@author: fatma
"""

from itertools import product
from cvxopt import matrix,solvers
import numpy as np
import pandas as pd
from kernel import kernel, kernel_gram
from KFold import KFold
from tqdm import tqdm
from mismatchkernel import *

class svm:
    def __init__(self, kernel='rbf', ld=0.1, sigma=1, k=3, d=1, e=11, beta=0.1, Nystrom=False, m=100, alphabet='ACGT', subA=4, mis=2):
        self.X_train=None 
        self.kernel = kernel #string specifying kernel used
        self.sigma = sigma #parameter for rbf kernel
        self.k = k #k-spectrum
        self.d = d #affine gap
        self.e = e #affine gap
        self.beta = beta #beta for LA
        self.Nystrom = Nystrom #approximation of gram matrix
        self.m = m #m for nystrom
        self.alphabet = alphabet #alphabet for mismatch
        self.subA = subA #substring length for mismatch
        self.mis = mis #mismatch tolerance
        self.A = None
        self.neighbours = None
        if self.kernel in ["mismatch","MKL"]:
            print("Preparing mismatch kernel in progress...")
            self.A = create_vocab(self.alphabet, self.subA)[0]
            self.neighbours = compute_neighbours(self.A, self.mis)
            print("Finished preparing mismatch kernel")
    
        self.alpha = None #alphas used to compute the prediction function f
        self.ld = ld #lambda parameter in langragian formulation
        self.sv = None #support vectors
        
        

    def fit(self,X_train,y_train):
        """
        X_train,y_train: dataframe
        """
        X_train = X_train.as_matrix()
        self.X_train = X_train
        y_train = y_train.as_matrix()
        
        #step1: compute gram matrix
        K = kernel_gram(self.kernel, X_train, self.sigma, self.k, self.d, self.e, self.beta, self.Nystrom, self.subA, self.A, self.neighbours) #returns Gram matrix of the specific kernel Ki,j=K(xi,xj) external or internal function?
        
        #step2: solve optimization problem
        n = X_train.shape[0]
        y_diag=np.diag(y_train)
        P = matrix(K,tc='d') 
        q = matrix(-y_train,tc='d') 
        G = matrix(np.vstack((-y_diag,y_diag)),tc='d') #-alphaiyi<0 and alphaiyi<1/2nlambda
        h = matrix(np.vstack((np.zeros((n,1)),np.ones((n,1))/(2*self.ld*n))),tc='d') #h=[0..01/2nlambda...1/2nlambda]
        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h)
        alpha = np.array(sol['x']).reshape((n,1))
        self.alpha = alpha
        
        #step3: set support vectors
        i, = np.where(np.abs(self.alpha[:,0]) > 1e-5)
        self.sv = self.X_train [i,]
        self.alpha = self.alpha [i,]

        
    def predict(self,X_test):
        X_test = X_test.as_matrix()
        n = np.shape(X_test)[0]
        
        y_pred = np.zeros((n,1))
        for i in tqdm(range(np.shape(X_test)[0])):
            x = X_test[i,]
            Kx = np.asarray([kernel(self.kernel, xi, x, self.sigma, self.k, self.d, self.e, self.beta, self.subA, self.A, self.neighbours) for xi in self.sv])
            y_pred[i,] = np.dot(Kx,self.alpha)
        
        y_pred = np.sign(y_pred).astype('int')
        y_pred_list = map(lambda x: x[0], y_pred)
        return(pd.Series(y_pred_list))
    
    def grid_search(self,params,X,y,cv=3):
        """
        params:dictionary with key: name of parameter, values type is list (even if it's one element)
        names should be kernel, ld, sigma, k, d, e, beta
        returns best score with corresponding parameters kernel, ld, sigma, k, d, e, beta, subA, mis
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
                self.ld=param[params_names.index('ld')]
            except ValueError:
                pass
            
            try:
                self.sigma=param[params_names.index('sigma')]
            except ValueError:
                pass
            
            try:
                self.k=param[params_names.index('k')]
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
            
            try:
                self.subA=int(param[params_names.index('subA')])
            except ValueError:
                pass
            
            try:
                self.mis=param[params_names.index('mis')]
            except ValueError:
                pass
            
            if self.kernel == "mismatch":
                self.A = create_vocab(self.alphabet, self.subA)[0]
                self.neighbours = compute_neighbours(self.A, self.mis)
            
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
            results['ld']=params_df['ld'][best_score_arg]
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
        
        try:
            results['subA']=params_df['subA'][best_score_arg]
        except KeyError:
            pass
        
        try:
            results['mis']=params_df['mis'][best_score_arg]
        except KeyError:
            pass
        
        return(results,params_df)
        
        