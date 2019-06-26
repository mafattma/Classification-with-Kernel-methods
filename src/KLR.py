# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:05:39 2019

@author: fatma
"""

from itertools import product
import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
from kernel import kernel, kernel_gram
from KFold import KFold
from tqdm import tqdm


def fsigma(u):
    try:
        result = 1/(1+math.exp(-u))
    except OverflowError:
        result = 0
    return result

def fcost(n,K,alpha,z,W,ld):
    return(1/n*np.dot(np.dot((np.dot(K,alpha)-z).T,W),(np.dot(K,alpha)-z))+ld*np.dot(np.dot(alpha.T,K),alpha))

class KLR:
    def __init__(self, kernel = 'rbf', ld = 0.1, sigma = 1, k = 3, d = 1, e = 11, beta = 0.1, Nystrom = False, m = 100):
        self.X_train = None 
        self.kernel = kernel #string specifying kernel used
        self.ld = ld #lambda parameter in langragian formulation
        self.sigma = sigma #parameter for rbf kernel
        self.k = k #k-spectrum
        self.d = d #affine gap
        self.e = e #affine gap
        self.beta = beta #beta for LA
        self.Nystrom = Nystrom #approximation of gram matrix
        self.m = m #m for nystrom
        self.alpha = None #alphas used to compute the prediction function f
        self.Niter = 0
    
    def fit(self, X_train, y_train):
        """
        X_train,y_train: dataframe
        """
        X_train = X_train.as_matrix()
        self.X_train = X_train
        n = X_train.shape[0]
        y_train = y_train.as_matrix().reshape((n,1))
        
        #step1: compute gram matrix
        K = kernel_gram(self.kernel, X_train, self.sigma, self.k, self.d, self.e, self.beta, self.Nystrom, self.m) #returns Gram matrix of the specific kernel Ki,j=K(xi,xj) external or internal function?
        
        #step2: solve optimization problem
        alpha = np.zeros((n,1)) #initialization acc=0.4825
        vsigma = np.vectorize(fsigma) #useful to apply sigma to arrays
        W = np.zeros((n,n))
        epsilon = 1e-6
        i = 0
        error = 1
        while error>epsilon:
            m = np.dot(K,alpha)
            np.fill_diagonal(W,vsigma(m)*vsigma(-m))
            z = m + y_train / vsigma(-y_train*m)
            alpha = np.sqrt(W).dot(inv(np.sqrt(W).dot(K.dot(np.sqrt(W)))+n*self.ld*np.eye(n)).dot(np.sqrt(W).dot(z))) #cf slides WKRR
            if i == 0:
                error = 1
                J_t_1 = fcost(n,K,alpha,z,W,self.ld)
                i += 1
            else:
                error = np.abs(fcost(n,K,alpha,z,W,self.ld)-J_t_1)/J_t_1
                J_t_1 = fcost(n,K,alpha,z,W,self.ld)
                i += 1
        
        self.alpha = alpha
        self.Niter = i 
        
    def predict(self,X_test):
        X_test = X_test.as_matrix()
        n = np.shape(X_test)[0]
        y_pred = np.zeros((n,1))
        for i in tqdm(range(n)):
            x = X_test[i,]
            Kx = np.asarray([kernel(self.kernel, xi, x, self.sigma, self.k, self.d, self.e, self.beta) for xi in self.X_train])
            y_pred[i,] = np.dot(Kx,self.alpha)
        y_pred = np.sign(y_pred).astype('int')
        y_pred_list = map(lambda x: x[0], y_pred)
        return(pd.Series(y_pred_list))
    
    
    def grid_search(self,params,X,y,cv=3):
        """
        params:dictionary with key: name of parameter, values type is list (even if it's one element)
        names should be kernel, ld, sigma, k, Niter
        returns best score with corresponding parameters kernel, ld, sigma, Niter
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
        
        return(results,params_df)

        
        