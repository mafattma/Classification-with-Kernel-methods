# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:12:32 2019

@author: fatma
"""
import numpy as np
from math import *
from blosum62 import blosum62_matrix

def LAkernel(x, y, d, e, beta):
    """
    x, y are strings
    d,e: int for affine gap penalty g
    beta: float param
    """
    s = blosum62_matrix()
    n_row=len(x)+1
    n_col=len(y)+1
    M=np.zeros((n_row,n_col))
    X=np.zeros((n_row,n_col))
    Y=np.zeros((n_row,n_col))
    X2=np.zeros((n_row,n_col))
    Y2=np.zeros((n_row,n_col))
    
    for i in range(1,n_row):
        for j in range(1,n_col):
            M[i,j]=exp(beta*int(s[x[i-1]][y[j-1]]))*(1+X[i-1,j-1]+Y[i-1,j-1]+M[i-1,j-1])
            X[i,j]=exp(beta*d)*M[i-1,j]+exp(beta*e)*X[i-1,j]
            Y[i,j]=exp(beta*d)*(M[i,j-1]+X[i,j-1])+exp(beta*e)*Y[i,j-1]
            X2[i,j]=M[i-1,j]+X2[i-1,j]
            Y2[i,j]=M[i,j-1]+X2[i,j-1]+Y2[i,j-1]
    return(1/beta*log(1+X2[-1,-1]+Y2[-1,-1]+M[-1,-1])) #K~
    