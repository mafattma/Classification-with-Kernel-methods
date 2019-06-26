# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:50:30 2019

@author: fatma
"""
import numpy as np
from itertools import product
from LAkernel import LAkernel
import numpy.linalg as la
from math import *
from power_iteration import power_iteration
from tqdm import tqdm
from mismatchkernel import mismatch_kernel

def kernel(method, xi, xj, sigma=1, k=3, d=1, e=11, beta=0.1, subA=4, A=None, neighbours=None): 
    """
    sigma : float rbf
    k : k-spectrum
    d,e,beta: LA
    A, subA, neighbours: (mismatch kernel) substring alphabet , substring length and neighbours 
    """
    if method == "linear":
        return np.dot(xi.T,xj)
    elif method == "quad":
        return np.dot(xi.T,xj)**2
    elif method == "rbf":
        return np.exp(-np.dot((xi-xj).T,(xi-xj))/(2*(sigma**2)))
    elif method == "spectral":
        alphabet_comb=[''.join(x) for x in list(product(set(xi+xj), repeat=k))]
        phi_xi=np.array([xi.count(c) for c in alphabet_comb])
        phi_xj=np.array([xj.count(c) for c in alphabet_comb])
        return np.dot(phi_xi.T,phi_xj)
    elif method == "LA":
        return(LAkernel(xi, xj, d, e, beta))
    elif method == "mismatch":
        return(mismatch_kernel(xi, xj, subA, A, neighbours))
    elif method == "MKL":
        return( kernel(method="spectral", xi=xi, xj=xj, k=k) + kernel(method="mismatch", xi=xi, xj=xj, subA=subA, A=A, neighbours=neighbours))


def kernel_gram(method, X, sigma=1, k=3, d=1, e=11, beta=0.1, Nystrom=False, subA=4, A=None, neighbours=None):
    """
    computes Gram Matrix
    with Nystrom approximation if True
    sigma : float rbf
    k : to k-spectrum
    d,e,beta: LA
    A, subA, neighbours: (mismatch kernel) substring alphabet , substring length and neighbours 
    """
    n=X.shape[0]
    
    if Nystrom == False: 
        K=np.zeros((n,n))
        for i in tqdm(range(n)): 
            K[i,i] = kernel(method, X[i,] ,X[i,], sigma, k, d, e, beta, subA, A, neighbours)
            for j in range(i):
                K[i,j] = K[j,i] = kernel(method, X[i,] ,X[j,], sigma, k, d, e, beta, subA, A, neighbours)
    else:
        #randomly generating m indices
        m = 100 
        np.random.seed(42)
        ind=np.random.choice(n,size=m, replace=False) #without replacement
        Xm=X[ind,]
        #defining Kmm
        Kmm=np.zeros((m,m))
        for i in range(m):
            Kmm[i,i] = kernel(method, Xm[i,] ,Xm[i,], sigma, k, d, e, beta, subA, A, neighbours)
            for j in range(i):
                Kmm[i,j] = Kmm[j,i] = kernel(method, Xm[i,] ,Xm[j,], sigma, k, d, e, beta, subA, A, neighbours)

        #defining Knm and Kmn 
        Knm=np.zeros((n,m))
        Kmn=np.zeros((m,n))
        for i in range(n):
            for j in range(m):
                Knm[i,j] = Kmn[j,i] = kernel(method, X[i,], Xm[j,], sigma, k, d, e, beta, subA, A, neighbours)
        #defining K~
        K = np.dot(Knm, np.dot(la.inv(Kmm),Kmn))

    if method == "LA":
        eig = -power_iteration(-K)[0]
        if eig > 0:
            min_neg_eig = 0
        else:
            min_neg_eig = eig
        K = K-min_neg_eig*np.eye(n) #LA-eig
     
    #normalization
    K_norm = np.zeros((n,n))
    for i in range (n):
        K_norm[i,i] = 1
        for j in range(i):
            K_norm[i,j] = K_norm[j,i] = K[i,j] / ( sqrt(K[i,i]) * sqrt(K[j,j]) )
    
    return(K_norm)
    
    
    

