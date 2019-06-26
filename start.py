# -*- coding: utf-8 -*-
"""
Created on Mon Apr 01 08:39:03 2019

@author: fatma
"""

import pandas as pd
import numpy as np
from svm import svm

"""
Importing data
"""
X_tr_0=pd.read_csv('Xtr0.csv')
X_tr_1=pd.read_csv('Xtr1.csv')
X_tr_2=pd.read_csv('Xtr2.csv')

y_tr_0=pd.read_csv('Ytr0.csv')
y_tr_1=pd.read_csv('Ytr1.csv')
y_tr_2=pd.read_csv('Ytr2.csv')

y_tr_0['Bound']=y_tr_0['Bound'].apply(lambda x: -1 if x==0 else x)
y_tr_1['Bound']=y_tr_1['Bound'].apply(lambda x: -1 if x==0 else x)
y_tr_2['Bound']=y_tr_2['Bound'].apply(lambda x: -1 if x==0 else x)

X_te_0=pd.read_csv('Xte0.csv')
X_te_1=pd.read_csv('Xte1.csv')
X_te_2=pd.read_csv('Xte2.csv')

"""
Training the model
"""
svm_mismatch = svm(kernel="mismatch", ld=1e-3, subA=8, mis=1)

##TF0
svm_mismatch.fit(X_tr_0['seq'],y_tr_0['Bound'])
y_te_0_mis = svm_mismatch.predict(X_te_0['seq'])
y_te_0_mis_df=pd.DataFrame(y_te_0_mis, columns=['Bound'])
y_te_0_mis_df.insert(loc=0,column='Id',value=X_tr_0['Id'])

##TF1
svm_mismatch.fit(X_tr_1['seq'],y_tr_1['Bound'])
y_te_1_mis = svm_mismatch.predict(X_te_1['seq'])
y_te_1_mis_df=pd.DataFrame(y_te_1_mis, columns=['Bound'])
y_te_1_mis_df.insert(loc=0,column='Id',value=X_tr_1['Id']-1000)

##TF2
svm_mismatch.fit(X_tr_2['seq'],y_tr_2['Bound'])
y_te_2_mis = svm_mismatch.predict(X_te_2['seq'])
y_te_2_mis_df=pd.DataFrame(y_te_2_mis, columns=['Bound'])
y_te_2_mis_df.insert(loc=0,column='Id',value=X_tr_1['Id'])

"""
Submission
"""
y_te_mis_df = pd.concat([y_te_0_mis_df,y_te_1_mis_df,y_te_2_mis_df])
y_te_mis_df['Bound'][y_te_mis_df['Bound']==-1] = 0
y_te_mis_df.to_csv('Yte.csv',index=False)
