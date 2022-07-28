import sys,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict,GridSearchCV,cross_val_score,RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import copy 
import re, string,sys
import matplotlib
from numpy import mean
from numpy import std
from collections import Counter

def ProcessCLI(args):
    
    studies={'Oluf':['Dag0','Dag4','Dag8','Dag42','Dag180'],
             'Raymond':['CTLD0','CTLD7','CTLD90','EXPD0','EXPD7','EXPD90']}
    
    applylasso(args[1],args[2],studies)
    
def applylasso(data,dfname,studies):
        
    model=LogisticRegressionCV(max_iter=10000,cv=5,penalty='l1',solver='liblinear')
    
    for s in studies.keys():
        
        X_train,y_train,X_test,y_test=selectdata(data,studies[s])
        scaler = StandardScaler()
        
        dftidx=dfidx(X_test)
    
        X_train=scaler.fit_transform(X_train.T).T
        X_test=scaler.fit_transform(X_test.T).T
        
        logcv=model.fit(X_train,y_train)
        
        y_score=logcv.predict_proba(X_test)
        
        
        print(s, ' Accuracy: ',logcv.score(X_test,y_test))
        
        dftidx['score']=y_score[:,1]
        dftidx.to_csv("%s-scores"%s,index=False,header=True, sep=' ')
        
    
def dfidx(df):
    df=df.reset_index()
    df=df[df.columns[[0,1,2,3]]]
    return df

def selectdata(data,lst):
    
    df=pd.read_csv(data, header=0, sep=" ")
    
    test=df[df["cluster"].isin(lst)]
    train=df[~df["cluster"].isin(lst)]
    
    y_test=test["status"]
    y_train=train["status"]
    
    X_train=train.set_index(['Pathway','cluster','Group','status'])
    X_test=test.set_index(['Pathway','cluster','Group','status'])
    
    return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    ProcessCLI(sys.argv)
