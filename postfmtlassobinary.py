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
    
    applylasso(args[1],args[2])
    
def applylasso(data,dfname):
        
    model=LogisticRegressionCV(max_iter=10000,cv=5,penalty='l1',solver='liblinear')
    X_train,y_train,X_test,y_test=selectdata(data)
    scaler = StandardScaler()
    
    dftidx=dfidx(X_test)
    
    X_train=scaler.fit_transform(X_train.T).T
    X_test=scaler.fit_transform(X_test.T).T
        
    logcv=model.fit(X_train,y_train)
        
    y_score=logcv.predict_proba(X_test)
        
    dftidx['score']=y_score[:,1]
    dftidx.to_csv("%s-scores"%dfname,index=False,header=True, sep=' ')
    
def dfidx(df):
    df=df.reset_index()
    df=df[df.columns[[0,1,2,3]]]
    return df

def selectdata(data):
    
    df=pd.read_csv(data, header=0, sep=" ")
    test=df[df['Class']==2]
    train=df[df["Class"]!=2]
    
    y_test=test["status"]
    y_train=train["status"]
    
    X_train=train.set_index(['Pathway','Class','status','cluster'])
    X_test=test.set_index(['Pathway','Class','status','cluster'])
    
    return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    ProcessCLI(sys.argv)
