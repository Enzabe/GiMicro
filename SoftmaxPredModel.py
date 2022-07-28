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
from sklearn.preprocessing import label_binarize

import copy 
import re, string,sys
import matplotlib
from numpy import mean
from numpy import std
from collections import Counter

def ProcessCLI(args):

     studies={'Watson':['cdiucwatson','cdiwatson','Dawatson','Dbwatson'],
             'Fricke':['cdifricke','donfricke'],
             'Milani':['INFDIAB','NONAB','MiCDI'],
             'Duan':['CDnC','CDnD','CDpD'],
             'Kim':['hmp','cdik'],
             'Oluf':['Dag0','Dag4','Dag8','Dag42','Dag180'],
             'Raymond':['CTLD0','CTLD7','CTLD90','EXPD0','EXPD7','EXPD90'],
             'Smillie':['smd','smr']}

     applylasso(args[1],studies)
    
def applylasso(data,studies):
    
    model=LogisticRegressionCV(max_iter=10000,cv=5,penalty='l1',solver='liblinear')
    #model = LogisticRegressionCV(penalty='l1',multi_class="multinomial",solver="saga",cv=5,random_state=42,max_iter=10000)
    
    for s in studies.keys():

        X_train,y_train,X_test,y_test=selectdata(data,studies[s])
        scaler = StandardScaler()

        dftidx=dfidx(X_test)

        X_train=scaler.fit_transform(X_train.T).T
        X_test=scaler.fit_transform(X_test.T).T

        logcv=model.fit(X_train,y_train)

        y_score=logcv.predict_proba(X_test)

        dftidx['score']=y_score.reshape(-1,1)[:,0]
        dftidx.to_csv("%s-scores"%s,index=False,header=True, sep=' ')

        ypred=logcv.predict(X_test)
        print(s)
        print(confusion_matrix(y_test,ypred,labels=[0,1,2]))
        print(classification_report(y_test,ypred,labels=[0,1,2]))


    
def selectdata(data,lst):

    df=pd.read_csv(data, header=0, sep=" ")

    test=df[df["cluster"].isin(lst)]
    train=df[~df["cluster"].isin(lst)]

    y_test=test["status"]
    y_train=train["status"]

    X_train=train.set_index(['Pathway','cluster','Group','status'])
    X_test=test.set_index(['Pathway','cluster','Group','status'])

    return X_train,y_train,X_test,y_test
        

def getmn(dct):
    d={}
    for k in dct.keys():
        d[c]=np.mean(dct[c], axis=0)
    return d

    
def getfeaturess(model,X,y,Xcolumns):
    selector=SelectFromModel(model)
    logcv=selector.fit(X,y)
    selecetdFeatures=list(Xcolumns[(logcv.get_support())])
    return selecetdFeatures

def pdct(dct):
    for k in list(dct.keys()):
        print(k,dct[k])

def writel(lst,filo):
    out=open(filo,"w")
    for k in lst:
        out.write(str(k)+'\n')
    out.close()
    
def wout(idx,weights,filo):
    
    out=open(filo,"w")
    for i,k in enumerate(idx):
        out.write(str(k)+' '+str(weights[i])+'\n')
    out.close()

def sselectdata(data):
    df=pd.read_csv(data, header=0, sep=" ") 
    y=df["status"]
    X=df.set_index(['Pathway','cluster','Group','status'])
    return X,y 

if __name__ == "__main__":
    ProcessCLI(sys.argv)
