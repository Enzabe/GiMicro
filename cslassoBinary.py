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
from sklearn.metrics import matthews_corrcoef

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
    
    studies={'Watson':['cdiucwatson','cdiwatson','Dawatson','Dbwatson'],
             'Fricke':['cdifricke','donfricke'],
             'Milani':['INFDIAB','NONAB','MiCDI'],
             'Duan':['CDnC','CDnD','CDpD'],
             'Kim':['hmp','cdik'],
             'Smillie':['smd','smr']}
    
    applylasso(args[1],args[2],studies)
    
def applylasso(data,dfname,studies):
        
    model=LogisticRegressionCV(max_iter=10000,cv=5,penalty='l1',solver='liblinear')
    colors={'Watson':'navy','Fricke':'turquoise','Milani':'darkorange','Duan':'pink','Kim':'gray','Smillie':'magenta'}
    fig, ax = plt.subplots()
    
    for s in studies.keys():
        
        X_train,y_train,X_test,y_test=selectdata(data,studies[s])
        scaler = StandardScaler()
        
        dftidx=dfidx(X_test)
    
        X_train=scaler.fit_transform(X_train.T).T
        X_test=scaler.fit_transform(X_test.T).T
        
        logcv=model.fit(X_train,y_train)
        
        y_score=logcv.predict_proba(X_test)
        
        fpr, tpr, thresholds = roc_curve(y_test,y_score[:,1])
        roc_auc = auc(fpr, tpr)
        mean_auc =auc(fpr,tpr)
        
        plt.plot(fpr, tpr,lw=2,label='{p} AUC {auc:.3f}'.format(p=s,auc=mean_auc),alpha=.8,color=colors[s])
        
        print(s, ' Accuracy: ',logcv.score(X_test,y_test))
        
        dftidx['score']=y_score[:,1]
        dftidx.to_csv("%s-scores"%s,index=False,header=True, sep=' ')
        ypred=(y_score[:,1]>0.5).astype('uint8')
        y_test = (y_test == 1).astype('uint8')
        
        print('MCC: ', matthews_corrcoef(y_test,ypred))        
        print(confusion_matrix(y_test,ypred))
        print(classification_report(y_test,ypred))
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red',label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    ax.legend(loc=0,frameon=False,fontsize='x-small')
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.50)
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)

    #plt.xlabel('False Positive Rate',fontsize=18,labelpad=12)
    #plt.ylabel('True Positive Rate',fontsize=18,labelpad=12)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4,3,forward=True)
    fig.savefig(dfname+'.png', dpi=300)
    fig.savefig(dfname+'.svg', dpi=300)
    plt.show()
    
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
