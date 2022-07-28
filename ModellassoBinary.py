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
    
    applylasso(args[1],args[2],args[3])
    
def applylasso(data,dfname,features):
    
    model=LogisticRegressionCV(max_iter=10000,cv=5,penalty='l1',solver='liblinear')
    
    X,y=selectdata(data)
    cols=X.columns
    selector=SelectFromModel(model)
    scaler = StandardScaler()

    X_train=scaler.fit_transform(X.T).T

    logcv=selector.fit(X_train,y)
    selecetdFeatures=list(X.columns[(logcv.get_support())])
    
    X[selecetdFeatures].to_csv(dfname,index=True, sep=' ',header=True)
    writel(selecetdFeatures,features)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    tprs = []
    fprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    X_in=scaler.fit_transform(X.T).T

    y_in = (y == 1).astype('uint8').values
    fig, ax = plt.subplots()
    acc=[]
    features=[]
    for train, test in cv.split(X_in,y_in):
        logc=LogisticRegressionCV(max_iter=10000,penalty='l1',solver='liblinear')
        mylasso=logc.fit(X_in[train,:],y_in[train])
        
        y_score=mylasso.predict_proba(X_in[test,:])
        fpr, tpr, thresholds = roc_curve(y_in[test],y_score[:,1])
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr,fpr,tpr)
        interp_tpr[0] = 0.0
        
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        acc.append(mylasso.score(X_in[test,:],y_in[test]))
        feature=getfeaturess(logc,X_in[train,:],y_in[train],cols)
        features+=feature
        
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Model training accuracy: ',np.mean(acc)) 

    plt.plot(mean_fpr,mean_tpr,color="black",lw=2,label="Lasso AUC = %0.3f" %mean_auc)
    
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

    plt.xlabel('False Positive Rate',fontsize=18,labelpad=12)
    plt.ylabel('True Positive Rate',fontsize=18,labelpad=12)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4,3,forward=True)
    fig.savefig(dfname+'.png', dpi=300)
    fig.savefig(dfname+'.svg', dpi=300)
    plt.show()
    if len(selecetdFeatures)== len(list(set(features))):
        return selecetdFeatures
    #if len(selecetdFeatures)!= len(list(set(features))):
    #    print('Selected features differ from those selected via customized CV steps. Using selected features via SelectFromModel')
    #    writel(list(set(features)),'CV-features')
    #    return selecetdFeatures
    
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

def selectdata(data):
    df=pd.read_csv(data, header=0, sep=" ")
    y=df["status"]
    X=df.set_index(['Pathway','cluster','Group','status'])
    return X,y 

if __name__ == "__main__":
    ProcessCLI(sys.argv)
