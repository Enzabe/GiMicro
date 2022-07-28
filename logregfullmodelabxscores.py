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
from sklearn.metrics import roc_curve,matthews_corrcoef

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
    
    colors={'Watson':'navy','Fricke':'turquoise','Milani':'darkorange','Duan':'pink','Kim':'gray','Smillie':'magenta'}
    fig, ax = plt.subplots()
    
    for s in studies.keys():
        
        X_train,y_train,X_test,y_test=selectdata(data,studies[s])
        scaler = StandardScaler()
        
        dftidx=dfidx(X_test)
    
        X_train=scaler.fit_transform(X_train.T).T
        X_test=scaler.fit_transform(X_test.T).T
    
        y_score=testscore(X_train,y_train,X_test,y_test)
        dftidx['score']=y_score.reshape(-1,1)[:,0]
        dftidx.to_csv("%s-scores"%s,index=False,header=True, sep=' ')
        
        ypred=(y_score.reshape(-1,1)[:,0]>0.5).astype('uint8')
        y_test = (y_test == 1).astype('uint8')

        print('%s MCC: '%s, matthews_corrcoef(y_test,ypred))
        print(confusion_matrix(y_test,ypred))
        print(classification_report(y_test,ypred))
    
def testscore(Xtrain,Ytrain,Xtest,Ytest):

    X_tr_arr=Xtrain
    y_tr_arr=(Ytrain == 1).astype('uint8').to_numpy().reshape(-1,1)[:,0]
    X_ts_arr= Xtest
    y_ts_arr=(Ytest == 1).astype('uint8').to_numpy().reshape(-1,1)[:,0]

    n_features = X_tr_arr.shape[1]
    w, b = weightInitialization(n_features)

    coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=5500)
    w = coeff["w"]
    b = coeff["b"]

    final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)

    return final_test_pred

def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b

def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result.astype(float)))
    return final_result

def model_optimize(w, b, X, Y):
    m = X.shape[0]
    final_result = sigmoid_activation(np.dot(w,X.T)+b)

    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost

def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        grads, cost = model_optimize(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)

        if (i % 100 == 0):
            costs.append(cost)

    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs


    
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
