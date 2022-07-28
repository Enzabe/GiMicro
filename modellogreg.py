import sys,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import copy 
import re, string,sys
import matplotlib

def ProcessCLI(args):
    
    inp_df1,out_df1,idx1=readdata(args[1],0)
    
    
    ptw=[inp_df1.shape[1]]
    dfs=[(inp_df1,out_df1.values)]
    
    idxs=[idx1]
    
    plotcv(dfs,5,ptw,idxs,args[2])

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

    
def plotcv(dfs,n,ft,idxs,name):
    fig, ax = plt.subplots()
    c=['dimgray','rosybrown','#FF007F','#7F00FF','#FF7F00','#7FFF00','#00FFFF','#007FFF','#0000FF']
    out=open('Modelperformance',"w")
    for i,(x,y) in enumerate(dfs):
        mean_fpr,mean_tpr,mean_auc,weigthts=getcv(x,y,n)
        plt.plot(mean_fpr, mean_tpr,lw=1.75,color=c[i],label='AUC {auc:.3f}'.format(auc=mean_auc),alpha=.8)
        out.write('mean_fpr'+' '+str(ft[i])+' '+' '.join([str(i) for i in list(mean_fpr)])+'\n')
        out.write('mean_tpr'+' '+str(ft[i])+' '+' '.join([str(i) for i in list(mean_tpr)])+'\n')
        out.write('mean_auc'+' '+str(ft[i])+ ' '+str(mean_auc)+'\n')
        
        wout(idxs[i],weigthts,'Species-weight-'+str(ft[i]))
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',label='Chance', alpha=.8) 
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    ax.legend(loc=0,frameon=False,fontsize='small')
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
    fig.savefig(name+'.png', dpi=300)
    fig.savefig(name+'.svg', dpi=300)
    plt.show()

def getcv(X_binary,y_binary,n):
    
    y_binary = (y_binary == 1).astype('uint8')
    cv = StratifiedKFold(n_splits=n,shuffle=True,random_state=42)
    tprs = []
    aucs = []
    weights=[]
    n_features=X_binary.shape[1]
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (train, test) in enumerate(cv.split(X_binary,y_binary)):
        yscore,w = logregTrTs(X_binary[train,:], y_binary[train], X_binary[test,:],y_binary[test],i)
        logit_roc_auc = roc_auc_score(y_binary[test],yscore.reshape(-1,1)[:,0],multi_class="ovo")
        fpr, tpr, thresholds = roc_curve(y_binary[test],yscore.reshape(-1,1)[:,0])
        interp_tpr = np.interp(mean_fpr,fpr,tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(logit_roc_auc)
        weights+=[list(w)]
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    meanweights=np.mean(weights,axis=0).tolist()
    out=open('Weigth-'+str(n_features),"w")
    out.write(' '.join([str(i) for i in meanweights]))

    return mean_fpr,mean_tpr,mean_auc,meanweights

    
def plotroc(pred,score,ft):
    fig, ax = plt.subplots()
    
    for i,j in enumerate(pred):
        
        logit_roc_auc = roc_auc_score(j,score[i].reshape(-1,1)[:,0],multi_class="ovo")
        fpr, tpr, thresholds = roc_curve(j,score[i].reshape(-1,1)[:,0])

        plt.plot(fpr, tpr,lw=2,label="LOGREG-{}".format(ft[i]))
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.legend(fontsize=12)

    ax.legend(loc="best")
    plt.show()

def readdata(data,n):
    
    df=pd.read_csv(data, header=0, sep=" ")
    
    final_df=df.drop(df.columns[4:][df.iloc[0:,4:].apply(lambda col: max(col) <n)], axis=1)

    train=final_df
    idx=list(train.columns[4:])
    
    inp_df = train.drop(train.columns[[0,1,2,3]], axis=1)
    out_df = train["status"]
    
    scaler = StandardScaler()

    inp_df = scaler.fit_transform(inp_df.T).T

    return inp_df,out_df,idx

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

def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred

def logregTrain(X_tr_arr,y_tr_arr,X_ts_arr,y_ts_arr):

    n_features = X_tr_arr.shape[1]
    w, b = weightInitialization(n_features)
    coeff,gradient,costs = model_predict(w,b,X_tr_arr,y_tr_arr,learning_rate=0.0001,no_iterations=10000)
    w = coeff["w"]
    b = coeff["b"]
    
    final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
    m_tr =  X_tr_arr.shape[0]
    m_ts =  X_ts_arr.shape[0]

    y_tr_pred = predict(final_train_pred, m_tr)
    accuracy_tr=accuracy_score(y_tr_pred.T, y_tr_arr)
    y_ts_pred = predict(final_test_pred, m_ts)
    accuracy_ts=accuracy_score(y_ts_pred.T, y_ts_arr)
    
    return accuracy_tr,accuracy_ts

def logregsklrn(X_tr_arr,y_tr_arr,X_ts_arr,y_ts_arr):
    
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_tr_arr, np.ravel(y_tr_arr))
    
    pred = clf.predict(X_ts_arr)
    accuracy=clf.score(X_ts_arr, np.ravel(y_ts_arr))
    
    return accuracy

def rocplot(X,y,n):
    y = (y == 1).astype('uint8')
    cv = StratifiedKFold(n_splits=n,shuffle=True)
    classifier=LogisticRegression(solver='lbfgs')
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],name='ROC fold {}'.format(i+1),alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def logregTrTs(X_tr_arr,y_tr_arr,X_ts_arr,y_ts_arr,i):

    n_features = X_tr_arr.shape[1]
    w, b = weightInitialization(n_features)

    coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=5500)

    w = coeff["w"]
    b = coeff["b"]

    final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
    
    m_tr =  X_tr_arr.shape[0]
    m_ts =  X_ts_arr.shape[0]

    y_tr_pred = predict(final_train_pred, m_tr)
    accuracy_tr=accuracy_score(y_tr_pred.T, y_tr_arr)

    y_ts_pred = predict(final_test_pred, m_ts)
    accuracy_ts=accuracy_score(y_ts_pred.T, y_ts_arr)
    return final_test_pred, w.reshape(-1,1)[:,0]

def lrcvfold(X_binary,y_binary,n):
    
    y_binary = (y_binary == 1).astype('uint8')
    cv = StratifiedKFold(n_splits=n,shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X_binary,y_binary)):
    
        yscore = logregTrTs(X_binary[train,:], y_binary[train], X_binary[test,:],y_binary[test])
        logit_roc_auc = roc_auc_score(y_binary[test],yscore.reshape(-1,1)[:,0],multi_class="ovo")
        fpr, tpr, thresholds = roc_curve(y_binary[test],yscore.reshape(-1,1)[:,0])
        plt.plot(fpr, tpr,lw=1, label='ROC fold {}'.format(i+1))
        
        interp_tpr = np.interp(mean_fpr,fpr,tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(logit_roc_auc)
        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def rfeLR(X,y,n):
    y = (y == 1).astype('uint8')
    cv = StratifiedKFold(n_splits=n,shuffle=True)
    cls=LogisticRegression(solver='lbfgs')
    min_features_to_select = 1
    rfecv = RFECV(estimator=cls, step=1, cv=StratifiedKFold(n),scoring='accuracy',min_features_to_select=min_features_to_select)
    rfecv.fit(X,y)
    print("Optimal number of features : %d" % rfecv.n_features_)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,len(rfecv.grid_scores_) + min_features_to_select),rfecv.grid_scores_)
    plt.show()
    
def testpostfmt(Xtrain,Ytrain,Xtest,Ytest):
    
    X_tr_arr=Xtrain
    y_tr_arr=(Ytrain == 1).astype('uint8').to_numpy().reshape(-1,1)[:,0]
    X_ts_arr= Xtest
    y_ts_arr=(Ytest == 1).astype('uint8').to_numpy().reshape(-1,1)[:,0]
    
    
    n_features = X_tr_arr.shape[1]
    w, b = weightInitialization(n_features)
    
    coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=10000)
    w = coeff["w"]
    b = coeff["b"]

    final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)

    m_tr =  X_tr_arr.shape[0]
    m_ts =  X_ts_arr.shape[0]

    y_tr_pred = predict(final_train_pred, m_tr)
    accuracy_tr=accuracy_score(y_tr_pred.T, y_tr_arr)

    y_ts_pred = predict(final_test_pred, m_ts)
    accuracy_ts=accuracy_score(y_ts_pred.T, y_ts_arr)
    
    return final_train_pred,final_test_pred,y_tr_pred,y_tr_arr,accuracy_tr,y_ts_pred,y_ts_arr,accuracy_ts

    
if __name__ == "__main__":
    ProcessCLI(sys.argv)
