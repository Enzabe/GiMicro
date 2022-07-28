import sys,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict,GridSearchCV,cross_val_score,RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from yellowbrick.classifier.threshold import discrimination_threshold

import copy 
import re, string,sys
import matplotlib
from numpy import mean
from numpy import std
from collections import Counter
import matplotlib as mpl

from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassificationReport
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from yellowbrick.model_selection import FeatureImportances
from sklearn.metrics import matthews_corrcoef as mcc 
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix


def ProcessCLI(args):

     df=pd.read_csv(args[1], header=0, sep=" ")
     subrepIdx,rsidx=readRepeatedMeasures(args[2],df)
     applylasso(args[1],subrepIdx)
     
def applylasso(data,subjRidx):
    
     model2=LogisticRegressionCV(max_iter=10000,cv=5,penalty='l1',solver='liblinear',random_state=42)
     model1 = LogisticRegression(max_iter=10000)
     
     fig, axes = plt.subplots(1,2,constrained_layout=True)

     X,y=sselectdata(data)

     cvrun(model1,X,y,axes[0],'LR model',subjRidx)
     cvrun(model2,X,y,axes[1],'LASSO-LR',subjRidx)
     
     fig = mpl.pyplot.gcf()
     fig.set_size_inches(8,3.5,forward=True)
     fig.savefig(sys.argv[3]+'.png', dpi=600)
     fig.savefig(sys.argv[3]+'.svg', dpi=600)
     fig.savefig(sys.argv[3]+'.pdf', dpi=600)

     plt.show()


def sselectdata(data):
     
    df=pd.read_csv(data, header=0, sep=" ") 
    y=df["status"]
    
    X=df.set_index(['Pathway','cluster','Group','status'])
    scaler = StandardScaler()
    X_in=scaler.fit_transform(X.T).T
    return X_in,y.values

def readRepeatedMeasures(infile,df):
    rdf=pd.read_csv(infile, header=0, sep=" ")
    rpsamples=list(rdf['samples'])
    subjects=list(rdf['subject'])
    rsidx=df[df['Pathway'].isin(rpsamples)].index.values
    
    subrepIdx={}
    for k in set(subjects):
         subrepIdx[k]=[]
    for i in range(len(subjects)):
         try:
              subrepIdx[subjects[i]].append(rsidx[i])
         except IndexError:
              continue
          
    return subrepIdx,rsidx

def cvrun(classifier,X,y,ax,cls,subjRidx):
     
     ax.grid(False)
     
     cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
     tprs = []
     aucs = []
     mean_fpr = np.linspace(0, 1, 100)

     for i, (train, test) in enumerate(cv.split(X, y)):
          
          for k in subjRidx.keys():
               if  len(set(subjRidx[k])&set(train)) >0 and len(set(subjRidx[k])&set(test))>0:
                    
                    if len(set(subjRidx[k])&set(train))>len(set(subjRidx[k])&set(test)):
                         
                         train=set(train).union(set(subjRidx[k])&set(test))
                         test=set(test).difference(set(subjRidx[k])&set(test))
                         
                    if len(set(subjRidx[k])&set(train))<len(set(subjRidx[k])&set(test)):

                         train=set(train).difference(set(subjRidx[k])&set(test))
                         test=set(test).union(set(subjRidx[k])&set(test))

          
          train=np.array(list(train))
          test=np.array(list(test))
          classifier.fit(X[train], y[train])
          
          viz = RocCurveDisplay.from_estimator(classifier,X[test],y[test],alpha=0.5,lw=1,label='_nolegend_',ax=ax,)
          interp_tpr = np.interp(mean_fpr, viz.fpr,viz.tpr)
          interp_tpr[0] = 0.0
          tprs.append(interp_tpr)
          aucs.append(viz.roc_auc)

     ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.6)
          
     mean_tpr = np.mean(tprs, axis=0)
     mean_tpr[-1] = 1.0
     mean_auc = auc(mean_fpr, mean_tpr)
     std_auc = np.std(aucs)

     ax.plot(mean_fpr,mean_tpr,color="black",label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),lw=2,alpha=0.9,)
     std_tpr = np.std(tprs, axis=0)
     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

     ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=0.6,label=r"$\pm$ 1 std. dev.",)
     
     ax.set(xlim=[-0.05, 1.05],ylim=[-0.05, 1.05],title=cls)
     ax.legend(loc=0,fontsize='x-small')
     ax.tick_params(axis='both', labelsize=14)
     ax.set_xlabel('FPR')
     ax.set_ylabel('TPR')
     for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.50)
        ax.spines[axis].set_edgecolor('black')
        
     plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
     plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
     plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
     plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
     #plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    
if __name__ == "__main__":
    ProcessCLI(sys.argv)
