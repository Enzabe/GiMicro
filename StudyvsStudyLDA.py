import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.patheffects as PathEffects

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from numpy import mean
from numpy import std

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
def ProcessCLI(args):
    
    plotlda(args[1])
    
def plotlda(data):
    
    df,y=selectdata(data)
    scaler = StandardScaler()
    X=scaler.fit_transform(df.values.T).T

    model = LinearDiscriminantAnalysis(n_components=2)
    
    ldafit=model.fit(X,y)
    
    print(ldafit.explained_variance_ratio_)
    
    Xr = ldafit.transform(X)
    lw = 2

    dc={'KIM':'gray','SM':"navy",'WATS':"darkorange",'LI':'turquoise','FRCK':'red','AC3R':'magenta','RAYM':'pink','MLN':'cyan','EREN':'green','HMP':'black','DUAN':'dodgerblue'}
    
    labels=[ i for i in list(set(y)) if i !='Group']
    
    ax = plt.subplot()
    
    for i in labels:
        ax.scatter(Xr[y == i, 0], Xr[y == i, 1], alpha=0.8,label=i, color=dc[i])
       
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.50)
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    ax.legend(loc="best", shadow=False, scatterpoints=1,frameon=False)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(5,4.5,forward=True)
    fig.savefig(sys.argv[2]+'.png', dpi=600)
    fig.savefig(sys.argv[2]+'.svg', dpi=600)

    plt.show()
    
def selectdata(data):
    
    df=pd.read_csv(data, header=0, sep=" ")
    y=df['Group']
    if 'Group' in list(df.columns):
        X=df.set_index(['Pathway','Group','status'])
    else:
        X=df.set_index(['Pathway','status'])
    return X,y


if __name__ == "__main__":
    ProcessCLI(sys.argv)
