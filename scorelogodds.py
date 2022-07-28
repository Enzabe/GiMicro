import sys,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as clt
import ptitprince as pt
import copy
import re, string,sys
import matplotlib
from numpy import mean
from numpy import std
from collections import Counter

def ProcessCLI(args):
    df=selectdata(args[1])
    pltdf(df,args[2])

def pltdf(df,figname):
    f, ax = plt.subplots(figsize=(5, 4))
    pal = "Set1"
    ort="v"
    dy='logScore'
    dx='Group'
    ax=sns.stripplot( x = dx, y = dy, data = df, palette = pal, edgecolor = "black",linewidth=1,#hue='cluster',
                      size = 5, jitter = True, zorder = 0, orient = ort)
    #ax = sns.violinplot(x=dx, y=dy,data=df,saturation=1)
    plt.axhline(0,color='black',linewidth=1)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.50)
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5,4,forward=True)
    fig.savefig(figname+'.png', dpi=300)
    fig.savefig(figname+'.svg', dpi=300)

    plt.show()

def selectdata(data):

    df=pd.read_csv(data, header=0, sep=" ")
    df['logScore']= np.log2(df['score']/(1-df['score']))
    return df
if __name__ == "__main__":
    ProcessCLI(sys.argv)
