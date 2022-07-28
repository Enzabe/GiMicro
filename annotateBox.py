import pandas as pd
import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def ProcessCLI(args):

    df=pd.read_csv(args[1], header=0,sep=' ')
    df['cpm']=np.log2(df['cpm']+1)
    plotbx(df,args[2]) 

def plotbx(df,name):
    
    fig, ax = plt.subplots(figsize=(16,4.5))

    pal = {'CDI+':"#7a7a7a",'CDI-ABX-':"#f5f5f5",'ABX+CDI-':"#ababab"}
    boxprops = {'edgecolor': 'k', 'linewidth': 1.3, 'facecolor': 'w'}
    lineprops = {'color': 'k', 'linewidth': 2}
    kwargs = {'palette': pal, 'hue_order': ['CDI-ABX-','ABX+CDI-','CDI+']}

    boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,'whiskerprops': lineprops, 'capprops': lineprops,'width': 0.75},**kwargs)
    stripplot_kwargs = dict({'linewidth': 0.6, 'size': 3, 'alpha': 0.7},**kwargs)
    box_pairs=bpairs(list(set(df['path'])))
    
    sns.boxplot(x='path', y='cpm', hue='Group', data=df,fliersize=0,ax=ax,palette=pal,)
    sns.stripplot(x='path', y='cpm', hue='Group', data=df,jitter=True, split=True,**stripplot_kwargs)
    #add_stat_annotation(ax, data=df, x='path', y='cpm', hue='Group', box_pairs=box_pairs,test='Mann-Whitney', loc='inside',linewidth=1, verbose=0,text_format='star')
    add_stat_annotation(ax, data=df, x='path', y='cpm', hue='Group', box_pairs=box_pairs,test='t-test_ind', loc='inside',linewidth=1, verbose=0,text_format='star')

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.xticks(fontsize=7)
    plt.xticks(rotation=30,ha='right')
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[0:3], labels[0:3],loc=0,fontsize='large',frameon=False,handletextpad=0.5)
    lgd.legendHandles[0]._sizes = [40]
    lgd.legendHandles[1]._sizes = [40]
    lgd.legendHandles[2]._sizes = [40]
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(name+'.svg', bbox_inches='tight',dpi=300)
    plt.savefig(name+'.pdf', dpi=300, bbox_inches='tight')
        
    plt.show()
    
def bpairs(p):
    bp=[]
    for i in list(set(p)):
        bp+=[((i,'CDI-ABX-'),(i,'ABX+CDI-')),((i,'CDI-ABX-'),(i,'CDI+')),((i,'ABX+CDI-'),(i,'CDI+'))]
    return bp
    
def readdata(inf):
    df=pd.read_csv(inf, header=0, sep=" ")
    return df

if __name__=="__main__":
    ProcessCLI(sys.argv)
