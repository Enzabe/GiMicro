import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import sys,re
import numpy as np
from statannot import add_stat_annotation

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
def ProcessCLI(args):

    df=getdata(args[1])
    x="Status"
    y="NormalizedCount"
    plotdf(df,'CDI-HLTHY',x,y)
    plotdfp(getdata(args[2]),'CDI-HLTHY-PostFMT',x,y)
    

def getdata(data):
    return pd.read_csv(data,header=0, sep=" ")

def plotdfp(df,name,x,y):
    pal = "Set2"
    ort="v"
    f, ax = plt.subplots(figsize=(7, 5))
    order=['CDI','HLHTY','postFMT']
    #t-test_ind
    ax=pt.half_violinplot( x = x, y = y, data = df, palette = pal, bw = .2, cut = 0.,scale = "area", width = .6, inner = None, orient = ort)
    ax=sns.stripplot( x = x, y = y, data = df, palette = pal, edgecolor = "white",size = 5, jitter = 1, zorder = 0, orient = ort)
    ax=sns.boxplot( x = x, y = y, data = df, color = "black", width = .15, zorder = 10,showcaps = True, boxprops={'facecolor':'none',"zorder":10},showfliers=False,whiskerprops ={'linewidth':2,"zorder":10},saturation = 1, orient = ort)
    test_results = add_stat_annotation(ax, data=df, x=x, y=y, order=order,box_pairs=[('CDI','HLHTY'),('HLHTY','postFMT'),('CDI','postFMT')],test='t-test_ind', text_format='full',loc='inside', verbose=2)
    #test_results = add_stat_annotation(ax, data=df, x=x, y=y, order=order,box_pairs=[('CDI','HLHTY'),('HLHTY','postFMT'),('CDI','postFMT')],test='Mann-Whitney', text_format='full',loc='inside', verbose=2)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_xticklabels(x, fontsize=20)
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
    #labels=['CDI positive','CDI negative', 'postFMT']
    ax.set_xticklabels(labels)
    plt.savefig(name+'.svg', bbox_inches='tight',dpi=300)
    plt.show()
        
def plotdf(df,name,x,y):
    pal = "Set2"
    ort="v"
    f, ax = plt.subplots(figsize=(7, 5))
    order=['CDI','HLHTY']
    ax=pt.half_violinplot( x = x, y = y, data = df, palette = pal, bw = .2, cut = 0.,scale = "area", width = .6, inner = None, orient = ort)
    ax=sns.stripplot( x = x, y = y, data = df, palette = pal, edgecolor = "white",size = 5, jitter = 1, zorder = 0, orient = ort)
    ax=sns.boxplot( x = x, y = y, data = df, color = "black", width = .15, zorder = 10,showcaps = True, boxprops={'facecolor':'none',"zorder":10},showfliers=False,whiskerprops ={'linewidth':2,"zorder":10},saturation = 1, orient = ort)
    #test_results = add_stat_annotation(ax, data=df, x=x, y=y, order=order,box_pairs=[('CDI','HLHTY')],test='Mann-Whitney', text_format='full',loc='inside', verbose=2)
    test_results = add_stat_annotation(ax, data=df, x=x, y=y, order=order,box_pairs=[('CDI','HLHTY')],test='t-test_ind', text_format='full',loc='inside', verbose=2)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_xticklabels(x, fontsize=20)
    plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 2)
    plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 2)
    labels=['CDI positive','CDI negative']
    ax.set_xticklabels(labels)

    
    plt.savefig(name+'.svg', bbox_inches='tight',dpi=300)
    plt.show()
    
if __name__ == "__main__":
    ProcessCLI(sys.argv)
