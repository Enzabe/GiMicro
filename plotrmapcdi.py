import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import sys,re
import numpy as np
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
def ProcessCLI(args):

    df=getdata(args[1])
    x="Status"
    y="NormalizedCount"
    


def getdata(data):
    return pd.read_csv(data,header=0, sep=" ")

def plotdfh(df,name,x,y):
    pal = "Set2"
    ort="h"
    f, ax = plt.subplots(figsize=(7, 5))

    ax=pt.half_violinplot( x = x, y = y, data = df, palette = pal, bw = .2, cut = 0.,scale = "area", width = .6, inner = None, orient = ort)
    ax=sns.stripplot( x = x, y = y, data = df, palette = pal, edgecolor = "white",size = 5, jitter = 1, zorder = 0, orient = ort)
    ax=sns.boxplot( x = x, y = y, data = df, color = "black", width = .15, zorder = 10,showcaps = True, boxprops={'facecolor':'none',"zorder":10},showfliers=False,whiskerprops ={'linewidth':2,"zorder":10},saturation = 1, orient = ort)
    
    if savefigs:
        plt.savefig(name+'.png', bbox_inches='tight')
        
if __name__ == "__main__":
    ProcessCLI(sys.argv)
