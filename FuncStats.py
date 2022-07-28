import re,sys, string
import pandas as pd
import numpy as np
import scipy as sc
from scipy import stats

#h=open('HLTHY','w')
#a=open('ABX','w')
#c=open('CDI','w')
#Pathway cluster Group status

df=pd.read_csv(sys.argv[1], header=0,sep=' ')
df2=pd.melt(df,id_vars=['Pathway','cluster','Group','status'],var_name='path', value_name='cpm')
df2.to_csv(sys.argv[2],sep=' ',index=False,header=True)


def getd(inf):
    d={'CDI+':{},'ABX+CDI-':{},'CDI-ABX-':{}}
    paths=[]
    with open(inf,'r') as file:
        
        for line in file:
            
            t=line.strip().split()
            #print(t) 
            if t[0]!='Pathway':
                paths.append(t[4])
                
                try:
                    d[t[3]][t[4]].append(float(t[5]))
                except KeyError:
                    d[t[3]][t[4]]=[]
                    d[t[3]][t[4]]=[float(t[5])]
    return d,paths

def getstats(d,paths):
    paths=list(set(paths))
    H=[]
    A=[]
    C=[]
    for p in paths:
        ha=stats.ttest_ind(d['CDI-ABX-'][p], d['ABX+CDI-'][p], equal_var=False)
        hc=stats.ttest_ind(d['CDI-ABX-'][p], d['CDI+'][p], equal_var=False)
        ac=stats.ttest_ind(d['CDI+'][p], d['ABX+CDI-'][p], equal_var=False)
        H.append(ha[1])
        A.append(hc[1])
        C.append(ac[1])
    dd={'path':paths,'HvsA':H,'HvC':A,'CvA':C}
    df=pd.DataFrame(data=dd)
    df.to_csv('PathStats',sep=' ',index=False,header=True)
        
#d,paths=getd(sys.argv[1])
#getstats(d,paths)
