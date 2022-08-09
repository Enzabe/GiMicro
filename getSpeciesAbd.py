import pandas as pd
import sys
import numpy as np
from collections import defaultdict
from collections import Counter


def ProcessCLI(args):

    df=getdata(args[1])
    df.to_csv(args[2],sep=" ", header=True)


def getdata(data):
    df=pd.read_csv(data, header=0, sep=" ")
    return df.groupby(['Pathway']).agg(['sum'])


    
def getnodes(infile):
    subj={}
    
    with open(infile,'r') as file:
        for line in file:
            line=line.strip().split()
            subj[line[0]]=line[1] 
                
    return subj

def inpfile(infile,subj):
    with open(infile,'r') as file:
        for line in file:
            line=line.strip().split()
            try:
        
                print(subj[line[0]],subj[line[1]],line[2])
             
            except KeyError:
                continue 
                
if __name__ == "__main__":
    ProcessCLI(sys.argv)
