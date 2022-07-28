import pandas as pd
import numpy as np 
import sys

def ProcessCLI(args):
     
    df=getdf(args[1]+".idxContigmreads")
    df.columns=["contig","Len","MapReads","UMapReads"]
    unmap=df['UMapReads'].sum()
    mapped=df['MapReads'].sum()
    
    print(args[1],mapped,unmap,mapped+unmap,mapped*1.0/(mapped+unmap))
    
def getdf(indf):
    df=pd.read_csv(indf,sep="\t", header=None)
    return df 


if __name__ == "__main__":
    ProcessCLI(sys.argv)
