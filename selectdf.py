import pandas as pd
import sys
import numpy as np
from collections import defaultdict
from collections import Counter


def ProcessCLI(args):

    df=getfile(args[1],getpath(args[2]))

    
def getpath(data):
    
    d={}
    for line in open(data,'r').readlines():
        d[line.rstrip()]=line.rstrip()
    return d


def getfile(infile,d):
    
    for line in open(infile,'r').readlines():
        
        if line.strip().split()[0]=="Pathway":
            print(line.strip())
        else:
            try:
                sline=line.rstrip().split(":")
                if d[sline[0]]:
                    print(line.strip())
                
            except KeyError:
                continue
            
    return

if __name__ == "__main__":
    ProcessCLI(sys.argv)
