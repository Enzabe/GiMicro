import pandas as pd
import sys
import numpy as np
from collections import defaultdict
from collections import Counter


def ProcessCLI(args):
    df=getfile(args[1])
    
def getfile(infile):
    for line in open(infile,'r').readlines():
        if line.strip().split()[0]=="Pathway":
            print(line.strip())
            samples=line.strip().split()[1:]
        else:
            try:
                sline=line.strip().split("|")
            
                species=sline[1].split()
                print(species[0].split(".s__")[1],' '.join(species[1:]))
                #for i,j in enumerate(samples):
                #    w=species[1:][i]
                #    print(species[1:])
                #    if float(w)>0:
                #        print( sline[0].split(":")[0],species[0].split(".s__")[1],j,species[1:][i])
        
            except IndexError:
                """
                abd=line.strip().split(":")
                wabd=[i for i in abd[1].split() if i.replace('.', '', 1).isdigit()]
                if len(wabd)==len(samples):
                    for i,j in enumerate(samples):
                        if float(wabd[i])>0:
                            print(abd[0],j,wabd[i])
                if len(wabd)>len(samples):
                    wa=wabd[-len(samples):]
                    for i,j in enumerate(samples):
                        if float(wa[i])>0:
                            print(abd[0],j,wa[i])
                """
                continue 
                #print(line.strip())
    return


if __name__ == "__main__":
    ProcessCLI(sys.argv)
