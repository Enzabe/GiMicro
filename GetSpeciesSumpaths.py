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
    return df.groupby(['species','sample']).abd.agg(['sum'])


if __name__ == "__main__":
    ProcessCLI(sys.argv)
