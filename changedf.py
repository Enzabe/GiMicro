import pandas as pd
import re,sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv(sys.argv[1], header=0, sep=" ")

paths=[ i.rstrip() for i  in open(sys.argv[2],"r")]
cols=list(df.columns[:4])+paths
df[cols].to_csv(sys.argv[2],sep=' ',na_rep='NA')





