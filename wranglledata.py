import string, re, sys

dfact={}

for k in open(sys.argv[1],'r'):
    k=k.strip().split()
    try:
        dfact[k[0]]=k[1:]
    except KeyError:
        dfact[k[0]]=k 

for s in open(sys.argv[2],'r'):
    s=s.strip().split()
    try:
        print(s[0],' '.join(dfact[s[0]]+s[4:]))
    except KeyError:
        print(s)
        
    
