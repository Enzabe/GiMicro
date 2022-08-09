import re, sys

f1=open(sys.argv[1],"r")
f2=open(sys.argv[2],"r")

cf2=[line.strip() for line in f2]
cf1=[line.strip() for line in f1 ]
nh={}
 
for line in cf1:
    k=line.split()

    nh[k[0].strip()]= ' '.join(k[1:3])
    
for c in cf2:
    try:
        print(c,nh[c.strip()])
    except KeyError:
        if c in ["Pathway","run","sample","Sample"]:
            print(c,'status')
        else:
            continue 
