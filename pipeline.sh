D=DATADIR
SCR=DIR/scr
WDIR=`pwd`

SIDS=DIRSAMP/samples

cd $WDIR

n=$(ls -l ${WDIR}/*pathab* | wc -l )

echo $n

### Regular functional analysis

humann_join_tables -i tempdata -o ${1}_pathabundance.tsv --file_name pathabundance.tsv
humann_renorm_table -i ${1}_pathabundance.tsv -o ${1}-cpm.tsv --units cpm --update-snames
grep -v UNINT ${1}-cpm.tsv | grep -v UNMA | grep -v "|"  > ${1}.integrated-pwz-red

awk -v nt="$n" '{printf $1 "\t"}{i = nt; for (--i; i >= 0; i--){ printf "%s\t",$(NF-i)} print ""}' ${1}.integrated-pwz-red | sed 's/://g' | sed 's/#/Pathway/g' | awk '$1=$1' > ${1}.pwz-cpm.tsv
sed -i 's/_Abundance-CPM//g' ${1}.pwz-cpm.tsv
head -n 1 ${1}.pwz-cpm.tsv | xargs -n 1 | awk '$1=$1' > ${1}-headers-abd
python $SCR/mrg2files.py $SIDS  ${1}-headers-abd | awk '$1=$1' >  ${1}-headers-abd-ordered
awk -f $SCR/transpose.awk ${1}-headers-abd-ordered | awk '$1=$1' > ${1}-dfs-paths
cat ${1}.pwz-cpm.tsv >> ${1}-dfs-paths
awk -f $SCR/transpose.awk ${1}-dfs-paths  | awk '!($4="")' | awk '$1=$1' >  ${1}-dfp-final

####Species contributing on the observed functional profiles 

grep -v UNINT ${1}-cpm.tsv | grep -v UNMA | awk '$1=$1' > ${1}-species-cpm.tsv
sed -i 's/_Abundance-CPM//g'  ${1}-species-cpm.tsv
sed -i 's/#//g' ${1}-species-cpm.tsv

python $SCR/getspeciesFile.py ${1}-species-cpm.tsv > ${1}-species-perpath-cpm.tsv
python  $SCR/getspecies.py ${1}-species-perpath-cpm.tsv ${1}-species-persample

sed -i '2d' ${1}-species-persample
awk 'NR==1 {$1="Pathway " $1} 1' ${1}-species-persample | awk '$1=$1' > ${1}-df-species-persample

head -n 1 ${1}-df-species-persample | xargs -n 1 | awk '$1=$1' > ${1}-headers
python $SCR/mrg2files.py $SIDS ${1}-headers | awk '$1=$1' >  ${1}-headers-ordered

awk -f $SCR/transpose.awk ${1}-headers-ordered | awk '$1=$1' > ${1}-dfs

cat ${1}-df-species-persample >> ${1}-dfs
sed -i '4,5d' ${1}-dfs

awk -f $SCR/transpose.awk  ${1}-dfs | awk '$1=$1' > ${1}-dfs-final






