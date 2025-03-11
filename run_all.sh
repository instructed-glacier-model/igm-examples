
for A in aletsch-1880-2100 aletsch-basic aletsch-invert Bueler2005C paleo-alps quick-demo quick-demo-mysmb rhone-invert synthetic
do
   cd $A
   igm_run +experiment=params
   cd ..
done
