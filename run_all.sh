
for A in aletsch-1880-2100 aletsch-basic aletsch-invert Bueler2005C paleo-alps synthetic
do
   cd $A
   igm_run +experiment=params
   cd ..
done

cd quick-demo
   igm_run +experiment=params_1
   igm_run +experiment=params_2
cd ..
