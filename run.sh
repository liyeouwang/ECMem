#!/bin/sh
# This file is called ~/script.sh
#for i in $(seq -f "%02g" 00 10)


for i in $(seq 0 1 9)  
do  
	echo $i
	#python SMT_solver.py < s_I6_K5/$i.in 
	python heuristic.py < l_I50_K200/$i.in l_I50_K200/${i}_ 
	#python verifier.py < test_enter_late/$i.in
	
done  
python cal_sol.py < sol.out l_I50_K200/ SA
#python cal_LB.py l_I50_K200/ LB_demand
#python cal_time.py < time.out s_I6_K5/
#rm time.out
rm sol.out


