import numpy as np
import sys
total_time = 0
total_sol = 0
sols = []
unfeasible_rates = []
experiment_type = sys.argv[2] 
out_file = open(sys.argv[1]+"result_"+ experiment_type +".out", "w")
for i in range(10):
	time = float(input())
	total_time += time
	sol = float(input())
	unfeasible_rate = float(input())
	total_sol += sol
	sols.append(sol)
	unfeasible_rates.append(unfeasible_rate)
	print(sol, end = " ")
	out_file.write(f"{sol} ")
out_file.write("\n")
print("\n")
print ("time", total_time/10)
print ("avg", total_sol/10)
print ("std", np.std(sols, ddof=1))
print ("max", max(sols))
print ("min", min(sols))
print ("unfeasible_rate", sum(unfeasible_rates)/10)
out_file.write(f"time: {total_time/10}\n")
out_file.write(f"avg: {total_sol/10}\n")
out_file.write(f"std: {np.std(sols, ddof=1)}\n")
out_file.write(f"max: {max(sols)}\n")
out_file.write(f"min: {min(sols)}\n")
out_file.write(f"unfeasible_rate: {sum(unfeasible_rates)/10}\n")

