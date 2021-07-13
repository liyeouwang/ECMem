import numpy as np
import sys
input_path = sys.argv[1]
LB_type = sys.argv[2] # LB_mem or LB_demand
out_file = open(sys.argv[1] + "/" + LB_type +".out", "w")
sum_avg = 0
sum_std = 0
sum_max = 0
sum_min = 0
for i in range(10):
	in_file = open(sys.argv[1] + "/" + str(i) + "_server_stats.out", "r")
	servers_load = in_file.readline()
	sum_avg += float(in_file.readline())
	sum_std += float(in_file.readline())
	sum_max += float(in_file.readline())
	sum_min += float(in_file.readline())

out_file.write(f"avg: {sum_avg/10}\n")
out_file.write(f"std: {sum_std/10}\n")
out_file.write(f"max: {sum_max/10}\n")
out_file.write(f"min: {sum_min/10}\n")

