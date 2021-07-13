import sys

out_file = open(sys.argv[1]+"SMT_time.out", "w")
total_time = 0
for i in range(10):
	time = float(input())
	total_time += time
avg_time = total_time/10
print (avg_time)
out_file.write(f"{avg_time}")