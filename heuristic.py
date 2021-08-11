import numpy as np
import math 
import sys
from copy import deepcopy
import time
import bisect
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit
T_MAX = 0
THRESHOLD = 0.5
config = {
    #Simulated annealing parameters
    "T": 10000,
    "r": 0.999 
}

'''
slot: (start, end]
ex: slot.start=0  slot.end = 5
duration: 5
availabe time: 0, 1, 2, 3, 4 
'''
class Slot:
	def __init__(self, start, end):
		self.start = start
		self.end = end
		self.duration = end - start 
		#for ideal slot
		self.S_deliver = -1

class Vehicle:
	def __init__(self, vehicle_id, num_servers): 
		self.vehicle_id = vehicle_id
		self.servers = []
		self.servers_in = []
		self.servers_out = []

class Service:
	def __init__(self, service_type, num_servers, num_vehicles
		):		
		self.service_type = service_type
		self.size = 0
		self.deadline = [0] * num_vehicles
		self.freshness = [0] * num_vehicles
		self.T_request = [0] * num_vehicles
		self.T_deliver = [0] * num_servers
		self.T_compute = [0] * num_servers
		self.T_compute_avg = 0.0
		self.T_request_avg = [0.0] * num_vehicles

class Server:
	def __init__(self, server_id, num_vehicles, T_MAX):
		self.server_id = server_id
		self.memory = [0] * T_MAX
		self.capacity = 0
		self.slots = []
		self.slots.append(Slot(0, T_MAX-1))
		self.loading = 0
		self.schedule_line = [0] * T_MAX

class Request:
	def __init__(self, vehicle_id, service_type, request_id, num_servers):
		self.service_type = service_type
		self.request_id = request_id
		self.vehicle_id = vehicle_id
		self.deadline = 0	
		self.T_request = 0
		self.T_compute = 0
		self.T_compute_avg = 0.0
		self.T_request_avg = 0.0
		self.T_deliver = 0
		self.freshness = 0
		self.size = 0
		self.S_exe = 0
		self.S_deliver = 0
		self.start_exe_time = 0
		self.catch_time = 0
		self.finish_time = 0
		self.ideal_slots = [0] * num_servers
		for j in range(num_servers):
			self.ideal_slots[j] = []
		self.is_feasible = True

'''
--------------------------------------------------------------------
Read the input file created by testcase_generator.py
'''
def read_inputs():
	num_vehicles = int(input())
	num_servers = int(input())	
	num_services = int(input())

	global T_MAX
	T_MAX = int(input())

	# Construct the server list 
	for j in range(num_servers):
		servers.append(Server(j, num_vehicles, T_MAX))

	# Construct the vehicle list 
	for i in range(num_vehicles):
		vehicles.append(Vehicle(i, num_servers))

	# Construct the service list 
	for k in range(num_services):
		services.append(Service(k, num_servers, num_vehicles))

	# Obtain the eariest start-execute time of each service k of vehicle i on server j 
	for k in range(num_services):
		for i in range(num_vehicles):
			services[k].T_request[i] = [0] * num_servers
	for i in range(num_vehicles):
		for j in range(num_servers):
			tmp = [int(x) for x in input().split()]
			for k in range(num_services):
				services[k].T_request[i][j] = tmp[k]
	
	# Calculate the average eariest start-execute time of each service k of vehicle i among all servers 
	for k in range(num_services):
		for i in range(num_vehicles):
			for j in range(num_servers):			
				services[k].T_request_avg[i] += services[k].T_request[i][j]
			services[k].T_request_avg[i] = services[k].T_request_avg[i] / num_servers

	# Obtain the computing time of each service k on server j 
	for j in range(num_servers):
		T_computes = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].T_compute[j] = T_computes[k]

	# Calculate the average computing time of each service k 
	for k in range(num_services):
		for j in range(num_servers):
			services[k].T_compute_avg += services[k].T_compute[j]
		services[k].T_compute_avg = services[k].T_compute_avg / num_servers


	# Obtain the delivering time of each service k from server j to server jj
	for k in range(num_services):
		for j in range(num_servers):
			services[k].T_deliver[j] = [0] * num_servers
	for j in range(num_servers):
		for jj in range(num_servers):
			T_delivers = [int(x) for x in input().split()]
			for k in range(num_services):
				services[k].T_deliver[j][jj] = T_delivers[k]

	# Obtain the deadline of each task 
	num_requests = 0
	for i in range(num_vehicles):
		deadlines = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].deadline[i] = deadlines[k]
			if(deadlines[k] != -1):
				requests.append(Request(i, k, num_requests, num_servers))
				num_requests += 1

	# Create the list of ideal time slots for each task on each server
	for r in range(len(requests)):
		for j in range(len(servers)):
			requests[r].ideal_slots[j] = []

	# Obtain the fresshness of each service k for each vehicle i
	for i in range(num_vehicles):
		freshnesses = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].freshness[i] = freshnesses[k]

	# Obtain the route of each vehicle i
	for i in range(num_vehicles):
		route = [x for x in input().split(", ")]
		travel_time = 0
		for interval in route:
			interval = interval.split()
			period = int(interval[0])
			server_id = int(interval[1])
			if(server_id == -1):
				travel_time = travel_time + period
				continue
			vehicles[i].servers.append(server_id)
			vehicles[i].servers_in.append(travel_time)
			vehicles[i].servers_out.append(travel_time + period - 1)
			travel_time = travel_time + period

	# Obtain the memory size of each service k
	sizes = [int(x) for x in input().split()]
	for k in range(num_services):
		services[k].size = sizes[k]

	# Obtain the capacity of each server (this attribute is not used)
	capacities = [int(x) for x in input().split()]
	for j in range(num_servers):
		servers[j].capacity = capacities[j]

	# Initialize the tasks
	for request in requests:
		request.deadline = services[request.service_type].deadline[request.vehicle_id]
		request.freshness = services[request.service_type].freshness[request.vehicle_id]
		request.size = services[request.service_type].size
		request.T_compute_avg = services[request.service_type].T_compute_avg
		request.T_request_avg = services[request.service_type].T_request_avg[request.vehicle_id]	

'''
--------------------------------------------------------------------
Check if the task can be scheduled in the ideal slot
'''
def ideal_fit(ideal_slot, request, server):
	is_scheduled = False
	T_compute = services[request.service_type].T_compute[server.server_id]

	# Iterate all slots in the server to find feasible ideal slot to schedule the task
	for slot in server.slots:
		# Case 1
		# ---[     ideal slot      ]---
		# ------[ feasible slot ]------
		if(slot.start >= ideal_slot.start and slot.end <= ideal_slot.end):
			if(slot.end - slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.end - T_compute
				new_slot = Slot(slot.start, request.start_exe_time)
				slot.start = request.start_exe_time + T_compute
				slot.duration = slot.end - slot.start
				if(new_slot.duration > 0):
					server.slots.insert(server.slots.index(slot)+1, new_slot)
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break
		
		# Case 2
		# ---[     ideal slot      ]---------
		# ------[    feasible slot    ]------
		if(slot.start >= ideal_slot.start and slot.start <= ideal_slot.end and slot.end >= ideal_slot.end):
			if(ideal_slot.end - slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.start
				slot.start = request.start_exe_time + T_compute
				slot.duration = slot.end - slot.start
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break
		
		# Case 3
		# ------[   ideal slot     ]---------
		# ----[  feasible slot  ]------------
		if(slot.start <= ideal_slot.start and slot.end >= ideal_slot.start and slot.end <= ideal_slot.end):
			if(slot.end - ideal_slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.end - T_compute
				slot.end = request.start_exe_time
				slot.duration = slot.end - slot.start
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break
		
		# Case 4
		# -------[   ideal slot  ]---------
		# ----[    feasible slot    ]------
		if(slot.start <= ideal_slot.start and slot.end >= ideal_slot.end):
			is_scheduled = True
			request.start_exe_time = ideal_slot.end - T_compute
			new_slot = Slot(slot.start, request.start_exe_time)
			slot.start = request.start_exe_time + T_compute
			slot.duration = slot.end - slot.start
			if(new_slot.duration > 0):
				server.slots.insert(server.slots.index(slot)+1, new_slot)
			if(slot.duration <= 0):
				server.slots.remove(slot)
			break

	return is_scheduled

'''
--------------------------------------------------------------------
Iterate all possible delivery edge servers to find an ideal slot to schedule
'''
def ideal_scheduling(request, S_exe):

	# Iterate all possible delivery edge servers of the vehicle
	# Start from the last edge server the vehicle will pass by
	for index in range(len(vehicles[request.vehicle_id].servers)-1, -1 , -1):
		S_deliver = vehicles[request.vehicle_id].servers[index]
		T_compute = services[request.service_type].T_compute[S_exe]
		T_deliver = services[request.service_type].T_deliver[S_exe][S_deliver]

		# There is no connection between the execution edge server and the possible delivery edge server
		if(T_deliver == -1):
			continue

		T_request = services[request.service_type].T_request[request.vehicle_id][S_exe]

		# The time the vehicle enter the covering range of the possible delivery edge server 
		t_in = vehicles[request.vehicle_id].servers_in[index]

		# The time the vehicle leave the covering range of the possible delivery edge server 
		t_out = vehicles[request.vehicle_id].servers_out[index]

		# Obtain the ideal slot 
		if(T_compute + T_deliver <= request.freshness and t_in <= request.deadline):
			start = max(T_request, t_in - T_deliver - T_compute)
			end = min(request.deadline - T_deliver, t_out - T_deliver)
			ideal_slot = Slot(start, end)
			ideal_slot.S_deliver = S_deliver 
			request.ideal_slots[S_exe].append(ideal_slot)

			# Check if the task can be scheduled in the ideal slot
			if((end - start >= T_compute) and (ideal_fit(ideal_slot, request, servers[S_exe]) == True)):
				request.S_exe = S_exe
				request.S_deliver = S_deliver
				request.T_compute = T_compute
				request.T_deliver = T_deliver
				request.catch_time = request.start_exe_time + T_compute + T_deliver 
				return True

	return False

'''
--------------------------------------------------------------------
Check if the task can be scheduled in the normal slot
'''
def non_ideal_fit(request, server):
	is_scheduled = False
	T_compute = services[request.service_type].T_compute[server.server_id]
	t_earliest_start = services[request.service_type].T_request[request.vehicle_id][server.server_id]
	if not request.ideal_slots[server.server_id]:
		return False
	t_latest_start = max(ideal_slot.end for ideal_slot in request.ideal_slots[server.server_id]) - T_compute
	

	ideal_ends = []
	for ideal_slot in request.ideal_slots[server.server_id]:
		ideal_ends.append(ideal_slot.end)

	# Iterate all slots in the server to find a feasible slot for the task 
	for slot in server.slots:
		if(slot.end - slot.start >= T_compute):
			if((t_latest_start< slot.start) or (t_earliest_start + T_compute >= slot.end) or (slot.end - T_compute < t_earliest_start) or (t_latest_start < t_earliest_start)):
				continue
			if(slot.end > t_latest_start + T_compute):
				t_start_exe = t_latest_start
			else:
				t_start_exe = slot.end - T_compute

			# Find delivery edge server
			index = bisect.bisect_left(ideal_ends, t_start_exe + T_compute)
			request.S_deliver = request.ideal_slots[server.server_id][index].S_deliver
			request.S_exe = server.server_id
			request.T_compute = T_compute
			request.T_deliver = services[request.service_type].T_deliver[request.S_exe][request.S_deliver]
			if(request.T_deliver == -1):
				print("wrong!!!!!!!")
			request.start_exe_time = t_start_exe
			T_wait_vehicle = request.ideal_slots[server.server_id][index].start - t_start_exe
			request.catch_time = request.start_exe_time+ request.T_compute + request.T_deliver + T_wait_vehicle
			
			# Update the slot list
			if(request.T_compute + request.T_deliver + T_wait_vehicle <= request.freshness):
				new_slot = Slot(slot.start, t_start_exe)
				slot.start = t_start_exe + request.T_compute + 1
				slot.duration = slot.end - slot.start
				if(new_slot.duration > 0):
					server.slots.insert(server.slots.index(slot)+1, new_slot)
				if(slot.duration <= 0):
					server.slots.remove(slot)
				is_scheduled = True
				return is_scheduled
			else:
				continue

'''
--------------------------------------------------------------------
Find a normal slot after iterating all servers and have not found an ideal slot to schedule the task
'''
def non_ideal_scheduling(request, S_exe):
	request.ideal_slots[S_exe].sort(key=lambda x: x.end)

	if(non_ideal_fit(request, servers[S_exe]) == True):
		return True
	else:
		return False


'''
--------------------------------------------------------------------
After scheduling a task in a edge server, update the server list to maitain the order
'''
def update_server_loading(request, server_loading_list):
	server_loading_list.remove((request.S_exe, servers[request.S_exe].loading))
	servers[request.S_exe].loading += request.T_compute
	for index in range(len(server_loading_list)):
		if(servers[request.S_exe].loading <= server_loading_list[index][1]):
			server_loading_list.insert(index, (request.S_exe, servers[request.S_exe].loading))
			return
	server_loading_list.append((request.S_exe, servers[request.S_exe].loading)) 


'''
--------------------------------------------------------------------
Get the summation of the max memory usage of each server 
'''
def accumulate_memory():
	all_max_memory_use = 0
	for server in servers:
		for t in range(T_MAX):
			server.memory[t] = 0
	for request in requests:
		request.finish_time = request.start_exe_time + request.T_compute + request.T_deliver
		for t in range(request.finish_time, request.catch_time):
			servers[request.S_deliver].memory[t] += request.size

	for server in servers:
		max_memory_use = max(server.memory)
		all_max_memory_use += max_memory_use

	return all_max_memory_use


'''
--------------------------------------------------------------------
(For Simulated Annealing)
'''
def remove_scheduled_task(request):
	S_exe = request.S_exe
	for i in range(len(servers[S_exe].slots)):
		if(servers[S_exe].slots[i].start == (request.start_exe_time + request.T_compute)):
			if(i+1 < len(servers[S_exe].slots)):
				if(servers[S_exe].slots[i+1].end == request.start_exe_time):
					servers[S_exe].slots[i].start = servers[S_exe].slots[i+1].start
					servers[S_exe].slots[i].duration = servers[S_exe].slots[i+1].duration + request.T_compute + servers[S_exe].slots[i].duration 
					servers[S_exe].slots.remove(servers[S_exe].slots[i+1])
				else:
					servers[S_exe].slots[i].start = request.start_exe_time
					servers[S_exe].slots[i].duration =  request.T_compute + servers[S_exe].slots[i].duration 
				return		
		elif(servers[S_exe].slots[i].end == request.start_exe_time):
			servers[S_exe].slots[i].end = request.start_exe_time + request.T_compute 
			servers[S_exe].slots[i].duration =  servers[S_exe].slots[i].duration + request.T_compute
			return

	new_slot = Slot(request.start_exe_time, request.start_exe_time + request.T_compute)
	index = 0
	for slot in servers[S_exe].slots:
		if(slot.end < new_slot.start):
			index = servers[S_exe].slots.index(slot)
			break
	servers[S_exe].slots.insert(index, new_slot)

'''
--------------------------------------------------------------------
(For Simulated Annealing)
Random switch the execution edge servers of two tasks. 
After switching the execution edge servers,
we reschedule these two tasks in their new execution edge servers and update the result. 

'''
def pick_neighbor():
	if(len(no_ideals) > 0):
		a = np.random.choice(no_ideals)
	else:
		while True:
			a = np.random.randint(0, len(requests))
			if(requests[a].is_feasible == True):
				break
	while True:
		b = np.random.randint(0, len(requests))
		if((requests[b].S_exe != requests[a].S_exe) and requests[b].is_feasible == True):
			break
	request_tmp1 = deepcopy(requests[a])	
	request_tmp2 = deepcopy(requests[b])

	remove_scheduled_task(request_tmp1)
	remove_scheduled_task(request_tmp2)

	request_tmp1.S_exe, request_tmp2.S_exe = request_tmp2.S_exe, request_tmp1.S_exe 

	requests_picked = [request_tmp1, request_tmp2]
	is_scheduled = False
	for request in requests_picked:
		is_scheduled = False
		S_exe = request.S_exe
		is_scheduled = ideal_scheduling(request, S_exe)
		if(is_scheduled == False):
			is_scheduled = non_ideal_scheduling(request, S_exe)			
		if(is_scheduled == False):
			break
	if(is_scheduled == True):
		if(len(no_ideals) > 0):
			no_ideals.remove(a)
		requests[a] = deepcopy(request_tmp1)
		requests[b] = deepcopy(request_tmp2)


'''
--------------------------------------------------------------------
Simulated Annealing
'''
def simulated_annealing(sol):
	#random seed
	np.random.seed(1)
	cur_sol = sol
	best_sol = cur_sol
	best_requests = deepcopy(requests)

	T = config["T"]
	r = config["r"]
	while T > 1:
		if(time.time() - start_time > 300):
			break
		pick_neighbor()
		new_sol = accumulate_memory()
		#print(new_sol)
		delta_cost = new_sol - cur_sol
		if(delta_cost <= 0):
			cur_sol = new_sol
			if(cur_sol < best_sol):
				best_sol = cur_sol
				best_requests = deepcopy(requests)
				print("best_sol = " + str(best_sol))				
				if(best_sol == 0):
					break
		else:
			if(np.random.rand() <= math.exp(- delta_cost / T )):
				cur_sol = new_sol			
		T *= r
	#print(best_sol)
	return best_sol	


'''
--------------------------------------------------------------------
main 
'''
if __name__ == '__main__':
	count = 0
	no_feasible = 0
	servers = []	
	services = []
	vehicles = []
	requests = []
	no_ideals = []
	unfeasibles = []

	# Read input file and initialize parameters
	read_inputs()
	start_time = time.time()
	# Initialize server-loading list
	server_loading_list = []
	for server in servers:
		server_loading_list.append((server.server_id, server.loading))
	
	# Sort tasks
	sum_C = 0
	for k in range(len(services)):
		sum_C += services[k].T_compute_avg * len(vehicles)

	# Decide which mode to sort the tasks
	if(sum_C/ (T_MAX*len(servers))  >= THRESHOLD):
		# If the load is heavy, sort the tasks by there computing demands in decreasing order
		sort_mode = 2
	else:
		# If the load is not heavy, sort the tasks by there memory sizes in decreasing order
		sort_mode = 3	
	#sort_mode = 2 # for load balance 

	#if(sys.argv[2] == "LB_demand"):
	#	sort_mode = 2
	#elif(sys.argv[2] == "LB_mem"):
	#	sort_mode = 3

	# Not used 
	if(sort_mode == 0):
		requests.sort(key=lambda x: x.T_compute_avg, reverse=True)
		requests_2 = requests[len(requests)//2:]
		requests_2.sort(key=lambda x: x.size, reverse=True)
		requests = requests[:len(requests)//2]
		for r in requests_2:
			requests.append(r)	
	# Not used 
	elif(sort_mode == 1):
		requests.sort(key=lambda x: x.deadline, reverse=False)
		requests_2 = requests[len(requests)//2:]
		requests_2.sort(key=lambda x: x.size, reverse=True)
		requests = requests[:len(requests)//2]
		for r in requests_2:
			requests.append(r)	

	# Sort by computing demands
	elif(sort_mode == 2):
		requests.sort(key=lambda x: x.T_compute_avg, reverse=True)

	# Sort by memory sizes
	elif(sort_mode == 3):
		requests.sort(key=lambda x: x.size, reverse=True)	


	# Schedule tasks
	# Try to schedule the task in an ideal slot 
	# If it can't be scheduled in any ideal slot, schedule it in normal slot. 
	#for request in tqdm(requests):
	index = 0
	for request in requests:
		is_scheduled = False
		server_loading_list_tmp = server_loading_list[:len(servers)//1]
		# Try ideal scheduling first
		for s in server_loading_list_tmp:
			S_exe = s[0]
			is_scheduled = ideal_scheduling(request, S_exe)
			if(is_scheduled == True):
				break
		# If the task can not be scheduled in any ideal slot, try non ideal scheduling 
		if(is_scheduled == False):
			no_ideals.append(index)
			for s in server_loading_list:
				S_exe = s[0]
				is_scheduled = non_ideal_scheduling(request, S_exe)
				if(is_scheduled == True):					
					break
		# If the task can not be scheduled in any ideal slot or any normal slot, record it as an infeasible task
		if(is_scheduled == False):
			no_feasible += 1
			request.is_feasible = False
			unfeasibles.append(index)
			no_ideals.remove(index)
		update_server_loading(request, server_loading_list)
		index += 1
	
	# Get the objective
	sol = accumulate_memory()
	
	'''
	# If the solution is not 0, we can run Simulated Annealing to refine the result
	if(sol > 0):
		sol = simulated_annealing(sol)
	'''

	# Count the number of infeasible tasks
	no_feasible = 0
	for request in requests:
		if(request.is_feasible == False):
			no_feasible += 1

	#print(time.time() - start_time)
	#print(sol)
	#print(f"no_feasible: {no_feasible} rate: {no_feasible/len(requests)}")
	
	# Write the output files
	running_time = time.time() - start_time
	outfile = open("sol.out", "a")
	outfile.write(f"{running_time}")
	outfile.write("\n")
	outfile.write(f"{sol}")
	outfile.write("\n")
	outfile.write(f"{no_feasible/len(requests)}")
	outfile.write("\n")

	outfile = open("temp.out", "w")
	outfile.write(f"{sol}")
	outfile.write("\n")
	for request in requests: 
		outfile.write((f"{request.vehicle_id} {request.service_type} {request.S_exe} {request.S_deliver} {request.start_exe_time} {request.catch_time}"))
		outfile.write("\n")

	# Write tasks stats 
	outfile = open(sys.argv[1]+"task_stats.out", "w")
	outfile.write(f"{no_feasible} {no_feasible/len(requests)}")
	outfile.write("\n")
	for request in requests:
		T_wait_vehicle = request.catch_time - request.finish_time
		T_total = request.T_compute + request.T_deliver + request.catch_time - request.finish_time
		outfile.write(f"{request.T_compute} {request.T_deliver} {T_wait_vehicle} {T_total}")
		outfile.write("\n")
	
	# Write server stats 
	outfile = open(sys.argv[1]+"server_stats.out", "w")
	loads = []
	for i in range(len(server_loading_list)):
		loads.append(server_loading_list[i][1])
		outfile.write(f"{server_loading_list[i][1]} ")
	outfile.write("\n")
	outfile.write(f"{float(sum(loads))/len(servers)}\n")
	outfile.write(f"{np.std(loads, ddof=1)}\n")
	outfile.write(f"{max(loads)}\n")
	outfile.write(f"{min(loads)}\n")
	outfile.close()

	'''
	timeslot_used = []
	for server in servers:
		timeslot_used.append(server.schedule_line)
	timeslot_used = np.array(timeslot_used)
	####
	#temp for visulaizeion
	for i in range(request.start_exe_time, request.start_exe_time + request.T_compute):
		servers[request.S_exe].schedule_line[i] = 1
	####
	#show scheduling result
	
	plt.imshow(timeslot_used)
	plt.show()
	#print(servers[0].schedule_line)
	'''
