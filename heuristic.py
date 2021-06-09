import numpy as np
import math 
import sys
import copy
import time
import bisect
import matplotlib.pyplot as plt
from tqdm import tqdm
import timeit
T_MAX = 0

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

def check_freshness(request, j, jj):
	if(services[request.service_type].T_compute[j] + services[request.service_type].T_deliver[j][jj] <= freshness):
		return True
	else:
		return False

def check_deadline(request, time):
	if(time <= request.deadline):
		return True
	else:
		return False

def read_inputs():

	num_vehicles = int(input())
	num_servers = int(input())	
	num_services = int(input())

	global T_MAX
	T_MAX = int(input())

	# Servers 
	for j in range(num_servers):
		servers.append(Server(j, num_vehicles, T_MAX))

	# Vehicles 
	for i in range(num_vehicles):
		vehicles.append(Vehicle(i, num_servers))

	# Services 
	for k in range(num_services):
		services.append(Service(k, num_servers, num_vehicles))

	# Eariest start-execute time of each service k of vehicle i on server j 
	for k in range(num_services):
		for i in range(num_vehicles):
			services[k].T_request[i] = [0] * num_servers
	for i in range(num_vehicles):
		for j in range(num_servers):
			tmp = [int(x) for x in input().split()]
			for k in range(num_services):
				services[k].T_request[i][j] = tmp[k]
	
	for k in range(num_services):
		for i in range(num_vehicles):
			for j in range(num_servers):			
				services[k].T_request_avg[i] += services[k].T_request[i][j]
			services[k].T_request_avg[i] = services[k].T_request_avg[i] / num_servers
			print(services[k].T_request_avg[i], end=" ")
		print("\n")


	# Computing time of each task k on server j 
	for j in range(num_servers):
		T_computes = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].T_compute[j] = T_computes[k]

	for k in range(num_services):
		for j in range(num_servers):
			services[k].T_compute_avg += services[k].T_compute[j]
		services[k].T_compute_avg = services[k].T_compute_avg / num_servers
		print(services[k].T_compute_avg, end=" ")
	print("\n")

	# Delivering time of each service k from server j to server jj
	for k in range(num_services):
		for j in range(num_servers):
			services[k].T_deliver[j] = [0] * num_servers
	for j in range(num_servers):
		for jj in range(num_servers):
			T_delivers = [int(x) for x in input().split()]
			for k in range(num_services):
				services[k].T_deliver[j][jj] = T_delivers[k]

	# Deadline of each task 
	num_requests = 0
	for i in range(num_vehicles):
		deadlines = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].deadline[i] = deadlines[k]
			if(deadlines[k] != -1):
				requests.append(Request(i, k, num_requests, num_servers))
				num_requests += 1

	# Ideal time slots of each task on each server
	for r in range(len(requests)):
		for j in range(len(servers)):
			requests[r].ideal_slots[j] = []

	# Fresshness of each service 
	for i in range(num_vehicles):
		freshnesses = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].freshness[i] = freshnesses[k]

	# Route of each vehicle 
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

	# Memory size of each service 
	sizes = [int(x) for x in input().split()]
	for k in range(num_services):
		services[k].size = sizes[k]

	# Capacity of each server (this attribute is not used)
	capacities = [int(x) for x in input().split()]
	for j in range(num_servers):
		servers[j].capacity = capacities[j]

	for request in requests:
		request.deadline = services[request.service_type].deadline[request.vehicle_id]
		request.freshness = services[request.service_type].freshness[request.vehicle_id]
		request.size = services[request.service_type].size
		request.T_compute_avg = services[request.service_type].T_compute_avg
		request.T_request_avg = services[request.service_type].T_request_avg[request.vehicle_id]

#def sort_requests(requests):
	

def ideal_fit(ideal_slot, request, server):
	is_scheduled = False
	T_compute = services[request.service_type].T_compute[server.server_id]


	for slot in server.slots:
		if(slot.start >= ideal_slot.start and slot.end <= ideal_slot.end):
			if(slot.end - slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.end - T_compute
				new_slot = Slot(slot.start, request.start_exe_time)
				slot.start = request.start_exe_time + T_compute
				if(new_slot.duration > 0):
					server.slots.insert(server.slots.index(slot)+1, new_slot)
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break

		if(slot.start >= ideal_slot.start and slot.start <= ideal_slot.end and slot.end >= ideal_slot.end):
			if(ideal_slot.end - slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.start
				slot.start = request.start_exe_time + T_compute
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break

		if(slot.start <= ideal_slot.start and slot.end >= ideal_slot.start and slot.end <= ideal_slot.end):
			if(slot.end - ideal_slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.end - T_compute
				slot.end = request.start_exe_time
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break

		if(slot.start <= ideal_slot.start and slot.end >= ideal_slot.end):
			is_scheduled = True
			request.start_exe_time = ideal_slot.end - T_compute
			new_slot = Slot(slot.start, request.start_exe_time)
			slot.start = request.start_exe_time + T_compute
			if(new_slot.duration > 0):
				server.slots.insert(server.slots.index(slot)+1, new_slot)
			if(slot.duration <= 0):
				server.slots.remove(slot)
			break
	#print("is_scheduled", is_scheduled)
	return is_scheduled

def ideal_scheduling(request, S_exe):

	#for s in server_loading_list:
		#S_exe = s[0]
	for index in range(len(vehicles[request.vehicle_id].servers)-1, -1 , -1):
		S_deliver = vehicles[request.vehicle_id].servers[index]
		T_compute = services[request.service_type].T_compute[S_exe]
		T_deliver = services[request.service_type].T_deliver[S_exe][S_deliver]
		T_request = services[request.service_type].T_request[request.vehicle_id][S_exe]
		t_in = vehicles[request.vehicle_id].servers_in[index]
		t_out = vehicles[request.vehicle_id].servers_out[index]
		if(T_compute + T_deliver <= request.freshness and t_in <= request.deadline):
			start = max(T_request, t_in - T_deliver - T_compute)
			#print(t_out)
			#print(request.deadline)
			end = min(request.deadline - T_deliver, t_out - T_deliver)
			ideal_slot = Slot(start, end)
			ideal_slot.S_deliver = S_deliver 
			request.ideal_slots[S_exe].append(ideal_slot)
			#print("S_exe", S_exe, "ideal_slot.start", ideal_slot.start, "ideal_slot.end", ideal_slot.end)
			if((end - start >= T_compute) and (ideal_fit(ideal_slot, request, servers[S_exe]) == True)):
				request.S_exe = S_exe
				request.S_deliver = S_deliver
				request.T_compute = T_compute
				request.T_deliver = T_deliver
				request.catch_time = request.start_exe_time + T_compute + T_deliver 
				#print("id", request.request_id, "S_exe", request.S_exe, "S_deliver", request.S_deliver, "request.start_exe_time", request.start_exe_time, "request.T_deliver", request.T_deliver, "request.catch_time", request.catch_time)
				return True

	return False

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

	for slot in server.slots:
		if(slot.end - slot.start >= T_compute):
			if((t_latest_start< slot.start) or (t_earliest_start + T_compute >= slot.end) or (slot.end - T_compute < t_earliest_start) or (t_latest_start < t_earliest_start)):
				continue

			if(slot.end > t_latest_start + T_compute):
				t_start_exe = t_latest_start
			else:
				t_start_exe = slot.end - T_compute

			# Find delivery server

			index = bisect.bisect_left(ideal_ends, t_start_exe + T_compute)
			request.S_deliver = request.ideal_slots[server.server_id][index].S_deliver
			request.S_exe = server.server_id
			request.T_compute = T_compute
			request.T_deliver = services[request.service_type].T_deliver[request.S_exe][request.S_deliver]
			request.start_exe_time = t_start_exe
			T_wait_vehicle = request.ideal_slots[server.server_id][index].start - t_start_exe
			request.catch_time = request.start_exe_time+ request.T_compute + request.T_deliver + T_wait_vehicle

			if(request.T_compute + request.T_deliver + T_wait_vehicle <= request.freshness):
				new_slot = Slot(slot.start, t_start_exe)
				slot.start = t_start_exe + request.T_compute + 1
				if(new_slot.duration > 0):
					server.slots.insert(server.slots.index(slot)+1, new_slot)
				if(slot.duration <= 0):
					server.slots.remove(slot)
				is_scheduled = True
				return is_scheduled
			else:
				continue

def non_ideal_scheduling(request, S_exe):
	#for s in server_loading_list:
	#	S_exe = s[0]
	request.ideal_slots[S_exe].sort(key=lambda x: x.end)

	if(non_ideal_fit(request, servers[S_exe]) == True):
		#print("non_ideal_fit")
		return True
	else:
		return False
	#	continue
	# Can't find any available slot
	#print("no feasible")
	#print(f"{request.freshness} {request.deadline} {request.size} {request.request_id}")
	#print("task count: " + str(count))


def update_server_loading(request, server_loading_list):
	#print("-------")
	#for index in range(len(server_loading_list)):
	#	print(server_loading_list[index][1])
	#print("-------")
	server_loading_list.remove((request.S_exe, servers[request.S_exe].loading))
	servers[request.S_exe].loading += request.T_compute
	for index in range(len(server_loading_list)):

		#print(servers[request.S_exe].loading, server_loading_list[index][0])
		if(servers[request.S_exe].loading <= server_loading_list[index][1]):
			#print("insert")
			server_loading_list.insert(index, (request.S_exe, servers[request.S_exe].loading))
			return
	server_loading_list.append((request.S_exe, servers[request.S_exe].loading)) 


def accumulate_memory():
	all_max_memory_use = 0

	for request in requests:
		request.finish_time = request.start_exe_time + request.T_compute + request.T_deliver
		for t in range(request.finish_time, request.catch_time):
			servers[request.S_deliver].memory[t] += request.size

	for server in servers:
		max_memory_use = max(server.memory)
		all_max_memory_use += max_memory_use

	return all_max_memory_use

if __name__ == '__main__':
	count = 0
	no_feasible = 0
	servers = []	
	services = []
	vehicles = []
	requests = []

	# Read input file and initialize parameters
	read_inputs()
	start_time = time.time()
	# Initialize server-loading list
	server_loading_list = []
	for server in servers:
		server_loading_list.append((server.server_id, server.loading))
	
	# sort requests
	'''
	#sort_requests(requests)
	requests.sort(key=lambda x: x.deadline, reverse=False)
	requests_2 = requests[len(requests)//2:]
	#requests_2.sort(key=lambda x: x.size, reverse=True)
	requests_2.sort(key=lambda x: x.T_compute_avg, reverse=True)
	requests = requests[:len(requests)//2]
	for r in requests_2:
		requests.append(r)	
	'''
	requests.sort(key=lambda x: x.T_compute_avg, reverse=True)
	#requests.sort(key=lambda x: x.size, reverse=True)	

	'''
	#secondary 
	for request in requests:
		for s in server_loading_list:
			S_exe = s[0]
			is_scheduled = ideal_scheduling(request, S_exe)
			if(is_scheduled == True):
				break
			is_scheduled = non_ideal_scheduling(request, S_exe)
			if(is_scheduled == True):
				break	
			if(s == len(server_loading_list)-1):
				print("no feasible")	
		update_server_loading(request, server_loading_list)

		#temp for visulaizeion
		for i in range(request.start_exe_time, request.start_exe_time + request.T_compute):
			servers[request.S_exe].schedule_line[i] = 1
	
	'''
	#first
	# Schedule tasks
	# Try to schedule the task in an ideal slot 
	# If it can't be scheduled in any ideal slot, schedule it in normal slot. 
	#for request in tqdm(requests):
	
	for request in requests:
		is_scheduled = False
		for s in server_loading_list:
			S_exe = s[0]
			is_scheduled = ideal_scheduling(request, S_exe)
			if(is_scheduled == True):
				break
		if(is_scheduled == False):
			for s in server_loading_list:
				S_exe = s[0]
				is_scheduled = non_ideal_scheduling(request, S_exe)
		if(is_scheduled == False):
			print("no feasible", "ID: ", request.request_id, "request.T_request: ", request.T_request, "deadline: ", request.deadline, "freshness: ", request.freshness, "T_compute_avg: ", services[request.service_type].T_compute_avg)	
		update_server_loading(request, server_loading_list)

		####
		#temp for visulaizeion
		for i in range(request.start_exe_time, request.start_exe_time + request.T_compute):
			servers[request.S_exe].schedule_line[i] = 1

		if (no_feasible == 100):
			break
	
		# print(str(int((count/len(requests) * 100)))  + "%")
		####
	
	timeslot_used = []
	for server in servers:
		timeslot_used.append(server.schedule_line)
	timeslot_used = np.array(timeslot_used)
	

	#show scheduling result
	'''
	plt.imshow(timeslot_used)
	plt.show()
	'''
	#print(servers[0].schedule_line)
	
	sol = accumulate_memory()

	# running time
	#print(time.time() - start_time)
	running_time = time.time() - start_time
	outfile = open("time.out", "a")
	outfile.write(f"{running_time}")
	outfile.write("\n")

	outfile = open("sol.out", "a")
	outfile.write(f"{sol}")
	outfile.write("\n")

	#print(sol)
	outfile = open("temp.out", "w")
	outfile.write(f"{sol}")
	outfile.write("\n")
	for request in requests: 
		#print(f"{request.T_compute} {request.T_deliver}")
		outfile.write((f"{request.vehicle_id} {request.service_type} {request.S_exe} {request.S_deliver} {request.start_exe_time} {request.catch_time}"))
		outfile.write("\n")
		#print(f"{request.vehicle_id} {request.service_type} {request.S_exe} {request.S_deliver} {request.start_exe_time} {request.catch_time}")

	for i in range(len(server_loading_list)):
		print(server_loading_list[i][1])
	#print(server_loading_list[len(servers)-1][1])
	#print(server_loading_list[0][1])

	# write csv
	import csv
	T_computes = []
	T_delivers = []
	T_wait_vehicles = []
	T_total = []
	T_freshness = []
	for request in requests:
		T_computes.append(request.T_compute)
		T_delivers.append(request.T_deliver)
		T_wait_vehicles.append(request.catch_time - request.finish_time)
		T_total.append(request.T_compute + request.T_deliver + request.catch_time - request.finish_time)
		T_freshness.append(request.freshness)
	rows = zip(T_computes, T_delivers, T_wait_vehicles, T_total, T_freshness)
	with open('stats.csv', "w") as f:
		writer = csv.writer(f)
		writer.writerow(['T_computes', 'T_delivers', 'T_wait_vehicles', 'T_total', 'T_freshness'])
		for row in rows:
			writer.writerow(row)


	'''
	rows = zip(kaggle_docid, kaggle_predicted_results)
	with open('test_NB.csv', "w") as f:
	    writer = csv.writer(f)
	    writer.writerow(['Id', 'Value'])
    for row in rows:
        writer.writerow(row)
	'''
