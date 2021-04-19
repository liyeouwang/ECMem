import numpy as np
import math 
import sys
import copy
import time
import bisect

T_MAX = 0

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
		self.T_request = [0] * num_servers
		self.T_deliver = [0] * num_servers
		self.T_compute = [0] * num_servers

class Server:
	def __init__(self, server_id, num_vehicles, T_MAX):
		self.server_id = server_id
		self.memory = [0] * T_MAX
		self.capacity = 0
		self.slots = []
		self.slots.append(Slot(0, T_MAX-1))
		self.loading = 0

class Request:
	def __init__(self, vehicle_id, service_type, request_id, num_servers):
		self.service_type = service_type
		self.request_id = request_id
		self.vehicle_id = vehicle_id
		self.deadline = 0	
		self.T_request = 0
		self.T_compute = 0
		self.T_deliver = 0
		self.freshness = 0
		self.size = 0
		self.S_exe = 0
		self.S_deliver = 0
		self.start_exe_time = 0
		self.catch_time = 0
		self.finish_time = 0
		self.ideal_slots = [0] * num_servers
		self.earliest_start_time = 0 # = T_request

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

	# Eariest start-execute time of each service k on server j 
	for j in range(num_servers):
		tmp = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].T_request[j] = tmp[k]

	# Computing time of each task k on server j 
	for j in range(num_servers):
		T_computes = [int(x) for x in input().split()]
		for k in range(num_services):
			services[k].T_compute[j] = T_computes[k]

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


def fit(ideal_slot, request, server):
	is_scheduled = False
	T_compute = services[request.service_type].T_compute[server.server_id]

	for slot in server.slots:
		if(slot.start >= ideal_slot.start and slot.end <= ideal_slot.end):
			if(slot.end - slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.end - T_compute
				new_slot = Slot(slot.start, request.start_exe_time)
				slot.start = request.start_exe_time + T_compute + 1
				if(new_slot.duration > 0):
					server.slots.insert(server.slots.index(slot)+1, new_slot)
				if(slot.duration <= 0):
					server.slots.remove(slot)
				break

		if(slot.start >= ideal_slot.start and slot.start <= ideal_slot.end and slot.end >= ideal_slot.end):
			if(ideal_slot.end - slot.start >= T_compute):
				is_scheduled = True
				request.start_exe_time = slot.start
				slot.start = request.start_exe_time + T_compute + 1
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
			slot.start = request.start_exe_time + T_compute + 1
			if(new_slot.duration > 0):
				server.slots.insert(server.slots.index(slot)+1, new_slot)
			if(slot.duration <= 0):
				server.slots.remove(slot)
			break

	return is_scheduled

def ideal_scheduling(request, server_loading_list):
	for s in server_loading_list:
		S_exe = s[0]
		for index in range(len(vehicles[request.vehicle_id].servers)):
			S_deliver = vehicles[request.vehicle_id].servers[index]
			T_compute = services[request.service_type].T_compute[S_exe]
			T_deliver = services[request.service_type].T_deliver[S_exe][S_deliver]
			T_request = services[request.service_type].T_request[S_exe]
			t_in = vehicles[request.vehicle_id].servers_in[index]
			t_out = vehicles[request.vehicle_id].servers_out[index]
			if(T_compute + T_deliver <= request.freshness and t_in <= request.deadline):
				start = max(T_request, t_in - T_deliver - T_compute)
				end = min(request.deadline - T_deliver, t_out - T_deliver)
				if((end - start > T_compute) and (fit(Slot(start, end), request, servers[S_exe]) == True)):
					request.S_exe = S_exe
					request.S_deliver = S_deliver
					request.T_compute = T_compute
					request.T_deliver = T_deliver
					request.catch_time = request.start_exe_time + T_compute + T_deliver
					return True



if __name__ == '__main__':

	servers = []	
	services = []
	vehicles = []
	requests = []

	# Read input file and initialize parameters
	read_inputs()

	# Initialize server-loading list
	server_loading_list = []
	#server_loading_dict = dict()
	for server in servers:
		#server_loading_dict.update({server.server_id: server.loading})
		server_loading_list.append((server.server_id, server.loading))
	
	# Schedule tasks
	# Try to schedule the task in an ideal slot 
	# If it can't be scheduled in any ideal slot, schedule it in normal slot. 
	for request in requests:
		is_scheduled = ideal_scheduling(request, server_loading_list)
		sys.exit()
		if(is_scheduled == True):	
			#update_server_loading(request, server_loading_list)
			continue
		#else:
			#non_ideal_scheduling(request, server_loading_list)
			#update_server_loading(request, server_loading_list)
