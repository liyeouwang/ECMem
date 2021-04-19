import numpy as np
import math 
import sys
import copy
import time
import bisect


def read_inputs(services, servers, vehicles, requests):

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
	for j in range(num_servers):
		for i in range(num_vehicles):
			servers[j].vehicle_in[i] = []
			servers[j].vehicle_out[i] = []
	for i in range(num_vehicles):
		route = [int(x) for x in input().split(", ")]
		for r in range(len(route))
			interval = r[0]
			server = r[1]
			print(interval)
			print(server)
	sys.exit()

	for i in range(num_vehicles):
		for j in range(num_servers):
			ranges = [int(x) for x in input().split()]
			
			for t in range(T_MAX):
				vehicles[i].range[j][t] = ranges[t]
				if(ranges[t] == 1):
					servers[j].in_range[i].append(t)
					if(t == 0):
						servers[j].vehicle_in[i].append(t)
						vehicles[i].servers.append(j)
					elif(ranges[t-1] == 0):
						servers[j].vehicle_in[i].append(t)
						vehicles[i].servers.append(j)
					if(t == T_MAX-1):
						servers[j].vehicle_out[i].append(t)
					elif(ranges[t+1] == 0):
						servers[j].vehicle_out[i].append(t)
			if(len(servers[j].vehicle_out[i])!=len(servers[j].vehicle_in[i])):
				print("in out not equal")
			#print(servers[j].in_range[i])
	'''
	for i in range(num_vehicles):
		for j in range(num_servers):
			print(vehicles[i].range[j])
	'''
	#sys.exit()

	sizes = [int(x) for x in input().split()]
	for k in range(num_services):
		services[k].size = sizes[k]
		#print(services[k].size)

	capacities = [int(x) for x in input().split()]
	for j in range(num_servers):
		servers[j].capacity = capacities[j]
		#print(servers[j].capacity)

if __name__ == '__main__':

	servers = []	
	services = []
	vehicles = []
	requests = []

	#read input file	
	read_inputs(services, servers, vehicles, requests)