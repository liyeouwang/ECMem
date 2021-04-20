import numpy as np
import sys

#IN_FILE = '/testcases/05.in'
OUT_FILE = 'temp.out'

# Read input
I = int(input())
J = int(input())
K = int(input())
T_MAX = int(input())

T = []
for j in range(0, J):
    T.append([])
    for val in [int(x) for x in input().split()]:
        T[-1].append(val)
T = np.array(T)

C = []
for j in range(0, J):
    C.append([])
    for val in [int(x) for x in input().split()]:
        C[-1].append(val)
C = np.array(C)

Tv = []
for j in range(0, J):
    Tv.append([])
    for jj in range(0, J):
        Tv[-1].append([])
        for val in [int(x) for x in input().split()]:
            Tv[-1][-1].append(val)
Tv = np.array(Tv)

D = []
for i in range(0, I):
    D.append([])
    for val in [int(x) for x in input().split()]:
        D[-1].append(val)
D = np.array(D)

F = []
for i in range(0, I):
    F.append([])
    for val in [int(x) for x in input().split()]:
        F[-1].append(val)
F = np.array(F)

R = []
for i in range(0, I):
    R.append([])
    R_it = input().split(',')
    for r in R_it:
        cum_t = int(r.split()[0])
        cur_j = int(r.split()[1])
        R[-1].extend([cur_j for _ in range (cum_t)])
R = np.array(R)

M = np.array([int(x) for x in input().split()])
M_bar = np.array([int(x) for x in input().split()])

def print_mistake(violation, dec_var):
    print(violation, dec_var)



obj_value = 0
decisions = []
with open(OUT_FILE) as f:
    obj_value = int(f.readline())
    for line in f:
        tmp = line.split()
        decisions.append({})
        decisions[-1]['i'] = int(tmp[0])
        decisions[-1]['k'] = int(tmp[1])
        decisions[-1]['e'] = int(tmp[2])
        decisions[-1]['d'] = int(tmp[3])
        decisions[-1]['s'] = int(tmp[4])
        decisions[-1]['t'] = int(tmp[5])

# This is a j x t table, where each element indicates what "server j excutes at time t".
# The format of a element is a tuple (s, k), 
# which indicates that "server is excuting k, and this execution begins at time s".
scheduling_table = [[ None for _ in range(T_MAX) ] for _ in range(J)]
for dec_var in decisions:
    i = dec_var['i']
    k = dec_var['k']
    e = dec_var['e']
    d = dec_var['d']
    s = dec_var['s']
    t = dec_var['t']

    # Constraint (3.1)
    # A server can only execute one service at a time
    
    # If 
    empty_flag = True
    for exec_time in range(s, s+C[e][k]):
        if (scheduling_table[e][exec_time] is not None):
            empty_flag = False
            break
    
    # print(scheduling_table)
    if (empty_flag):
        for exec_time in range(s, s+C[e][k]):
            scheduling_table[e][exec_time] = (s, k)
    else:
        print(dec_var, C[e][k], T[e][k])
        print(scheduling_table[e])
        # if (scheduling_table[e][s] is None):
        #     print_mistake('(3.1.1)', dec_var)
        if (scheduling_table[e][s][0] != s or scheduling_table[e][s][1] != k):
            print_mistake('(3.1.2)', dec_var)

    # Constraint (3.2)
    # Start execution time must be larger than the earliest execution time.
    if (s < T[e][k]):
        print_mistake('(3.2)', dec_var)
        # print(f'Excution time: {e}, earliest execution time: {}')
    
    # Constraint (3.3)
    # Only after being sent to the delivery server, the result can be delivered.
    if (t < s + C[e][k]):
        print_mistake('(3.3)', dec_var)
    
    # Constraint (3.4)
    # The result must be delivered before its deadline.
    if (t > D[i][k]):
        print('(3.4)', dec_var)

    # Constraint (3.5)
    # The result must be fresh.
    if (t > s + F[i][k]):
        print('(3.5)', dec_var)

    # Constraint (3.6)
    # The vehicle must be reachable.

    if (R[i][t] != d):
        print('(3.6)', dec_var)
