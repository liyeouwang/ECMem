import numpy as np
import os
import json
from random import shuffle

config = {
    'TESTCASE_NAME': 'medium_easy',
    'TESTCASE_NUMS': 10,
    # The servers will be allocated at a X * Y grid.
    # J should always eqaul to X * Y.
    'X': 4,
    'Y': 4,
    'J': 16,

    # The number of vehicles.
    'I': 40,
    
    # The number of service types.
    'K': 5,
    # The time unit limit
    'MAX_TIME': 500,
    'EARLIEST_DEADLINE': 250,

    # Propagation delay
    # This indicates the size of the result of a service.
    # If the size if big, then it costs more propagation delay to send the
    # result from execution server to delivery server.
    # Currently, it is uniform distribution.
    'PROPAGATION_DELAY': 1,

    # The properties of a service.
    #'DIFF_LOW': 20,
    #'DIFF_HIGH': 60,
    'DIFF_LOW': 10,
    'DIFF_HIGH': 30,
    'PROB_LOW': 1,
    'PROB_HIGH': 1,

    # The computation distribution of servers
    #'COMPUTATION_POWER': [1, 3, 5, 7],
    #'COMPUTATION_DIST': [0.25, 0.25, 0.25, 0.25],
    'COMPUTATION_POWER': [1, 3],
    'COMPUTATION_DIST': [0.5, 0.5],

    'FRESHNESS_MEAN': 100,
    'FRESHNESS_STD': 20,
    'FRESHNESS_MIN': 50,

    'LINGER_MEAN': 5,
    'LINGER_STD': 1,

    'MEMORY_MEAN': 20,
    'MEMORY_STD': 5
}
TESTCASE_NAME = config['TESTCASE_NAME']
TESTCASE_NUMS = config['TESTCASE_NUMS']
X = config['X']
Y = config['Y']

I = config['I']
J = config['J']
K = config['K']
MAX_TIME = config['MAX_TIME']


if (not os.path.exists(TESTCASE_NAME)):
    os.mkdir(TESTCASE_NAME)

for testcase_index in range(TESTCASE_NUMS):
    # Initialization
    T = np.zeros((J, K), dtype=int)
    C = np.full((J, K), np.inf, dtype=int)
    Tv = np.full((J, J, K), np.inf, dtype=int)
    D = np.full((I, K), -1, dtype=int)
    F = np.full((I, K), -1, dtype=int)
    # R = np.full((I, J, MAX_TIME), 0, dtype=int)
    R = [[] for _ in range(I)]
    M = np.zeros((K), dtype=int)
    M_bar = np.full((J), 9999999, dtype=int)

    '''
    --------------------------------------------------------------------
    This part deals with the transmitting time T between servers.
    It is proportional to the distance between two servers, The
    distance is |x1-x2| + |y1-y2|.
    We also randomly define Service_Tv to be the computed result loadings 
    of services. 
    Then the transmitting time is Service_Tv * (|x1-x2| + |y1-y2|).
    TODO:
        1. The distribution of Service_Tv. Right now it is uniform(1, 5).
    '''

    PROPAGATION_DELAY = config['PROPAGATION_DELAY']
    Service_Tv = np.random.randint(1, high=PROPAGATION_DELAY+1, size=K)
    for j1 in range(J):
        for j2 in range(J):
            Tv[j1, j2, :] = Service_Tv * (abs(j1 // X - j2 // X) + abs(j1 % X - j2 % X))


    '''
    --------------------------------------------------------------------
    This part define T.
    TODO:
        1. How to randomly decide the value?
    '''
    T = np.random.randint(0, MAX_TIME // 4, size=(J, K))

    '''
    --------------------------------------------------------------------
    This part defines C in an easy way. Each service has a difficulty 
    Service_difficulty = uniform(DIFF_LOW, DIFF_HIGH),and each server
    has an Server_ability = uniform(1, ABILITY_LEVEL) to describe if 
    the server is good at service k.
    TODO:
        1. Uniformly decide the power of servers may not be a good idea.
        2. Right now, we don't have specialized servers that is good at
        specific services.
    '''
    DIFF_LOW = config['DIFF_LOW']
    DIFF_HIGH = config['DIFF_HIGH']
    COMPUTATION_POWER = config['COMPUTATION_POWER']
    COMPUTATION_DIST = config['COMPUTATION_DIST']

    # Fill in Server_ability for each server
    Server_ability = np.full((J, 1), COMPUTATION_DIST[-1])
    accumulated_prob = 0.0
    accumulated_dist = [] # [0.5, 0.3, 0.2] -> [0.5, 0.8, 1.0]
    for prob in COMPUTATION_DIST:
        accumulated_prob += prob
        accumulated_dist.append(accumulated_prob)
    for j in range(J):
        tmp_prob = np.random.random()
        for i, accum_prob in enumerate(accumulated_dist):
            if (tmp_prob < accum_prob):
                Server_ability[j] = 1 / COMPUTATION_POWER[i]
                break

    Service_difficulty = np.random.randint(DIFF_LOW, high=DIFF_HIGH, size=(1, K))
    C = Server_ability * Service_difficulty
    C = np.int_(np.ceil(C))
    C_avg = np.mean(C, axis=0) # The average computation time of each service


    '''
    --------------------------------------------------------------------
    This part decides D and F.
    TBD:
    1. The probability of a service: uniform(PROB_LOW, PROB_HIGH)
    2. The distribution of deadline: uniform(0, MAX_TIME)
    3. The distribution of freshness: normal(FRESHNESS_MEAN, FRESHNESS_STD)
    '''
    PROB_LOW = config['PROB_LOW']
    PROB_HIGH = config['PROB_HIGH']
    FRESHNESS_MEAN = config['FRESHNESS_MEAN']
    FRESHNESS_STD = config['FRESHNESS_STD']
    FRESHNESS_MIN = config['FRESHNESS_MIN']
    EARLIEST_DEADLINE = config['EARLIEST_DEADLINE']
    workload = 0
    Service_prob = np.array([(PROB_HIGH - PROB_LOW) * np.random.random() + PROB_LOW for _ in range(K)])
    for k in range(K):
        for i in range(I):
            if (np.random.random() < Service_prob[k]):
                workload += C_avg[k]
                D[i][k] = np.random.randint(EARLIEST_DEADLINE, high=MAX_TIME, dtype=int)

                fresh = int(np.random.normal(FRESHNESS_MEAN,  FRESHNESS_STD) // 1)
                F[i][k] = FRESHNESS_MIN if fresh < FRESHNESS_MIN else fresh



    workload = workload / (J * MAX_TIME)

    '''
    --------------------------------------------------------------------
    This part decides R. The map is circullar, which means that when a
    vehicle exceeds the edge, it will appear at the other side.
    TBD:
    1. The distribution of time that a vehicle is lingering in a server's 
        area: normal(LINGER_MEAN, LINGER_STD)
    '''
    LINGER_MEAN = config['LINGER_MEAN']
    LINGER_STD = config['LINGER_STD']
    for i in range(I):
        t = 0
        src_loc = np.random.randint(0, high=J, dtype=int)
        while True:
            # Determine destination.
            dst_loc = np.random.randint(0, high=J, dtype=int)
            src_x = src_loc // X
            src_y = src_loc % X
            dst_x = dst_loc // X
            dst_y = dst_loc % X
            delta_x = dst_x - src_x
            delta_y = dst_y - src_y


            '''
                The direction definition.
                Below permutate the direction of the vehicle.
                For example, the vehicle is going from (1, 1) to (3, 3).
                The route can be any permutation of [1, 1, 3, 3].
                The number indicates the direction.
                            4
                            |
                            |
                    2 ------------> 1 x
                            |
                            |     
                            v
                            3
                            y
            '''
            route = []
            if (delta_x > 0):
                route.extend([1 for _ in range(delta_x)])
            if (delta_x < 0):
                route.extend([2 for _ in range(-delta_x)])
            if (delta_y > 0):
                route.extend([3 for _ in range(delta_x)])
            if (delta_y < 0):
                route.extend([4 for _ in range(-delta_y)])
            shuffle(route)

            
            for r in route:
                linger_time = int(np.random.normal(LINGER_MEAN, LINGER_STD)//1)
                if linger_time <= 0:
                    linger_time = 1
                if (t + linger_time >= MAX_TIME):
                    R[i].append((MAX_TIME - t, src_x * X + src_y))
                    t += linger_time
                    break
                R[i].append((linger_time, src_x * X + src_y))
                t += linger_time
                if (r == 1):
                    src_x = src_x + 1
                elif (r == 2):
                    src_x = src_x - 1
                elif (r == 3):
                    src_y = src_y + 1
                elif (r == 4):
                    src_y = src_y - 1
                src_x = src_x % X
                src_y = src_y % Y

            if t >= MAX_TIME:
                break

            src_loc = dst_loc


    '''
    --------------------------------------------------------------------
    This part decides M.
    TBD:
    1. The distribution of M: normal(MEMORY_MEAN, MEMORY_STD)
    '''
    MEMORY_MEAN = config['MEMORY_MEAN']
    MEMORY_STD = config['MEMORY_STD']
    M = np.array(np.random.normal(MEMORY_MEAN, MEMORY_STD, K) // 1, dtype=int)



# --------------------------------------------
# Generate files
    #with open(f'./{TESTCASE_NAME}/{testcase_index}_{workload:5.4f}.in', 'w+') as f:
    with open(f'./{TESTCASE_NAME}/{testcase_index}.in', 'w+') as f:
        f.write(f"{I}\n")
        # J
        f.write(f"{J}\n")
        # K
        f.write(f"{K}\n")
        # T_MAX
        f.write(f"{MAX_TIME}\n")
        # T
        for j in range(J):
            for k in range(K):
                f.write(f"{T[j][k]} ")
            f.write("\n")
        # C
        for j in range(J):
            for k in range(K):
                f.write(f"{C[j][k]} ")
            f.write("\n")
        # Tv
        for j in range(J):
            for jj in range(J):
                for k in range(K):
                    f.write(f"{Tv[j][jj][k]} ")
                f.write("\n")
        # D
        for i in range(I):
            for k in range(K):
                f.write(f"{D[i][k]} ")
            f.write("\n")
        # F
        for i in range(I):
            for k in range(K):
                f.write(f"{F[i][k]} ")
            f.write("\n")
        # R
        for i in range(I):
            for r, route in enumerate(R[i]):
                if r == len(R[i])-1:
                    f.write(f"{route[0]} {route[1]}")
                else:
                    f.write(f"{route[0]} {route[1]}, ")
            f.write("\n")
        # M
        for k in range(K):
            f.write(f"{M[k]} ")
        f.write("\n")
        # M_bar
        for j in range(J):
            f.write(f"{M_bar[j]} ")
        f.close()

with open(f'{TESTCASE_NAME}/config.json', 'w+') as file:
    json.dump(config, file)