import itertools
from pysmt.shortcuts import Symbol, And, Or, GE, LT, Plus, Equals, Int, get_model, Bool, Exists, Iff, TRUE, Not, Max, is_sat
from pysmt.typing import INT, ArrayType, BOOL
import time

def symbol_name(name, *args):
    result = str(name)
    for x in args: result += "_" + str(x)
    return result

def main():
    # Given variables =================================================

    I = int(input())
    J = int(input())
    K = int(input())
    T_MAX = int(input())

    array_1D = Symbol("1D", ArrayType(INT, INT))
    array_2D = Symbol("2D", ArrayType(INT, ArrayType(INT, INT)))

    # T[j][k]
    # Earliest start execution time of service k on server j
    '''
    T = Symbol("T", ArrayType(INT, ArrayType(INT, INT)))
    for j in range(0, J):
        T_row = array_1D
        k = 0
        for val in [int(x) for x in input().split()]:
            # print(symbol_name("T", j, k), " -> ", str(val))
            T_row = T_row.Store(Int(k), Int(val))
            k = k + 1
        T = T.Store(Int(j), T_row)
    '''
    T = Symbol("T", ArrayType(INT, ArrayType(INT, ArrayType(INT, INT))))
    for i in range(0, I):
        T_mat = array_2D
        for j in range(0, J):
            T_mat_row = array_1D
            k = 0
            for val in [int(x) for x in input().split()]:
                # print(symbol_name("T", j, k), " -> ", str(val))
                T_mat_row = T_mat_row.Store(Int(k), Int(val))
                k = k + 1
            T_mat = T_mat.Store(Int(j), T_mat_row)
        T = T.Store(Int(i), T_mat)

    # C[j][k]
    # The computation time of service k on server j
    C = Symbol("C", ArrayType(INT, ArrayType(INT, INT)))
    for j in range(0, J):
        C_row = array_1D
        k = 0
        for val in [int(x) for x in input().split()]:
            # print(symbol_name("C", j, k), " -> ", str(val))
            C_row = C_row.Store(Int(k), Int(val))
            k = k + 1
        C = C.Store(Int(j), C_row)

    # Tv[j][j'][k]
    # the transmission time (delivery) of service k from server j to server j'
    Tv = Symbol("Tv", ArrayType(INT, ArrayType(INT, ArrayType(INT, INT))))

    for j in range(0, J):
        Tv_mat = array_2D
        for jj in range(0, J):
            Tv_mat_row = array_1D
            k = 0
            for val in [int(x) for x in input().split()]:
                # print(symbol_name("Tv", j, jj, k), " -> ", str(val))
                Tv_mat_row = Tv_mat_row.Store(Int(k), Int(val))
                k = k + 1
            Tv_mat = Tv_mat.Store(Int(jj), Tv_mat_row)
        Tv = Tv.Store(Int(j), Tv_mat)

    # D[i][k]
    # Vehicle i's reception deadline for service k
    # valid[i][k] == -1 iff vehicle i doesn't demand service k
    D = Symbol("D", ArrayType(INT, ArrayType(INT, INT)))
    valid = []

    for i in range(0, I):
        D_row = array_1D
        k = 0
        valid.append([])
        for val in [int(x) for x in input().split()]:
            # print(symbol_name("D", i, k), " -> ", str(val))
            D_row = D_row.Store(Int(k), Int(val))
            k = k + 1
            if(val == -1): valid[-1].append(False)
            else: valid[-1].append(True)
            
        D = D.Store(Int(i), D_row)

    # F[i][k]
    # The required freshness of vehicle i for service k
    F = Symbol("F", ArrayType(INT, ArrayType(INT, INT)))

    for i in range(0, I):
        F_row = array_1D
        k = 0
        for val in [int(x) for x in input().split()]:
            # print(symbol_name("F", i, k), " -> ", str(val))
            F_row = F_row.Store(Int(k), Int(val))
            k = k + 1
        F = F.Store(Int(i), F_row)

    # R[i][j][t]
    # if vehicle i is in the communication of server j at time t
    '''
    R = Symbol("R", ArrayType(INT, ArrayType(INT, ArrayType(INT, INT))))
    for i in range(0, I):
        R_mat = array_2D
        for j in range(0, J):
            R_mat_row = array_1D
            t = 0
            for val in [int(x) for x in input().split()]:
                # print(symbol_name("R", i, j, t), " -> ", str(val))
                R_mat_row = R_mat_row.Store(Int(t), Int(val))
                t = t + 1
            R_mat = R_mat.Store(Int(j), R_mat_row)
        R = R.Store(Int(i), R_mat)
    '''
    R = Symbol("R", ArrayType(INT, ArrayType(INT, INT)))
    for i in range(0, I):
        R_mat = array_1D

        R_it = input().split(',')
        t = 0
        for r in R_it:
            cum_t = int(r.split()[0])
            cur_j = int(r.split()[1])
            for _ in range(cum_t):
                R_mat = R_mat.Store(Int(t), Int(cur_j))
                t += 1
        R = R.Store(Int(i), R_mat)

    # M[k]
    # the required memory size (delivery) of service k
    M = [int(x) for x in input().split()]

    # M_bar[j]
    # the memory size of server j
    M_bar = [int(x) for x in input().split()]


    start_time = time.time()
    # Decision variables ================================================= 

    class DV: # Decision variable
        def __init__(self, i, k):
            self.i_int = i
            self.k_int = k
            self.i = Int(i)
            self.k = Int(k)
            self.e = Symbol(symbol_name("e", i, k), INT)
            self.d = Symbol(symbol_name("d", i, k), INT)
            self.s = Symbol(symbol_name("s", i, k), INT)
            self.t = Symbol(symbol_name("t", i, k), INT)


    DV_set = set()

    for i in range(0, I):
        for k in range(0, K):
            if valid[i][k]:
                DV_set.add(DV(i, k))

    # Constraints ================================================= 

    # Variable domain
    domain = And(
        [And(GE(dv.e, Int(0)), LT(dv.e, Int(J))) for dv in DV_set] + 
        [And(GE(dv.d, Int(0)), LT(dv.d, Int(J))) for dv in DV_set] +
        [And(GE(dv.s, Int(0)), LT(dv.s, Int(T_MAX))) for dv in DV_set] + 
        [And(GE(dv.t, Int(0)), LT(dv.t, Int(T_MAX))) for dv in DV_set]
    )

    # Execution constraint
    eq1 = And([Or(dv1.s + C.Select(dv1.e).Select(dv1.k) <= dv2.s, dv1.s >= dv2.s + C.Select(dv2.e).Select(dv2.k), And(dv1.k.Equals(dv2.k), dv1.s.Equals(dv2.s))) for (dv1, dv2) in itertools.combinations(DV_set, 2)])

    # Timing constraint
    #eq2 = And([T.Select(dv.e).Select(dv.k) <= dv.s for dv in DV_set])
    eq2 = And([T.Select(dv.i).Select(dv.e).Select(dv.k) <= dv.s for dv in DV_set])
    eq3 = And([dv.s + C.Select(dv.e).Select(dv.k) + Tv.Select(dv.e).Select(dv.d).Select(dv.k) <= dv.t for dv in DV_set])
    eq4 = And([dv.t <= D.Select(dv.i).Select(dv.k) for dv in DV_set])
    eq5 = And([dv.t <= dv.s + F.Select(dv.i).Select(dv.k) for dv in DV_set])
    #eq3_1 = And([Tv.Select(dv.e).Select(dv.d).Select(dv.k) >= 0])

    # Pinpoint constraint
    #eq6 = And([R.Select(dv.i).Select(dv.d).Select(dv.t).Equals(Int(1)) for dv in DV_set])
    eq6 = And([R.Select(dv.i).Select(dv.t).Equals(dv.d) for dv in DV_set])


    # Memory constraint
    eq7 = And(Bool(True))
    memory_sums = [] # for objective function
    for t in range(0, T_MAX):
        memory_sums.append(Int(0))
        for j in range(0, J):
            memory_sum = Int(0) # for a single server at time t
            for dv in DV_set:
                indicator = Symbol(symbol_name("ind", j, t, dv.i_int, dv.k_int), INT)
                existence_formula = And(dv.d.Equals(j), t >= dv.s + C.Select(dv.e).Select(dv.k) + Tv.Select(dv.e).Select(dv.d).Select(dv.k), t < dv.t)
                # print(existence_formula.Iff(indicator.Equals(1)))
                eq7 = eq7.And(Not(existence_formula).Iff(indicator.Equals(0)))
                eq7 = eq7.And(existence_formula.Iff(indicator.Equals(1)))
                memory_sum += M[dv.k_int] * indicator
            eq7 = eq7.And(memory_sum <= M_bar[j])
            memory_sums[-1] += memory_sum
            # print(memory_sum)
    # print(eq7)
    # print(memory_sums[0])
        
    # memory_sums[t] == the sum of memory usage of all server at time t
    



    # # Decision version
    # # Objective constraint
    # obj = Max(memory_sums) <= 4
    # model = get_model(And(domain, eq1, eq2, eq3, eq4, eq5, eq6, eq7, obj))
    # print(model)

    # Optimization version
    # Minimize the maximum memory used
    constraints = And(domain, eq1, eq2, eq3, eq4, eq5, eq6, eq7)
    if(not is_sat(And(constraints, Max(memory_sums) <= sum(M_bar)), solver_name="z3")):
        print("Unsat")
    else:
        if(is_sat(And(constraints, Max(memory_sums) <= 0), solver_name="z3")):
            print(f"Objective value = 0")
            #print(get_model(And(constraints, Max(memory_sums) <= 0)))
        else:
            l = 1
            r = sum(M_bar) + 1
            while l < r:
                # print(f"{l}, {r}")
                m = int((l + r) / 2)
                if is_sat(And(constraints, Max(memory_sums) <= m), solver_name="z3"):
                    # print(m)
                    r = m
                else:
                    l = m + 1
            print(f"Objective value = {l}")
            #print(get_model(And(constraints, Max(memory_sums) <= m)))


    # print(is_sat(And(domain, eq1, eq2, eq3, eq4, eq5, eq6, eq7, obj)))
    running_time = time.time() - start_time
    outfile = open("time.out", "a")
    outfile.write(f"{running_time}")
    outfile.write("\n")

    #outfile = open("sol.out", "a")
    #outfile.write(f"{sol}")
    #outfile.write("\n")

if __name__ == "__main__":
    main()