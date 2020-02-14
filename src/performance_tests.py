import os
import datetime
import math
import time
import pickle
import socket

## Importing my code
from common.convert import ineq_to_constraints
from refine_space import check_deeper
from load import get_all_f, load_pickled_data
from common.mathematics import create_intervals

## LOAD DATA
start_time = time.time()
D3 = load_pickled_data("Data_two_param")
print(f"  It took {socket.gethostname()} {time.time() - start_time} seconds to load data")

## LOAD POLYNOMIALS
start_time = time.time()
functions = get_all_f("./sem*[0-9].txt", "prism", True)
print(f"  It took {socket.gethostname()} {time.time() - start_time} seconds to load polynomials")

## GET TO THE RIGHT DIRECTORY
os.chdir("..")
try:
    os.mkdir("performance_results")
except:
    print("Folder performance_results probably already exists")
os.chdir("performance_results")

## LOAD PARAMETER VALUES
p_values = sorted([0.028502714675268215, 0.45223461506339047, 0.8732745414252937, 0.6855555397734584, 0.13075717833714784])
q_values = sorted([0.5057623641293089, 0.29577906622244676, 0.8440550299528644, 0.8108008054929994, 0.03259111103419188])

## SET THE RANGE OF TESTS
alphas = [0.95]
n_sampless = [100, 1500, 3500]
depths = [12]
epsilons = [10e-6]
coverage_threshs = [0.95]
populations = [2, 3, 5, 10]
methods = [1, 2, 3]
sample_size = 2

## INITIALISATION OF THE RESULTS
results = pickle.load(open("results.p", "rb"))
coverages = pickle.load(open("coverages.p", "rb"))
averages_and_deviations = pickle.load(open("averages_and_deviations.p", "rb"))

if not results:
    raise OSError("results.p is empty")

## THESE SETTINGS WENT TIME OUT
results[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.29577906622244676)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.29577906622244676)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.29577906622244676)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.03259111103419188)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.03259111103419188)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.03259111103419188)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.8108008054929994)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.8108008054929994)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.028502714675268215, 0.8108008054929994)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.13075717833714784, 0.5057623641293089)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.13075717833714784, 0.5057623641293089)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.13075717833714784, 0.5057623641293089)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.03259111103419188)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.03259111103419188)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.03259111103419188)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.8108008054929994)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.8108008054929994)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.8108008054929994)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.8440550299528644)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.8440550299528644)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.45223461506339047, 0.8440550299528644)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 1, 0.6855555397734584, 0.29577906622244676)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 1, 0.6855555397734584, 0.29577906622244676)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 1, 0.6855555397734584, 0.29577906622244676)] = ["TO", "TO"]

results[(0.95, 100, 12, 1e-05, 5, 2, 0.45223461506339047, 0.8108008054929994)] = ["TO", "TO"]
coverages[(0.95, 100, 12, 1e-05, 5, 2, 0.45223461506339047, 0.8108008054929994)] = ["TO", "TO"]
averages_and_deviations[(0.95, 100, 12, 1e-05, 5, 2, 0.45223461506339047, 0.8108008054929994)] = ["TO", "TO"]

results[(0.95, 1500, 12, 1e-05, 10, 1, 0.45223461506339047, 0.29577906622244676)] = ["TO", "TO"]
coverages[(0.95, 1500, 12, 1e-05, 10, 1, 0.45223461506339047, 0.29577906622244676)] = ["TO", "TO"]
averages_and_deviations[(0.95, 1500, 12, 1e-05, 10, 1, 0.45223461506339047, 0.29577906622244676)] = ["TO", "TO"]

results[(0.95, 1500, 12, 1e-05, 10, 1, 0.45223461506339047, 0.5057623641293089)] = ["TO", "TO"]
coverages[(0.95, 1500, 12, 1e-05, 10, 1, 0.45223461506339047, 0.5057623641293089)] = ["TO", "TO"]
averages_and_deviations[(0.95, 1500, 12, 1e-05, 10, 1, 0.45223461506339047, 0.5057623641293089)] = ["TO", "TO"]

## RUN THE PERFORMANCE TEST
now = datetime.datetime.now()
print("semisynchronous/synchronous")
for alpha in alphas:
    for n_samples in n_sampless:
        for N in populations:
            for depth in depths:
                for epsilon in epsilons:
                    for coverage_thresh in coverage_threshs:
                        for method in methods:
                            for v_p in p_values:
                                for v_q in q_values:
                                    print("alpha, n_samples, max_depth, min_rect_size , population, algorithm, v_p, v_q")
                                    print(f"{alpha}, {n_samples}, {depth}, {epsilon}, {N}, {method}, {v_p}, {v_q}")
                                    if (alpha, n_samples, depth, epsilon, N, method, v_p, v_q) in results.keys():
                                        print(f"times: {results[(alpha, n_samples, depth, epsilon, N, method, v_p, v_q)]}")
                                        continue

                                    ## INITIALISATION OF THE OUTPUT STRUCTURES
                                    runs = []
                                    results[(alpha, n_samples, depth, epsilon, N, method, v_p, v_q)] = []
                                    coverages[(alpha, n_samples, depth, epsilon, N, method, v_p, v_q)] = []

                                    ## RUN THE REFINEMENT
                                    print("Now computing, current time is: ", datetime.datetime.now())
                                    for run in range(0, sample_size):
                                        start_time = time.time()
                                        ##      check_deeper(      region,                    constraints,                                                                                         recursion_depth, epsilon,   coverage,    silent,version, sample_size=False, debug=False, save=False, title="", where=False, show_space=True, solver="z3", delta=0.001, gui=False)
                                        space = check_deeper([(0, 1), (0, 1)], ineq_to_constraints(functions[N], create_intervals(alpha, n_samples, D3[("synchronous_parallel_", N, n_samples, v_p, v_q)])), depth, epsilon, coverage_thresh, True, method)
                                        time_foo = round(time.time() - start_time, 2)
                                        runs.append(time_foo)

                                        results[(alpha, n_samples, depth, epsilon, N, method, v_p, v_q)].append(str(time_foo))
                                        coverages[(alpha, n_samples, depth, epsilon, N, method, v_p, v_q)].append(round(space.get_coverage(), 4))

                                    ## COMPUTE AVERAGE RUN AND VARIATION
                                    average = sum(runs) / len(runs)
                                    deviations = []
                                    for run in range(0, sample_size):
                                        deviations.append((runs[run] - average) * (runs[run] - average))
                                    averages_and_deviations[(alpha, n_samples, depth, epsilon, N, method, v_p, v_q,)] = (average, math.sqrt(sum(deviations) / sample_size))

                                    ## PICKLE THE OUTPUT OF EXPERIMENTS WITH THIS SETTING SETTING
                                    pickle.dump(results, open("results.p", 'wb'))
                                    pickle.dump(coverages, open("coverages.p", 'wb'))
                                    pickle.dump(averages_and_deviations, open("averages_and_deviations.p", 'wb'))
