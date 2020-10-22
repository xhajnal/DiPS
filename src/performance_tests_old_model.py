import os
import datetime
import math
import time
import pickle
import socket

## Importing my code
from common.convert import ineq_to_constraints
from common.files import pickle_dump, pickle_load
from refine_space import check_deeper
from load import get_f
from common.mathematics import create_intervals


cwd = os.getcwd()
test = os.path.join(cwd, "../performance_test")

## LOAD DATA
start_time = time.time()
D3 = pickle_load(os.path.join(test, "data/Data_two_param"))
print(f"  It took {socket.gethostname()} {time.time() - start_time} seconds to load data")

## LOAD POLYNOMIALS
start_time = time.time()

populations = [2, 3, 5, 10]
functions = {}
for population_size in populations:
    functions[population_size] = get_f(os.path.join(test, f"data/synchronous_{population_size}.txt"), "prism", True)
print(f"  It took {socket.gethostname()} {time.time() - start_time} seconds to load polynomials")

## LOAD PARAMETER VALUES
p_values = sorted([0.028502714675268215, 0.45223461506339047, 0.8732745414252937, 0.6855555397734584, 0.13075717833714784])
q_values = sorted([0.5057623641293089, 0.29577906622244676, 0.8440550299528644, 0.8108008054929994, 0.03259111103419188])

## SET THE RANGE OF TESTS
alphas = [0.95]
n_sampless = [100, 1500, 3500]
depths = [12]
epsilons = [10e-6]
coverage_threshs = [0.95]

methods = [1, 2, 3]
sample_size = 2

## INITIALISATION OF THE RESULTS
results = pickle_load(os.path.join(test, "Freya_results.p"))
coverages = pickle_load(os.path.join(test, "Freya_coverages.p"))
averages_and_deviations = pickle_load(os.path.join(test, "Freya_averages_and_deviations.p"))

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
        for population_size in populations:
            for depth in depths:
                for epsilon in epsilons:
                    for coverage_thresh in coverage_threshs:
                        for method in methods:
                            for v_p in p_values:
                                for v_q in q_values:
                                    print("alpha, n_samples, max_depth, min_rect_size , population, algorithm, v_p, v_q")
                                    print(f"{alpha}, {n_samples}, {depth}, {epsilon}, {population_size}, {method}, {v_p}, {v_q}")
                                    if (alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q) in results.keys():
                                        print(f"times: {results[(alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q)]}")
                                        continue

                                    ## INITIALISATION OF THE OUTPUT STRUCTURES
                                    runs = []
                                    results[(alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q)] = []
                                    coverages[(alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q)] = []

                                    ## RUN THE REFINEMENT
                                    print("Now computing, current time is: ", datetime.datetime.now())
                                    for run in range(0, sample_size):
                                        start_time = time.time()
                                        ##      check_deeper(      region,                    constraints,                                                                                      recursion_depth, epsilon,   coverage,    silent,version, sample_size=False, debug=False, save=False, title="", where=False, show_space=True, solver="z3", delta=0.001, gui=False)
                                        space = check_deeper([(0, 1), (0, 1)], ineq_to_constraints(functions[population_size], create_intervals(alpha, n_samples, D3[("synchronous_", population_size, n_samples, v_p, v_q)])), depth, epsilon, coverage_thresh, True, method)
                                        time_foo = round(time.time() - start_time, 2)
                                        runs.append(time_foo)

                                        results[(alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q)].append(str(time_foo))
                                        coverages[(alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q)].append(round(space.get_coverage(), 4))

                                    ## COMPUTE AVERAGE RUN AND VARIATION
                                    average = sum(runs) / len(runs)
                                    deviations = []
                                    for run in range(0, sample_size):
                                        deviations.append((runs[run] - average) * (runs[run] - average))
                                    averages_and_deviations[(alpha, n_samples, depth, epsilon, population_size, method, v_p, v_q,)] = (average, math.sqrt(sum(deviations) / sample_size))

                                    ## PICKLE THE OUTPUT OF EXPERIMENTS WITH THIS SETTING SETTING
                                    pickle_dump(results, os.path.join(test, "Freya_results.p"))
                                    pickle_dump(coverages, os.path.join(test, "Freya_coverages.p"))
                                    pickle_dump(averages_and_deviations, os.path.join(test, "Freya_averages_and_deviations.p"))
