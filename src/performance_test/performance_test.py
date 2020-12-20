import multiprocessing
from datetime import datetime
from sympy import factor
from time import time
from termcolor import colored

import refine_space
from common.convert import ineq_to_constraints
from common.files import pickle_load
from common.mathematics import create_intervals
from load import load_data, get_f, parse_params_from_model
from mc import call_storm
from mc_informed import general_create_data_informed_properties
from metropolis_hastings import initialise_sampling, HastingsResults
from space import RefinedSpace
from optimize import *
import sys
import os


from common.config import load_config

spam = load_config()
data_dir = spam["data"]
model_path = spam["models"]
property_path = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
tmp = spam["tmp"]
del spam

sys.setrecursionlimit(4000000)
timeout = 3600

global debug
globals()['debug'] = True


def load_functions(path, debug):
    try:
        functions = pickle_load(os.path.join(storm_results, path))
        functions = list(map(factor, functions))
        functions = list(map(str, functions))
    except FileNotFoundError:
        try:
            functions = get_f(os.path.join(storm_results, path + ".txt"), "storm", factorize=False)
            functions = list(map(factor, functions))
            functions = list(map(str, functions))
        except FileNotFoundError:
            try:
                functions = pickle_load(os.path.join(prism_results, path))
                functions = list(map(factor, functions))
                functions = list(map(str, functions))
            except FileNotFoundError:
                functions = get_f(os.path.join(prism_results, path + ".txt"), "prism", factorize=True)

    print(colored(f"Loaded function file: {path}", "blue"))
    if debug:
        # print("functions", functions)
        print("functions[0]", functions[0])
    else:
        print("functions[0]", functions[0])
    return functions


def refine(text, parameters, parameter_domains, constraints, timeout):
    max_depth = 15
    coverage = 0.95
    alg = 3
    epsilon = 0
    print(colored(
        f"Refining, {text}", "yellow"))
    print("Now computing, current time is: ", datetime.now())
    print("max_depth", max_depth, "coverage", coverage)
    space = RefinedSpace(parameter_domains, parameters)
    spam = refine_space.check_deeper(space, constraints[i], max_depth, epsilon=epsilon, coverage=coverage,
                                     silent=True, version=alg, sample_size=0, debug=False, save=False,
                                     solver="z3", delta=0.01, gui=False, show_space=False,
                                     iterative=False, timeout=timeout)
    print("coverage reached", spam.get_coverage())
    if debug:
        print("refined space", spam.nice_print())


if __name__ == '__main__':
    # ## CROWDS
    # model_file = os.path.join(model_path, f"crowds.pm")
    # consts, parameters = parse_params_from_model(model_file, silent=True)
    # property_file = os.path.join(property_path, f"crowds.pctl")
    # if debug:
    #     print("parameters", parameters)
    #
    # ## SETUP PARAMETER AND THEIR DOMAINS
    # parameter_domains = []
    # for item in parameters:
    #     parameter_domains.append([0, 1])
    # if debug:
    #     print("parameter_domains", parameter_domains)
    #
    # ## LOAD FUNCTIONS
    # functions = load_functions("crowds", debug)
    #
    # data_sets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # for data_set in data_sets:
    #     data_set = [data_set]
    #     if debug:
    #         print("data_set", data_set)
    #         print(type(data_set[0]))
    #
    #     ## COMPUTE INTERVALS
    #     C = 0.95
    #     n_samples = [100, 1500, 3500]
    #
    #     ## TODO more indices of n_samples
    #     i = 0
    #     intervals = []
    #     for i in range(len(n_samples)):
    #         intervals.append(create_intervals(float(C), int(n_samples[i]), data_set))
    #         if debug:
    #             print(f"Intervals, confidence level {C}, n_samples {n_samples[i]}: {intervals[i]}")
    #
    #     ## OPTIMIZE PARAMS
    #     start_time = time()
    #     result_1 = optimize(functions, parameters, parameter_domains, data_set)
    #     print(colored(
    #         f"Optimisation, data {data_set}, took {round(time() - start_time, 4)} seconds", "yellow"))
    #     if debug:
    #         print(result_1)
    #
    #     constraints = []
    #     for i in range(len(n_samples)):
    #         constraints.append(ineq_to_constraints(functions, intervals[i], decoupled=True))
    #         if debug:
    #             print(f"constraints with {n_samples[i]} samples :{constraints[i]}")
    #
    #     ## SAMPLE SPACE
    #     grid_size = 25
    #     for i in range(len(n_samples)):
    #         space = RefinedSpace(parameter_domains, parameters)
    #         start_time = time()
    #         sampling = space.grid_sample(constraints[i], grid_size, silent=True, debug=debug)
    #         print(colored(
    #             f"Sampling, dataset {data_set}, # of samples {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
    #         if debug:
    #             print(sampling)
    #
    #     ## REFINE SPACE
    #     for i in range(len(n_samples)):
    #         refine(f"dataset {data_set}, # of samples {n_samples[i]}", parameters, parameter_domains, constraints, timeout)
    #
    #     ## METROPOLIS-HASTINGS
    #     space = RefinedSpace(parameter_domains, parameters)
    #     iterations = 500000
    #     ## TODO more indices of n_samples
    #     for i in range(len(n_samples)):
    #         start_time = time()
    #         mh_results = initialise_sampling(space, data_set, functions, n_samples[i], iterations, 0, where=True, metadata=False, timeout=timeout)
    #         assert isinstance(mh_results, HastingsResults)
    #         print(colored(
    #             f"this was MH, dataset {data_set}, # of samples, {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
    #         if debug:
    #             print("# of accepted points", len(mh_results.accepted))
    #             if mh_results.last_iter > 0:
    #                 print("current iteration", mh_results.last_iter)
    #                 iter_time = (time() - start_time) * (iterations / mh_results.last_iter)
    #                 print("time it would take to finish", iter_time)
    #             print()

        # ## LIFTING WITH STORM HERE
        # for i in range(len(n_samples)):
        #     general_create_data_informed_properties(property_file, intervals[i], silent=False)
        #     call_storm(model_file=model_file, params=parameters, param_intervals=parameter_domains,
        #                property_file=property_file, storm_output_file=f"tmp_storm_pop_size_{population_size}_data_{data_index}_nsamples_{n_samples[i]}.txt",
        #                time=True, silent=False)

    ## semisynchronous models
    for multiparam in [False, True]:                                                ## [False, True]
        for population_size in [15]:                                          ## [2, 3, 4, 5, 10, 15]
            for data_index, data_dir_subfolder in enumerate(["data", "data_1"]):    ## ["data", "data_1"]
                if multiparam:
                    params = "multiparam"
                    prefix = "multiparam_"
                else:
                    params = "2-param"
                    prefix = ""

                ## SETUP PARAMETER AND THEIR DOMAINS
                try:
                    model_file = os.path.join(model_path, f"semisynchronous/{params}/{population_size}_{prefix}semisynchronous.pm")
                    parameters = parse_params_from_model(model_file, silent=True)[1]
                except FileNotFoundError as err:
                    print(colored(f"model {population_size}_{prefix}semisynchronous.pm not found, skipping this test", "red"))
                    break
                property_file = os.path.join(property_path, f"prop_{population_size}.pctl")

                if debug:
                    print("parameters", parameters)
                parameter_domains = []
                for item in parameters:
                    parameter_domains.append([0, 1])
                if debug:
                    print("parameter_domains", parameter_domains)

                ## LOAD FUNCTIONS
                functions = load_functions(f"{params}/semisynchronous/{prefix}semisynchronous_{population_size}", debug)

                ## LOAD DATA
                data_set = load_data(os.path.join(data_dir, f"{data_dir_subfolder}/{params}/data_n={population_size}.csv"), debug=debug)
                if debug:
                    print("data_set", data_set)
                    print(type(data_set[0]))
                # data_set_2 = load_data(os.path.join(data_dir, f"data_1/data_n={population_size}.csv"))
                # print("data_set_2", data_set_2)

                ## COMPUTE INTERVALS
                C = 0.95
                n_samples = [3500, 1500]  ## [100, 1500, 3500]

                ## TODO more indices of n_samples
                i = 0
                intervals = []
                for i in range(len(n_samples)):
                    intervals.append(create_intervals(float(C), int(n_samples[i]), data_set))
                    if debug:
                        print(f"Intervals, confidence level {C}, n_samples {n_samples[i]}: {intervals[i]}")

                ## OPTIMIZE PARAMS
                start_time = time()
                result_1 = optimize(functions, parameters, parameter_domains, data_set)
                print(colored(f"Optimisation, pop size {population_size}, dataset {data_index+1}, multiparam {bool(multiparam)}, took {round(time() - start_time, 4)} seconds", "yellow"))
                if debug:
                    print(result_1)

                constraints = []
                for i in range(len(n_samples)):
                    constraints.append(ineq_to_constraints(functions, intervals[i], decoupled=True))
                    if debug:
                        print(f"constraints with {n_samples[i]} samples :{constraints[i]}")

                # ## SAMPLE SPACE
                # grid_size = 25
                # for i in range(len(n_samples)):
                #     space = RefinedSpace(parameter_domains, parameters)
                #     start_time = time()
                #     sampling = space.grid_sample(constraints[i], grid_size, silent=True, debug=debug)
                #     print(colored(f"Sampling, pop size {population_size}, dataset {data_index+1}, multiparam {bool(multiparam)}, # of samples {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
                #     if debug:
                #         print(sampling)

                ## REFINE SPACE
                for i in range(len(n_samples)):
                    text = f"pop size {population_size}, dataset {data_index + 1}, multiparam {bool(multiparam)}, # of samples {n_samples}"
                    refine(text, parameters, parameter_domains, constraints, timeout)

                # ## METROPOLIS-HASTINGS
                # space = RefinedSpace(parameter_domains, parameters)
                # iterations = 500000
                # ## TODO more indices of n_samples
                # for i in range(len(n_samples)):
                #     start_time = time()
                #     mh_results = initialise_sampling(space, data_set, functions, n_samples[i], iterations, 0, where=True, metadata=False, timeout=timeout)
                #     assert isinstance(mh_results, HastingsResults)
                #     print(colored(f"this was MH, pop size {population_size}, dataset {data_index+1}, multiparam {bool(multiparam)}, # of samples, {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
                #     if debug:
                #         print("# of accepted points", len(mh_results.accepted))
                #         if mh_results.last_iter > 0:
                #             print("current iteration", mh_results.last_iter)
                #             iter_time = (time() - start_time) * (iterations / mh_results.last_iter)
                #             print("time it would take to finish", iter_time)
                #         print()

                # ## LIFTING WITH STORM HERE
                # for i in range(len(n_samples)):
                #     general_create_data_informed_properties(property_file, intervals[i], silent=False)
                #     call_storm(model_file=model_file, params=parameters, param_intervals=parameter_domains,
                #                property_file=property_file, storm_output_file=f"tmp_storm_pop_size_{population_size}_data_{data_index}_nsamples_{n_samples[i]}.txt",
                #                time=True, silent=False)
