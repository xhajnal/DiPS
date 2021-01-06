import subprocess
import sys
import os
import multiprocessing
from copy import copy
from datetime import datetime
from os.path import basename, isfile
from typing import Iterable
from sympy import factor
from time import time
from termcolor import colored

import refine_space
from common.convert import ineq_to_constraints
from common.files import pickle_load
from common.mathematics import create_intervals
from load import load_data, get_f, parse_params_from_model
from mc import call_storm, call_prism_files
from mc_informed import general_create_data_informed_properties
from metropolis_hastings import initialise_sampling, HastingsResults
from space import RefinedSpace
from optimize import *
from common.config import load_config

## PATHS
spam = load_config()
data_dir = spam["data"]
model_path = spam["models"]
property_path = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
refinement_results = spam["refinement_results"]
tmp = spam["tmp"]

sys.setrecursionlimit(4000000)

## SET OF RUNS
run_prism_benchmark = False
run_semisyn = True
shallow = True  ## only a single setting will be chosen from a set within the benchmark

## SET METHODS
run_optimise = True
run_sampling = True
run_refine = True
run_mh = True
run_lifting = False

## GLOBAL SETTINGS
model_checker = "storm"  ## choose from "prism", "storm", "both"

global debug
globals()['debug'] = False
global silent
globals()['silent'] = True
global factorise
globals()['factorise'] = True

## SAMPLING SETTING
grid_size = spam["grid_size"]  # 25

## INTERVALS SETTINGS
C = spam["confidence_level"]  # 0.95

## REFINEMENT SETTINGS
max_depth = spam["max_depth"]  # -1
coverage = spam["coverage"]  # 0.95
epsilon = 0
refine_timeout = spam["refine_timeout"]  # 3600

## MH SETTINGS
iterations = spam["mh_iterations"]  # 500000
mh_timeout = spam["mh_timeout"]  # 3600
del spam


def load_functions(path, factorise=False, debug=False):
    """ Loads rational functions from a file (path)

    Args
        path (Path or string): path to file
        factorise (bool): flag whether to factorise rational functions
        debug (bool): if True extensive print will be used
    """
    try:
        functions = pickle_load(os.path.join(storm_results, path))
    except FileNotFoundError:
        try:
            functions = get_f(os.path.join(storm_results, path + ".txt"), "storm", factorize=False)
        except FileNotFoundError:
            try:
                functions = pickle_load(os.path.join(prism_results, path))
            except FileNotFoundError:
                functions = get_f(os.path.join(prism_results, path + ".txt"), "prism", factorize=True)

    print(colored(f"Loaded function file: {path}", "blue"))
    if factorise:
        print(colored(f"Factorising function file: {path}", "blue"))
        functions = list(map(factor, functions))
        functions = list(map(str, functions))

    if debug:
        # print("functions", functions)
        print("functions[0]", functions[0])
    else:
        print("functions[0]", functions[0])
    return functions


def compute_functions(model_file, property_file, output_path=False, parameter_domains=False, silent=False, debug=False):
    """ Runs parametric model checking of a model (model_file) and property (property_file).
        With given model parameters domains (parameter_domains)
        It saves output, rational functions, to a file (output_path).

    Args
        model_file (string or Path): model file path
        property_file (string or Path): property file path
        output_path (string or Path): output, rational functions, file path
        parameter_domains (list of pairs): list of intervals to be used for respective parameter (default all intervals are from 0 to 1)
        silent (bool): if True output will be printed
        debug (bool): if True extensive print will be used
    """
    constants, parameters = parse_params_from_model(model_file, silent=silent, debug=debug)
    if parameter_domains is False:
        parameter_domains = []
        for param in parameters:
            parameter_domains.append([0, 1])

    if output_path is False:
        ## Using default paths
        prism_output_path = prism_results
        storm_output_path = storm_results
    elif isinstance(output_path, Iterable):
        prism_output_path = output_path[0]
        storm_output_path = output_path[0]
    else:
        prism_output_path = output_path
        storm_output_path = output_path
    storm_output_file = os.path.join(storm_output_path, basename(model_file))
    storm_output_file = str(storm_output_file.split(".")[0]) + ".txt"

    if model_checker == "prism" or model_checker == "both":
        call_prism_files(model_file, [], param_intervals=parameter_domains, seq=False, no_prob_checks=False,
                         memory="", model_path="", properties_path=property_path,
                         property_file=property_file, output_path=prism_output_path,
                         gui=False, silent=silent)
    if model_checker == "storm" or model_checker == "both":
        call_storm(model_file=model_file, params=[], param_intervals=parameter_domains,
                   property_file=property_file, storm_output_file=storm_output_file,
                   time=True, silent=silent)


def refine(text, parameters, parameter_domains, constraints, timeout, silent, debug, alg=4):
    """ Runs space refinement

    Args
        text (string): text to be printed
        parameters (list of strings): list of model parameters
        parameter_domains (list of pairs): list of intervals to be used for respective parameter
        constraints  (list of strings): array of properties
        timeout (int): timeout in seconds (set 0 for no timeout)
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        alg (Int): version of the algorithm to be used
    """
    print(colored(
        f"Refining, {text}", "yellow"))
    print("Now computing, current time is: ", datetime.now())
    print("max_depth", max_depth, "coverage", coverage)
    space = RefinedSpace(parameter_domains, parameters)
    spam = refine_space.check_deeper(space, constraints[i], max_depth, epsilon=epsilon, coverage=coverage,
                                     silent=silent, version=alg, sample_size=0, debug=debug, save=False,
                                     solver="z3", delta=0.01, gui=False, show_space=False,
                                     iterative=False, timeout=timeout)
    print("coverage reached", spam.get_coverage())
    if debug:
        print("refined space", spam.nice_print())


if __name__ == '__main__':
    ## PRISM BENCHMARKS
    test_cases = ["crowds", "brp", "nand"]  ## ["crowds", "brp", "nand"]
    for test_case in test_cases:
        ## Skip PRISM BENCHMARKS?
        if not run_prism_benchmark:
            break
        if test_case == "crowds":
            if shallow:
                settings = [[3, 5]]
            else:
                settings = [[3, 5], [5, 5], [10, 5], [15, 5], [20, 5]]
        if test_case == "brp":
            if shallow:
                settings = [[16, 2]]
            else:
                settings = [[16, 2], [128, 2], [128, 5], [256, 2], [256, 5]]
        if test_case == "nand":
            if shallow:
                settings = [[10, 1]]
            else:
                settings = [[10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5]]

        for constants in settings:
            model_name = f"{test_case}_{constants[0]}-{constants[1]}"
            model_file = os.path.join(model_path, test_case, f"{model_name}.pm")
            consts, parameters = parse_params_from_model(model_file, silent=True)
            property_file = os.path.join(property_path, test_case, f"{test_case}.pctl")
            if debug:
                print("parameters", parameters)

            ## SETUP PARAMETER AND THEIR DOMAINS
            parameter_domains = []
            for item in parameters:
                parameter_domains.append([0, 1])
            if debug:
                print("parameter_domains", parameter_domains)

            ## LOAD FUNCTIONS
            try:
                functions = load_functions(model_name, factorise=False, debug=debug)
            except FileNotFoundError as err:
                ## compute functions
                compute_functions(model_file, property_file, output_path=False, debug=debug)
                # functions = load_functions(model_name, debug)

            if shallow:
                data_sets = [0.5]
            else:
                data_sets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            for data_set in data_sets:
                data_set = [data_set]
                if debug:
                    print("data_set", data_set)
                    print(type(data_set[0]))

                ## COMPUTE INTERVALS
                if shallow:
                    n_samples = [100]
                else:
                    n_samples = [100, 1500, 3500]

                ## TODO more indices of n_samples
                i = 0
                intervals = []
                for i in range(len(n_samples)):
                    intervals.append(create_intervals(float(C), int(n_samples[i]), data_set))
                    if debug:
                        print(f"Intervals, confidence level {C}, n_samples {n_samples[i]}: {intervals[i]}")

                ## OPTIMIZE PARAMS
                if run_optimise:
                    start_time = time()
                    result_1 = optimize(functions, parameters, parameter_domains, data_set)
                    print(colored(
                        f"Optimisation, data {data_set}, took {round(time() - start_time, 4)} seconds", "yellow"))
                    if debug:
                        print(result_1)

                constraints = []
                for i in range(len(n_samples)):
                    constraints.append(ineq_to_constraints(functions, intervals[i], decoupled=True))
                    if debug:
                        print(f"Constraints with {n_samples[i]} samples :{constraints[i]}")

                ## SAMPLE SPACE
                for i in range(len(n_samples)):
                    if not run_sampling:
                        break
                    space = RefinedSpace(parameter_domains, parameters)
                    start_time = time()
                    sampling = space.grid_sample(constraints[i], grid_size, silent=True, debug=debug)
                    print(colored(
                        f"Sampling, dataset {data_set}, # of samples {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
                    if debug:
                        print(sampling)

                ## REFINE SPACE
                if not run_refine:
                    break
                for i in range(len(n_samples)):
                    ## TODO this fails
                    # refine(f"dataset {data_set}, # of samples {n_samples[i]}", parameters, parameter_domains, constraints, timeout, silent, debug, alg=5)
                    refine(f"dataset {data_set}, # of samples {n_samples[i]}", parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=4)
                    refine(f"dataset {data_set}, # of samples {n_samples[i]}", parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=3)

                ## METROPOLIS-HASTINGS
                # space = RefinedSpace(parameter_domains, parameters)
                ## TODO more indices of n_samples
                for i in range(len(n_samples)):
                    if not run_mh:
                        break
                    start_time = time()
                    mh_results = initialise_sampling(parameters, parameter_domains, data_set, functions, n_samples[i], iterations, 0, where=True, metadata=False, timeout=mh_timeout)
                    assert isinstance(mh_results, HastingsResults)
                    print(colored(
                        f"this was MH, {iterations} iterations, dataset {data_set}, # of samples, {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
                    if debug:
                        print("# of accepted points", len(mh_results.accepted))
                        if mh_results.last_iter > 0:
                            print("current iteration", mh_results.last_iter)
                            iter_time = (time() - start_time) * (iterations / mh_results.last_iter)
                            print("time it would take to finish", iter_time)
                        print()

                ## LIFTING WITH STORM HERE
                storm_parameter_domains = copy(parameter_domains)
                storm_parameter_domains = list(map(lambda x: (x[0] + 0.000001, x[1] - 0.000001), storm_parameter_domains))
                for i in range(len(n_samples)):
                    if not run_lifting:
                        break
                    spam = general_create_data_informed_properties(property_file, intervals[i], silent=False)
                    conf = str(C).replace(".", ",")
                    data_sett = str(*data_set).replace(".", ",")
                    save_informed_property_file = os.path.join(results_dir, "data-informed_properties", str(basename(property_file).split(".")[0]) + f"{n_samples[i]}_samples_{conf}_confidence" + ".pctl")
                    try:
                        os.mkdir(os.path.join(results_dir, "data-informed_properties"))
                    except FileExistsError as err:
                        pass

                    with open(save_informed_property_file, "w") as f:
                        for item in spam:
                            f.write(item + "\n")

                    try:
                        os.mkdir(os.path.join(refinement_results, model_name))
                    except FileExistsError as err:
                        pass

                    call_storm(model_file=model_file, params=parameters, param_intervals=storm_parameter_domains,
                               property_file=save_informed_property_file, storm_output_file=os.path.join(refinement_results, model_name, f"{model_name}_refined_{data_sett}_data_set_{n_samples[i]}_samples_{conf}_confidence.txt"),
                               time=True, silent=False)

    ## Semisynchronous models
    for multiparam in [False, True]:                                                ## [False, True]
        for population_size in [2]:                                                 ## [2, 3, 4, 5, 10, 15]
            for data_index, data_dir_subfolder in enumerate(["data", "data_1"]):    ## ["data", "data_1"]
                ## SKIP Semisynchronous models
                if not run_semisyn:
                    break
                if shallow and data_index > 0:
                    continue

                if multiparam:
                    params = "multiparam"
                    prefix = "multiparam_"
                else:
                    params = "2-param"
                    prefix = ""

                ## LOAD MODELS AND PROPERTIES
                try:
                    test_case = "bee"
                    # model_name = f"semisynchronous/{prefix}semisynchronous_{population_size}"
                    model_name = f"{prefix}semisynchronous_{population_size}_bees"
                    model_file = os.path.join(model_path, test_case, model_name + ".pm")
                    parameters = parse_params_from_model(model_file, silent=True)[1]
                except FileNotFoundError as err:
                    print(colored(f"model {model_name} not found, skipping this test", "red"))
                    if debug:
                        print(err)
                    break
                property_file = os.path.join(property_path, test_case, f"prop_{population_size}_bees.pctl")
                if not isfile(property_file):
                    with open(property_file, "r") as file:
                        pass

                ## SETUP PARAMETERS AND THEIR DOMAINS
                if debug:
                    print("parameters", parameters)
                parameter_domains = []
                for item in parameters:
                    parameter_domains.append([0, 1])
                if debug:
                    print("parameter_domains", parameter_domains)

                ## LOAD FUNCTIONS
                try:
                    functions = load_functions(model_name, debug=debug)
                except FileNotFoundError as err:
                    ## Compute functions
                    compute_functions(model_file, property_file, output_path=False, debug=debug)
                    functions = load_functions(model_name, debug=debug)

                ## LOAD DATA
                data_set = load_data(os.path.join(data_dir, f"{data_dir_subfolder}/{params}/data_n={population_size}.csv"), debug=debug)
                if debug:
                    print("data_set", data_set)
                    print(type(data_set[0]))
                # data_set_2 = load_data(os.path.join(data_dir, f"data_1/data_n={population_size}.csv"))
                # print("data_set_2", data_set_2)

                ## COMPUTE INTERVALS
                if shallow:
                    n_samples = [3500]
                else:
                    n_samples = [100, 1500, 3500]  ## [100, 1500, 3500]

                ## TODO more indices of n_samples
                i = 0
                intervals = []
                for i in range(len(n_samples)):
                    intervals.append(create_intervals(float(C), int(n_samples[i]), data_set))
                    if debug:
                        print(f"Intervals, confidence level {C}, n_samples {n_samples[i]}: {intervals[i]}")

                ## OPTIMIZE PARAMS
                if run_optimise:
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

                ## SAMPLE SPACE
                for i in range(len(n_samples)):
                    if not run_sampling:
                        break
                    space = RefinedSpace(parameter_domains, parameters)
                    start_time = time()
                    sampling = space.grid_sample(constraints[i], grid_size, silent=True, debug=debug)
                    print(colored(f"Sampling, pop size {population_size}, dataset {data_index+1}, multiparam {bool(multiparam)}, # of samples {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
                    if debug:
                        print(sampling)

                ## REFINE SPACE
                for i in range(len(n_samples)):
                    if not run_refine:
                        break
                    text = f"pop size {population_size}, dataset {data_index + 1}, multiparam {bool(multiparam)}, # of samples {n_samples[i]}"
                    refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug)

                ## METROPOLIS-HASTINGS
                # space = RefinedSpace(parameter_domains, parameters)
                ## TODO more indices of n_samples
                for i in range(len(n_samples)):
                    if not run_mh:
                        break
                    start_time = time()
                    mh_results = initialise_sampling(parameters, parameter_domains, functions, data_set, n_samples[i], iterations, 0, where=True, metadata=False, timeout=mh_timeout)
                    assert isinstance(mh_results, HastingsResults)
                    print(colored(f"this was MH, {iterations} iterations, pop size {population_size}, dataset {data_index+1}, multiparam {bool(multiparam)}, # of samples, {n_samples[i]} took {round(time() - start_time, 4)} seconds", "yellow"))
                    if debug:
                        print("# of accepted points", len(mh_results.accepted))
                        if mh_results.last_iter > 0:
                            print("current iteration", mh_results.last_iter)
                            iter_time = (time() - start_time) * (iterations / mh_results.last_iter)
                            print("time it would take to finish", iter_time)
                        print()

                ## LIFTING WITH STORM HERE
                for i in range(len(n_samples)):
                    if not run_lifting:
                        break
                    general_create_data_informed_properties(property_file, intervals[i], silent=False)
                    call_storm(model_file=model_file, params=parameters, param_intervals=parameter_domains,
                               property_file=property_file, storm_output_file=f"tmp_storm_pop_size_{population_size}_data_{data_index}_nsamples_{n_samples[i]}.txt",
                               time=True, silent=False)
