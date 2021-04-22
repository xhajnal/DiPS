import sys
import os
from termcolor import colored
from performance_test import repeat_sampling, load_functions, repeat_refine
from common.convert import ineq_to_constraints
from common.mathematics import create_proportions_interval, create_intervals_hsb
from load import load_data
from common.model_stuff import parse_params_from_model
from space import RefinedSpace
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

model_checker = "storm"  # "either"

## PATHS
sys.setrecursionlimit(4000000)

debug = False
silent = True
factorise = True

show_space = False

## REFINEMENT SETTING
timeout = 3600
single_call_timeout = 0  ## 30 or 0
default_alg = 2

## INTERVALS SETTINGS
n_samples_list = [100, 3500]
C = 0.95

## EXPERIMENT SETUP
cores_list = [True, 8, 4, 2, 1, False, "False4"]  ## [True, 8, 4, 2, 1, False]  ## This is CPU depending setting
precision = 4
repetitions = 20

# sample_guided = True  ## Set True to run sampling-guided version, False for map version
# is_async = False  ## Set True to run asynchronous version, False for map version

## END OF SETTINGS
del spam
if show_space:
    where = False
else:
    where = None

if __name__ == '__main__':
    for n_samples in n_samples_list:
        for multiparam in [True]:  ## [False, True]
            for population_size in [4]:  ## [2, 3, 5, 10]
                for is_async in [True]:  ## [False, True]
                    for sample_guided in [True]:  ## [False, True]
                        if multiparam and population_size == 2:
                            print(colored("Skipping 2 bees multiparam as this model is equivalent to 2-param model", "red"))
                            continue
                        if not multiparam and population_size == 4:
                            print(colored("Skipping 4 bees 2-param. We go straight to 5 bees.", "red"))
                            continue
                        if multiparam and population_size == 10:
                            print(colored("Skipping 10 bees multiparam. This will almost certainly not scale.", "red"))
                            continue

                        print(colored(f"Population size {population_size}", "blue"))
                        if population_size != 0:
                            ## LOAD FUNCTIONS
                            if multiparam:
                                functions = load_functions(f"bee/multiparam_semisynchronous_{population_size}_bees", debug=debug, source=model_checker)
                            else:
                                functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug, source=model_checker)

                            ## LOAD DATA
                            if multiparam:
                                data_set = load_data(os.path.join(data_dir, f"bee/multiparam/data_n={population_size}.csv"), debug=debug)
                            else:
                                data_set = load_data(os.path.join(data_dir, f"bee/2-param/data_n={population_size}.csv"), debug=debug)

                        ## COMPUTE INTERVALS
                        n_samples_list = [n_samples]  # [100, 1500, 3500]

                        ## SETUP PARAMETERS AND THEIR DOMAINS
                        if multiparam:
                            parameters = parse_params_from_model(os.path.join(model_path, f"bee/multiparam_semisynchronous_{population_size}_bees.pm"), silent=silent)[1]
                        else:
                            parameters = parse_params_from_model(os.path.join(model_path, f"bee/semisynchronous_{population_size}_bees.pm"), silent=silent)[1]
                        print("model parameters: ", parameters)

                        if debug:
                            print("parameters", parameters)
                        parameter_domains = []
                        for item in parameters:
                            parameter_domains.append([0, 1])
                        if debug:
                            print("parameter_domains", parameter_domains)

                        if population_size != 0:
                            ## COMPUTE CONFIDENCE INTERVALS
                            ## Intervals with adjusted Wald method used in Hajnal et al. hsb 2019
                            intervals = create_intervals_hsb(float(C), int(n_samples), data_set)
                            ## THIS IS AGRESTI-COULL METHOD
                            # intervals = [create_proportions_interval(float(C), int(n_samples), data_point, method="AC") for data_point in data_set]
                            if population_size == 2:
                                print(intervals)
                            constraints = ineq_to_constraints(functions, intervals, decoupled=True)
                        else:
                            parameters = ["p", "q"]
                            parameter_domains = [[0, 1], [0, 1]]
                            constraints = ["q+p >= 1.0", "q+p <= 9.0"]
                            data_set = None

                        ## REFINE SPACE
                        for cores in cores_list:
                            if cores == "False4":
                                alg = 4
                                alg2 = None
                                cores = False
                            else:
                                alg = default_alg
                                alg2 = 5

                            print(f"parallel: {cores},{' async calls,' if is_async else ' map calls,'} sample guided: {sample_guided}, single call timeout: {single_call_timeout}")
                            text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                            space = RefinedSpace(parameter_domains, parameters)

                            spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                                 single_call_timeout=single_call_timeout, debug=debug, alg=alg, solver="z3",
                                                 sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                                 parallel=cores, is_async=is_async)

                            spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                                 single_call_timeout=single_call_timeout, debug=debug, alg=alg, solver="dreal",
                                                 sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                                 parallel=cores, is_async=is_async)
                            if multiparam:
                                print(colored("This will probably not halt within reasonable time.", "yellow"))

                            spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                                 single_call_timeout=single_call_timeout, debug=debug, alg=alg2, solver="z3",
                                                 sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                                 parallel=cores, is_async=is_async)
                            print()

        ### Knuth die
        ## LOAD FUNCTIONS
        for param_count in [1, 2, 3]:
            functions = load_functions(f"Knuth/parametric_die_{param_count}_param_BSCCs", debug=debug, source=model_checker)

            ## LOAD DATA
            data_set = load_data(os.path.join(data_dir, f"knuth_3params_generated_data_p1_0,4_p2_0,7_p3_0,5_1000_samples.p"), debug=debug)
            # print(data_set)
            # >> [0.208, 0.081, 0.1, 0.254, 0.261, 0.096]

            ## SETUP PARAMETERS AND THEIR DOMAINS
            if param_count == 1:
                parameters = ["p"]
            elif param_count == 2:
                parameters = ["p1", "p2"]
            elif param_count == 3:
                parameters = ["p1", "p2", "p3"]

            if debug:
                print("parameters", parameters)

            parameter_domains = []
            for item in parameters:
                parameter_domains.append([0, 1])
            if debug:
                print("parameter_domains", parameter_domains)

            ## COMPUTE CONFIDENCE INTERVALS
            ## Intervals with adjusted Wald method used in Hajnal et al. hsb 2019
            intervals = create_intervals_hsb(float(C), int(n_samples), data_set)
            ## THIS IS AGRESTI-COULL METHOD
            # intervals = [create_proportions_interval(float(C), int(n_samples), data_point, method="AC") for data_point in data_set]
            constraints = ineq_to_constraints(functions, intervals, decoupled=True)

            ## REFINE SPACE
            for sample_guided in [False, True]:  ## [False, True]
                for is_async in [True]:  ## [False, True]
                    for cores in cores_list:
                        if cores == "False4":
                            alg = 4
                            alg2 = None
                            cores = False
                        else:
                            alg = default_alg
                            alg2 = 5

                        print(f"parallel: {cores},{' async calls,' if is_async else ' map calls,'} sample guided: {sample_guided}, single call timeout: {single_call_timeout}")
                        text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                        space = RefinedSpace(parameter_domains, parameters)

                        alg = default_alg

                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=alg, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=alg, solver="dreal",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=alg2, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        print()

        ### bounded retransmission protocol (brp model)
        for version in ["3-2", "16-2"]:
            ## LOAD FUNCTIONS
            functions = load_functions(f"brp/brp_{version}", debug=debug, source=model_checker)

            ## LOAD DATA
            data_set = [0.5]

            ## SETUP PARAMETERS AND THEIR DOMAINS
            consts, parameters = parse_params_from_model(os.path.join(model_path, f"brp/brp_{version}.pm"), silent=silent)

            if debug:
                print("parameters", parameters)
            parameter_domains = []
            for item in parameters:
                parameter_domains.append([0, 1])
            if debug:
                print("parameter_domains", parameter_domains)

            ## COMPUTE CONFIDENCE INTERVALS
            ## Intervals with adjusted Wald method used in Hajnal et al. hsb 2019
            intervals = create_intervals_hsb(float(C), int(n_samples), data_set)
            ## THIS IS AGRESTI-COULL METHOD
            # intervals = [create_proportions_interval(float(C), int(n_samples), data_point, method="AC") for data_point in data_set]
            constraints = ineq_to_constraints(functions, intervals, decoupled=True)

            ## REFINE SPACE
            for sample_guided in [False, True]:  ## [False, True]
                for is_async in [True]:  ## [False, True]
                    for cores in cores_list:
                        if cores == "False4":
                            alg = 4
                            alg2 = None
                            cores = False
                        else:
                            alg = default_alg
                            alg2 = 5

                        print(f"parallel: {cores},{' async calls,' if is_async else ' map calls,'} sample guided: {sample_guided}, single call timeout: {single_call_timeout}")
                        text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                        space = RefinedSpace(parameter_domains, parameters)

                        alg = default_alg

                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=alg, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=alg, solver="dreal",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=alg2, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        print()
