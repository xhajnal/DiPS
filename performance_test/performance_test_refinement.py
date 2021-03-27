import sys
import os
from termcolor import colored
from performance_test import repeat_sampling, load_functions, repeat_refine
from common.convert import ineq_to_constraints
from common.mathematics import create_intervals
from load import load_data, parse_params_from_model
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

## REFINEMENT SETTING
timeout = 3600
single_call_timeout = 30  ## 30 or 0

show_space = False

## INTERVALS SETTINGS
C = 0.95

## EXPERIMENT SETUP
cores_list = [True, 8, 4, 2, 1, False]  ## [True, 8, 4, 2, 1, False]
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
    for population_size in [2, 3, 5, 10]:  ## [0, 2, 3, 5, 10]
        for is_async in [False, True]:  ## [False, True]
            for sample_guided in [False, True]:  ## [False, True]
                multiparam = False
                print(colored(f"population size {population_size}", "blue"))
                if population_size != 0:
                    ## LOAD FUNCTIONS
                    functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug, source=model_checker)

                    ## LOAD DATA
                    data_set = load_data(os.path.join(data_dir, f"data/2-param/data_n={population_size}.csv"), debug=debug)

                ## COMPUTE INTERVALS
                n_samples_list = [100]  # [100, 1500, 3500]

                ## SETUP PARAMETERS AND THEIR DOMAINS
                parameters = ["p", "q"]

                if debug:
                    print("parameters", parameters)
                parameter_domains = []
                for item in parameters:
                    parameter_domains.append([0, 1])
                if debug:
                    print("parameter_domains", parameter_domains)

                for n_samples in n_samples_list:
                    if population_size != 0:
                        intervals = create_intervals(float(C), int(n_samples), data_set)
                        constraints = ineq_to_constraints(functions, intervals, decoupled=True)
                    else:
                        parameters = ["p", "q"]
                        parameter_domains = [[0, 1], [0, 1]]
                        constraints = ["q+p >= 1.0", "q+p <= 9.0"]
                        data_set = None

                    ## REFINE SPACE
                    # print(colored(f"Refinement, dataset {data_set}, confidence level {C}, {n_samples} samples", "green"))
                    for cores in cores_list:
                        print(f"parallel: {cores},{' async calls,' if is_async else ' map calls,'} sample guided: {sample_guided}, single call timeout: {single_call_timeout}")
                        text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                        space = RefinedSpace(parameter_domains, parameters)

                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=2, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)

                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=2, solver="dreal",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)

                        spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                             single_call_timeout=single_call_timeout, debug=debug, alg=5, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        print()

    ## Knuth die 3 params
    ## LOAD FUNCTIONS
    functions = load_functions(f"Knuth/parametric_die_3_paramsBSCCs", debug=debug, source=model_checker)

    ## LOAD DATA
    data_set = load_data(os.path.join(data_dir, f"knuth_3params_generated_data_p1_0,4_p2_0,7_p3_0,5_1000_samples.p"), debug=debug)

    ## COMPUTE INTERVALS
    n_samples = 100

    ## SETUP PARAMETERS AND THEIR DOMAINS
    parameters = ["p1", "p2", "p3"]

    if debug:
        print("parameters", parameters)
    parameter_domains = []
    for item in parameters:
        parameter_domains.append([0, 1])
    if debug:
        print("parameter_domains", parameter_domains)

    intervals = create_intervals(float(C), int(n_samples), data_set)
    constraints = ineq_to_constraints(functions, intervals, decoupled=True)

    ## REFINE SPACE
    # print(colored(f"Refinement, dataset {data_set}, confidence level {C}, {n_samples} samples", "green"))

    for sample_guided in [False, True]:  ## [False, True]
        for is_async in [False, True]:  ## [False, True]
            for cores in cores_list:
                print(
                    f"parallel: {cores},{' async calls,' if is_async else ' map calls,'} sample guided: {sample_guided}, single call timeout: {single_call_timeout}")
                text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                space = RefinedSpace(parameter_domains, parameters)

                spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                     single_call_timeout=single_call_timeout, debug=debug, alg=2, solver="z3",
                                     sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                     parallel=cores, is_async=is_async)
                spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                     single_call_timeout=single_call_timeout, debug=debug, alg=2, solver="dreal",
                                     sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                     parallel=cores, is_async=is_async)
                spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout=timeout, silent=silent,
                                     single_call_timeout=single_call_timeout, debug=debug, alg=5, solver="z3",
                                     sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                     parallel=cores, is_async=is_async)
                print()
