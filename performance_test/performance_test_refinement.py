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

### SETTINGS

## PATHS
cwd = os.path.dirname(os.path.abspath(__file__))
spam = load_config(os.path.join(cwd, "config.ini"))
data_dir = spam["data"]
model_path = spam["models"]
property_path = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
refinement_results = spam["refinement_results"]
tmp = spam["tmp"]

## GENERAL SETTINGS
# Set True for debug mode (extensive output)
debug = False
# Set False for no command line print (or True for some output)
silent = True
# Set True to show visualisation(s)
show_space = False

## RATIONAL FUNCTIONS SETTINGS
## Parametric Model checker to run/load results from
model_checker = "storm"  # "prism" for PRISM, "storm" for Storm, and "either" for any of PRISM or Storm
# Flag whether to use factorised rational functions
factorise = False

## REFINEMENT SETTING
# Refinement timeout (in seconds)
timeout = 3600
# List of settings of sampling-guided refinement to run
sample_guided_list = [False, True]  ## [False, True]
# List of flags whether to use asynchronous calls refinement to run
# In this test we look only at synchronous check calls using pool.map:
async_list = [False]  ## [False]

## INTERVALS SETTINGS
# Number of samples
n_samples_list = [100]
# Confidence level
C = 0.95

## EXPERIMENT SETUP
# List of cores to Use, True for core_count -1, positive int for number of cores, False for sequential method, and "False4" for using running with alg4 instead of default alg
cores_list = [True, 8, 4, 2, 1, False, "False4"]  ## [True, 8, 4, 2, 1, False]  ## This is CPU depending setting
# Number of significant numbers in results (time it took)
precision = 4
# Number of refinements per individual setting
repetitions = 20

# Setting of each of the models, sizes, properties, etc. is within the section of the respective models

## INTERNAL SETTINGS - Experimental - please do not alter this part
sys.setrecursionlimit(4000000)
default_alg = 2
factorise = True
# single refinement call timeout
single_call_timeout = 0  ## 30 or 0

## END OF SETTINGS
del spam
if show_space:
    where = False
else:
    where = None

if __name__ == '__main__':
    for n_samples in n_samples_list:
        ### BEE MODELS
        # 2-param vs. multi-parametric - False for 2-param, True for #params = #bees
        for multiparam in [False, True]:  ## [False, True]
            # Number of bees
            for population_size in [2, 3, 4, 5, 10]:  ## [2, 3, 4, 5, 10]
                for is_async in async_list:
                    for sample_guided in sample_guided_list:
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
                        ## LOAD FUNCTIONS
                        if multiparam:
                            functions = load_functions(f"bee/multiparam_semisynchronous_{population_size}_bees", debug=debug, source=model_checker, factorise=factorise)
                        else:
                            functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug, source=model_checker, factorise=factorise)

                        ## LOAD DATA
                        if multiparam:
                            data_set = load_data(os.path.join(data_dir, f"bee/multiparam/data_n={population_size}.csv"), debug=debug)
                        else:
                            data_set = load_data(os.path.join(data_dir, f"bee/2-param/data_n={population_size}.csv"), debug=debug)

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

                        ## COMPUTE CONFIDENCE INTERVALS
                        ## Intervals with adjusted Wald method used in Hajnal et al. hsb 2019
                        intervals = create_intervals_hsb(float(C), int(n_samples), data_set)
                        ## THIS IS AGRESTI-COULL METHOD
                        # intervals = [create_proportions_interval(float(C), int(n_samples), data_point, method="AC") for data_point in data_set]

                        ## COMPUTE CONSTRAINTS
                        constraints = ineq_to_constraints(functions, intervals, decoupled=True)

                        ## REFINE SPACE
                        for cores in cores_list:
                            if cores == "False4":
                                alg = 4
                                second_alg = None
                                cores = False
                            else:
                                alg = default_alg
                                second_alg = 5

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
                                                 single_call_timeout=single_call_timeout, debug=debug, alg=second_alg, solver="z3",
                                                 sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                                 parallel=cores, is_async=is_async)
                            print()

        ### KNUTH DIE
        # Number of parameters 1, 2, or 3
        for param_count in [1, 2, 3]:  # this selects respective model version

            ## LOAD FUNCTIONS
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

            ## COMPUTE CONSTRAINTS
            constraints = ineq_to_constraints(functions, intervals, decoupled=True)

            ## REFINE SPACE
            for is_async in async_list:
                for sample_guided in sample_guided_list:
                    for cores in cores_list:
                        if cores == "False4":
                            alg = 4
                            second_alg = None
                            cores = False
                        else:
                            alg = default_alg
                            second_alg = 5

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
                                             single_call_timeout=single_call_timeout, debug=debug, alg=second_alg, solver="z3",
                                             sample_size=False, sample_guided=sample_guided, repetitions=repetitions, where=where,
                                             parallel=cores, is_async=is_async)
                        print()
