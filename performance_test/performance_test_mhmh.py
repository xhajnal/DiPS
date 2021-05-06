import sys
import os
import time
from termcolor import colored

import performance_test
from mhmh import initialise_mhmh
from performance_test import repeat_mhmh, load_functions
from common.convert import ineq_to_constraints
from common.mathematics import create_intervals_hsb
from load import load_data
from space import RefinedSpace
from common.config import load_config

## SETTINGS
## PATHS
cwd = os.getcwd()
spam = load_config(os.path.join(cwd, "config.ini"))
data_dir = spam["data"]
model_path = spam["models"]
property_path = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
refinement_results = spam["refinement_results"]
tmp = spam["tmp"]

## RATIONAL FUNCTIONS SETTINGS
## Parametric Model checker to run/load results from
model_checker = "storm"  # "prism" for PRISM, "storm" for Storm, and "either" for any of PRISM or Storm
# Flag whether to use factorised rational functions
factorise = True

## PATHS
sys.setrecursionlimit(4000000)

## GENERAL SETTINGS
# Set True for debug mode (extensive output)
debug = False
# Set False for no command line print (or True for some output)
silent = True
# Set True to show visualisation(s)
show_space = False

## REFINEMENT SETTING
# Refinement part of MHMH
# Refinement timeout (in seconds)
timeout = 3600
# Recursion depth, set -1 for no depth
max_depth = -1
# Desired coverage
coverage = 0.95
# Minimal rectangle size (area/volume) to check
epsilon = 0

### MH SETTINGS
# Metropolis-Hastings part of MHMH
mhmh_bins = 11  ## number of bins per dimension
mhmh_iterations = 1000
mhmh_timeout = 3600
mhmh_where = None  ## skip showing figures
mhmh_gui = False  ## skip showing progress
mhmh_metadata = False  ## skip showing metadata

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
# Number of runs per individual setting
repetitions = 20
# Flag whether to use sampling-guided refinement
sample_guided = False

del spam

if __name__ == '__main__':
    for n_samples in n_samples_list:
        ### BEE MODELS
        # 2-param vs. multi-parametric - False for 2-param, True for #params = #bees
        for multiparam in [False, True]:  ## [False, True]
            for population_size in [2, 3, 5, 10, 15]:  ## [2, 3, 5, 10, 15]
                for cores in cores_list:
                    ## LOAD FUNCTIONS
                    print(colored(f"population size {population_size}", "blue"))
                    functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug, factorise=factorise)

                    ## LOAD DATA
                    data_set = load_data(os.path.join(data_dir, f"bee/2-param/data_n={population_size}.csv"), debug=debug)

                    ## COMPUTE INTERVALS
                    n_samples_list = [100, 1500, 3500]

                    ## SETUP PARAMETERS AND THEIR DOMAINS
                    parameters = ["p", "q"]

                    if debug:
                        print("parameters", parameters)
                    parameter_domains = []
                    for item in parameters:
                        parameter_domains.append([0, 1])
                    if debug:
                        print("parameter_domains", parameter_domains)

                    intervals = create_intervals_hsb(float(C), int(n_samples), data_set)
                    constraints = ineq_to_constraints(functions, intervals, decoupled=True)

                    print("parallel:", cores)
                    text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples, mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                                debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints,
                                recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=2, solver="z3",
                                gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions, parallel=cores)

                    time.sleep(2)
                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples, mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                                debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints,
                                recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=2, solver="dreal",
                                gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions, parallel=cores)
                    time.sleep(2)
                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples, mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                                debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints,
                                recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=5, solver="dreal",
                                gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions,
                                parallel=cores)
                    time.sleep(2)

        ### Knuth die
        # Number of parameters 1,2, or 3 - this pics different model versions
        for param_count in [1, 2, 3]:

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

