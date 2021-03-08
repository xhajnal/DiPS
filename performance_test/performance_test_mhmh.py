import sys
import os
import time

from termcolor import colored

import performance_test
from mhmh import initialise_mhmh
from performance_test import repeat_mhmh, load_functions
from common.convert import ineq_to_constraints
from common.mathematics import create_intervals
from load import load_data
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

model_checker = "prism"

## PATHS
sys.setrecursionlimit(4000000)

global debug
globals()['debug'] = False
global silent
globals()['silent'] = True
global factorise
globals()['factorise'] = True

## REFINEMENT SETTING
timeout = 3600
max_depth = -1
coverage = 0.95
epsilon = 0

## INTERVALS SETTINGS
C = 0.95

precision = 4
repetitions = 20
sample_guided = False

## MHMH settings
mhmh_bins = 11  ## number of bins per dimension
mhmh_iterations = 1000
mhmh_timeout = 3600
mhmh_where = None  ## skip showing figures
mhmh_gui = False  ## skip showing progress
mhmh_metadata = False  ## skip showing metadata

del spam

if __name__ == '__main__':
    ## LOAD FUNCTIONS
    for population_size in [2, 3, 5, 10, 15]:  ## [2, 3, 5, 10, 15]
        cores_list = [False]  ## [True, 8, 4, 2, 1, False]
        multiparam = False
        # population_size = 10
        print(colored(f"population size {population_size}", "blue"))
        functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug)

        ## LOAD DATA
        data_set = load_data(os.path.join(data_dir, f"data/2-param/data_n={population_size}.csv"), debug=debug)

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

        for n_samples in n_samples_list:
            intervals = create_intervals(float(C), int(n_samples), data_set)
            constraints = ineq_to_constraints(functions, intervals, decoupled=True)

            ## MHMH
            for cores in cores_list:
                print("parallel:", cores)
                text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                            sample_size=n_samples, mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                            debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints,
                            recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=2, solver="z3",
                            gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions, parallel=cores)

                # time.sleep(2)
                repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                            sample_size=n_samples, mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                            debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints,
                            recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=2, solver="dreal",
                            gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions, parallel=cores)
                # time.sleep(2)
