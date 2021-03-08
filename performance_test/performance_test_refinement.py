import sys
import os
from termcolor import colored

import performance_test
from performance_test import repeat_sampling, load_functions, repeat_refine
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

## INTERVALS SETTINGS
C = 0.95

precision = 4
repetitions = 20
sample_guided = True

del spam

if __name__ == '__main__':
    ## LOAD FUNCTIONS
    for population_size in [2, 3, 5, 10, 15]:  ## [2, 3, 5, 10, 15]
        cores_list = [True, 8, 4, 2, 1, False]  ## [True, 8, 4, 2, 1, False]
        multiparam = False
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

            ## REFINE SPACE
            # print(colored(f"Refinement, dataset {data_set}, confidence level {C}, {n_samples} samples", "green"))
            for cores in cores_list:
                print("parallel:", cores)
                text = f"dataset {data_set}, confidence level {C}, {n_samples} samples"
                space = RefinedSpace(parameter_domains, parameters)

                # spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout, silent, debug, alg=2,
                #                      solver="z3", sample_size=False, sample_guided=False,
                #                      repetitions=repetitions, parallel=cores, where=False)
                # spam[1].show(is_sampling_guided=sample_guided, is_parallel_refinement=cores)

                spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout, silent, debug, alg=2,
                                     solver="z3", sample_size=False, sample_guided=sample_guided,
                                     repetitions=repetitions, parallel=cores, where=False)
                # spam[1].show(is_sampling_guided=False, is_parallel_refinement=cores)

                # spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout, silent, debug, alg=2,
                #                      solver="dreal", sample_size=False, sample_guided=False,
                #                      repetitions=repetitions, parallel=cores, where=False)
                # spam[1].show(is_sampling_guided=False, is_parallel_refinement=False)
                spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout, silent, debug, alg=2,
                                     solver="dreal", sample_size=False, sample_guided=sample_guided,
                                     repetitions=repetitions, parallel=cores, where=False)
                # spam[1].show(is_sampling_guided=False, is_parallel_refinement=False)

                spam = repeat_refine(text, parameters, parameter_domains, constraints, timeout, silent, debug, alg=5,
                                     solver="z3", sample_size=False, sample_guided=sample_guided,
                                     repetitions=repetitions, parallel=cores, where=False)
                # spam[1].show(is_sampling_guided=False, is_parallel_refinement=False)
