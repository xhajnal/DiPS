import sys
import os
from termcolor import colored

from performance_test import repeat_sampling, load_functions
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

## SAMPLING SETTING
grid_size = 25

## INTERVALS SETTINGS
C = 0.95

precision = 4
del spam

if __name__ == '__main__':
    ## LOAD FUNCTIONS
    for population_size in [2, 3, 5, 10, 15]:
        cores_list = [False, 1, 2, 3, 5, 10, True]
        multiparam = False
        # population_size = 10
        print(colored(f"population size {population_size}", "blue"))
        functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug)

        ## LOAD DATA
        data_set = load_data(os.path.join(data_dir, f"data/2-param/data_n={population_size}.csv"), debug=debug)

        ## COMPUTE INTERVALS
        n_samples = 100

        ## SETUP PARAMETERS AND THEIR DOMAINS
        parameters = ["p", "q"]

        if debug:
            print("parameters", parameters)
        parameter_domains = []
        for item in parameters:
            parameter_domains.append([0, 1])
        if debug:
            print("parameter_domains", parameter_domains)

        intervals = create_intervals(float(C), int(n_samples), data_set)
        constraints = ineq_to_constraints(functions, intervals, decoupled=True)

        ## SAMPLE SPACE
        print(colored(f"Sampling, dataset {data_set}, grid size {grid_size}, {n_samples} samples", "green"))
        for cores in cores_list:
            print("parallel:", cores)
            space = RefinedSpace(parameter_domains, parameters)
            repeat_sampling(space, constraints, grid_size, silent=silent, save=False, debug=debug,
                            quantitative=False, parallel=cores, repetitions=3000)