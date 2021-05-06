import os
from termcolor import colored

from performance_test import repeat_sampling, load_functions
from common.convert import ineq_to_constraints
from common.mathematics import create_intervals_hsb
from load import load_data
from common.model_stuff import parse_params_from_model
from space import RefinedSpace
from common.config import load_config

### SETTINGS

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
del spam

## GENERAL SETTINGS
# Set True for debug mode (extensive output)
debug = False
# Set False for no command line print (or True for some output)
silent = True
# set True to show visualisation
show_space = False

## RATIONAL FUNCTIONS SETTINGS
## Parametric Model checker to run/load results from
model_checker = "storm"  # "prism" for PRISM, "storm" for Storm, and "either" for any of PRISM or Storm
# Flag whether to use factorised rational functions
factorise = False

## SAMPLING SETTING
# Number of sampling points per dimension
grid_size = 25

## INTERVALS SETTINGS
# Number of samples
n_samples_list = [100]
# Confidence level
C = 0.95

## EXPERIMENT SETUP
# List of cores to Use, True for core_count -1, positive int for number of cores, False for sequential method
cores_list = [True, 12, 8, 4, 2, 1, False]  ## [True, 8, 4, 2, 1, False]  ## This is CPU depending setting
# Number of significant numbers in results (time it took)
precision = 4
# Number of refinements per individual setting
repetitions = 300  ## 300

if __name__ == '__main__':
    for n_samples in n_samples_list:
        ### BEE MODELS
        # 2-param vs. multi-parametric - False for 2-param, True for #params = #bees
        for multiparam in [False]:  ## [False, True]
            # Number of bees
            for population_size in [2, 3, 4, 5, 10, 15]:
                if multiparam and population_size == 2:
                    print(colored("Skipping 2 bees multiparam as this model is equivalent to 2-param model", "red"))
                    continue
                if not multiparam and population_size == 4:
                    print(colored("Skipping 4 bees 2-param. We go straight to 5 bees.", "red"))
                    continue

                if multiparam and population_size >= 5:
                    print(colored(f"Skipping {population_size} bees multiparam. This would probably take too much time.", "yellow"))
                    continue

                ## LOAD FUNCTIONS
                if multiparam:
                    functions = load_functions(f"bee/multiparam_semisynchronous_{population_size}_bees", debug=debug, source=model_checker, factorise=factorise)
                else:
                    functions = load_functions(f"bee/semisynchronous_{population_size}_bees", debug=debug, source=model_checker, factorise=factorise)

                print("functions", functions)

                ## LOAD DATA
                if multiparam:
                    data_set = load_data(os.path.join(data_dir, f"bee/multiparam/data_n={population_size}.csv"), silent=silent, debug=debug)
                else:
                    data_set = load_data(os.path.join(data_dir, f"bee/2-param/data_n={population_size}.csv"), silent=silent, debug=debug)

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

                ## SAMPLE SPACE
                print(colored(f"Sampling, dataset {data_set}, grid size {grid_size}, {n_samples} samples", "green"))
                for cores in cores_list:
                    print("parallel:", cores)
                    repeat_sampling(parameters, parameter_domains, constraints, grid_size, silent=silent, save=False, debug=debug,
                                    quantitative=False, parallel=cores, repetitions=repetitions, show_space=show_space)

        ### KNUTH DIE
        # Number of parameters 1, 2, or 3
        for param_count in [1, 2, 3]:  # this selects respective model version

            ## LOAD FUNCTIONS
            functions = load_functions(f"Knuth/parametric_die_{param_count}_param_BSCCs", debug=debug, source=model_checker)

            ## LOAD DATA
            data_set = load_data(os.path.join(data_dir, f"knuth_3params_generated_data_p1_0,4_p2_0,7_p3_0,5_1000_samples.p"), debug=debug)

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

            # SAMPLE SPACE
            print(colored(f"Sampling, dataset {data_set}, grid size {grid_size}, {n_samples} samples", "green"))
            for cores in cores_list:
                print("parallel:", cores)
                repeat_sampling(parameters, parameter_domains, constraints, grid_size, silent=silent, save=False, debug=debug,
                                quantitative=False, parallel=cores, repetitions=repetitions, show_space=show_space)
