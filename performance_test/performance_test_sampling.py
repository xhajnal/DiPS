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

## GENERAL SETTINGS
# set True for debug mode
debug = False
# set False for command line print
silent = True
# set True to show visualisation
show_space = False

## SAMPLING SETTING
# Number of sampling points per dimension
grid_size = 25

## INTERVALS SETTINGS
# Number of samples
n_samples_list = [3500, 100]
# Confidence level
C = 0.95

## EXPERIMENT SETUP
# List of cores to Use, True for core_count -1, positive int for number of cores, False for sequential method, and "False4" for using running with alg4 instead of default alg
cores_list = [True, 8, 4, 2, 1, False, "False4"]  ## [True, 8, 4, 2, 1, False]  ## This is CPU depending setting
# Number of significant numbers in results (time it took)
precision = 4
# Number of refinements per individual setting
repetitions = 20

del spam


if __name__ == '__main__':
    for n_samples in n_samples_list:

        ### BEE MODELS
        # 2-param vs. multi-parametric - False for 2-param, True for #params = #bees
        for multiparam in [False, True]:  ## [False, True]
            # Number of bees
            for population_size in [2, 3, 5, 10, 15]:
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

                ## SETUP PARAMETERS AND THEIR DOMAINS
                parameters = ["p", "q"]

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
                    space = RefinedSpace(parameter_domains, parameters)
                    repeat_sampling(space, constraints, grid_size, silent=silent, save=False, debug=debug,
                                    quantitative=False, parallel=cores, repetitions=repetitions, show_space=show_space)

        ### KNUTH DIE 3-params
        # Number of parameters 1,2, or 3 - this pics different model versions
        for param_count in [1, 2, 3]:

            ## LOAD FUNCTIONS
            functions = load_functions(f"Knuth/parametric_die_{param_count}_param_BSCCs", debug=debug, source=model_checker)

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
                space = RefinedSpace(parameter_domains, parameters)
                repeat_sampling(space, constraints, grid_size, silent=silent, save=False, debug=debug,
                                quantitative=False, parallel=cores, repetitions=repetitions, show_space=show_space)

        ### Bounded Retransmission Protocol (brp model)
        # Model version, first number is N, number of chunks, second number is MAX, maximum number of retransmissions
        for version in ["3-2", "16-2"]:
            ## LOAD FUNCTIONS
            functions = load_functions(f"brp/brp_{version}", debug=debug, source=model_checker)

            ## LOAD DATA
            data_set = [0.5]

            ## COMPUTE INTERVALS
            n_samples = 100

            ## SETUP PARAMETERS AND THEIR DOMAINS
            consts, parameters = parse_params_from_model(os.path.join(model_path, f"brp/brp_{version}.pm"), silent=True)

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
                space = RefinedSpace(parameter_domains, parameters)
                repeat_sampling(space, constraints, grid_size, silent=silent, save=False, debug=debug,
                                quantitative=False, parallel=cores, repetitions=repetitions, show_space=show_space)
