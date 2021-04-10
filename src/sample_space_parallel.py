import multiprocessing
from time import time
import numpy as np
from termcolor import colored

## Importing my code
# from common.convert import normalise_constraint, split_constraints
from common.mathematics import create_matrix, cartesian_product
from common.my_z3 import is_this_z3_function, translate_z3_function
from common.config import load_config
# from space import RefinedSpace

spam = load_config()
results_dir = spam["results"]
refinement_results = spam["refinement_results"]
refine_timeout = spam["refine_timeout"]
z3_path = spam["z3_path"]
del spam

global glob_params
global glob_debug
global glob_compress
global glob_constraints


def check_sample(parameter_value, save_memory=False):
    """ Checks whether constraints are satisfied in the given point. """
    ## If sort constraint is not sat we simply skipp the point and not put it in the space.samples
    for param in range(len(glob_params)):
        locals()[glob_params[param]] = float(parameter_value[param])
        if glob_debug:
            print("type(locals()[glob_params[param]])", type(locals()[glob_params[param]]))
            print(f"locals()[glob_params[param]] = {glob_params[param]} = {float(parameter_value[param])}")

    sat_list = []
    ## For each constraint (inequality - interval bound)
    for constraint_index, constraint in enumerate(glob_constraints):
        if glob_debug:
            print(f"constraints[{constraint_index}]", constraint)
            print(f"eval(constraints[{constraint_index}])", eval(constraint))

        try:
            is_sat = eval(constraint)
        except Exception as err:
            print(colored(f"An error occurred while evaluating parameter point {parameter_value} and constraint number {constraint_index +1}", "red"))
            print(colored(f"   {err}", "red"))
            print(colored("   skipping this point", "red"))
            if glob_compress:
                return None
            else:
                is_sat = None
        if glob_compress and not is_sat:
            return False
        sat_list.append(is_sat)
    if glob_compress:
        return True
    else:
        return sat_list


def sample_space(boundaries, constraints, sample_size, compress_sample=False, compress_region=False, silent=True,
                 debug: bool = False, parallel=10):
    """ Samples the space in **sample_size** samples in each dimension and saves if the point is in respective interval.

    Args:
        constraints  (list of strings): list of constraints
        sample_size (int): number of samples in dimension
        boundaries (list of intervals): subspace to sample, False for default region of space
        compress_sample (bool): if True, only a conjunction of the values (prop in the interval) is used for a sample
        compress_region (bool): if True, only qualitative information (safe, unsafe, neither) about region is returned
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        parallel (bool): flag to run this in parallel mode
    """
    start_time = time()
    global glob_params
    global glob_debug
    global glob_compress
    global glob_constraints

    if compress_region is True:
        compress_sample = True

    if debug:
        silent = False

    if parallel > 1:
        pool_size = parallel
    else:
        pool_size = multiprocessing.cpu_count() - 1

    ## Convert z3 functions
    for index, constraint in enumerate(constraints):
        if is_this_z3_function(constraint):
            constraints[index] = translate_z3_function(constraint)

    parameter_values = []
    if debug:
        print("params", glob_params)
        print("region", boundaries)
        print("sample_size", sample_size)

    ## Create list of parameter values to sample
    for index in range(len(glob_params)):
        parameter_values.append(np.linspace(boundaries[index][0], boundaries[index][1], sample_size, endpoint=True))
    sampling = create_matrix(sample_size, len(glob_params))
    parameter_values = cartesian_product(*parameter_values)

    if not silent:
        print("sampling here")
        print("sample_size", sample_size)
        print("params", glob_params)
        print("parameter_values", parameter_values)
    if debug:
        print("sampling", sampling)

    del sampling
    glob_debug = debug
    glob_compress = compress_sample
    glob_constraints = constraints

    print(colored(f"Sampling initialisation took {round(time() - start_time, 4)} seconds", "yellow")) if not silent else None

    ## ACTUAL SAMPLING
    if parallel:
        ## Parallel sampling
        with multiprocessing.Pool(pool_size) as p:
            sat_list = list(p.map(check_sample, parameter_values))
            ## TODO check how to alter progress when using Pool

    if compress_region:
        if True in sat_list:
            if False in sat_list:
                return "neither"
            else:
                return "safe"
        else:
            return "unsafe"
    else:
        return sat_list

