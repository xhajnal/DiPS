import multiprocessing
from copy import copy
from time import time
import numpy as np
from termcolor import colored

## Importing my code
from common.convert import normalise_constraint, split_constraints
from common.mathematics import create_matrix, cartesian_product
from common.my_z3 import is_this_z3_function, translate_z3_function
from common.config import load_config
from space import RefinedSpace

spam = load_config()
results_dir = spam["results"]
refinement_results = spam["refinement_results"]
refine_timeout = spam["refine_timeout"]
z3_path = spam["z3_path"]
del spam


global glob_sort
global glob_space
global glob_debug
global glob_compress
global glob_constraints


def check_sample(parameter_value):
    """ Checks whether constraints are satisfied in the given point """
    ## If sort constraint is not sat we simply skipp the point and not put it in the space.samples
    if glob_sort:
        if (parameter_value != np.sort(parameter_value)).any():
            return None
    for param in range(len(glob_space.params)):
        locals()[glob_space.params[param]] = float(parameter_value[param])
        if glob_debug:
            print("type(locals()[space.params[param]])", type(locals()[glob_space.params[param]]))
            print(f"locals()[space.params[param]] = {glob_space.params[param]} = {float(parameter_value[param])}")

    sat_list = []
    ## For each constraint (inequality - interval bound)
    for constraint_index, constraint in enumerate(glob_constraints):
        if glob_debug:
            print(f"constraints[{constraint_index}]", constraint)
            print(f"eval(constraints[{constraint_index}])", eval(constraint))

        is_sat = eval(constraint)
        if glob_compress and not is_sat:
            ## Skip evaluating other point as one of the constraint is not sat
            # print(f"{parameter_value} unsat")
            # print(f"new space {glob_space}")
            ## TODO, the following line works only for the sequential version
            glob_space.add_unsat_samples([list(parameter_value)])
            return False
        sat_list.append(is_sat)
    if glob_compress:
        ## TODO, the following line works only for the sequential version
        glob_space.add_sat_samples([list(parameter_value)])
        return True
    else:
        return sat_list


def sample_sat_degree(parameter_value):
    """ Computes satisfaction degree of constraints in the given point """
    if glob_sort:
        if (parameter_value != np.sort(parameter_value)).any():
            return None
    for param in range(len(glob_space.params)):
        locals()[glob_space.params[param]] = float(parameter_value[param])
        if glob_debug:
            print("type(locals()[space.params[param]])", type(locals()[glob_space.params[param]]))
            print(f"locals()[space.params[param]] = {glob_space.params[param]} = {float(parameter_value[param])}")

    distance_list = []

    for constraint_index, constraint in enumerate(glob_constraints):
        if glob_debug:
            print(f"constraints[{constraint_index}]", constraint)
            print(f"eval(constraints[{constraint_index}])", eval(constraint))

        mid = eval(constraint[1])
        if constraint[2] is None:
            dist = eval(f"{mid} - {eval(constraint[0])}")
        else:
            dist = min(eval(f"{mid} - {eval(constraint[0])}"), eval(f"{eval(constraint[2])} - {mid}"))

        distance_list.append(dist)

    if glob_compress:
        return sum(distance_list)
    else:
        return distance_list


def sample_space(space, constraints, sample_size, compress=False, silent=True, save=False, debug: bool = False,
                 progress=False, quantitative=False, sort=False, parallel=10):
    """ Samples the space in **sample_size** samples in each dimension and saves if the point is in respective interval

    Args:
        space: (space.RefinedSpace): space
        constraints  (list of strings): array of properties
        sample_size (int): number of samples in dimension
        compress (bool): if True, only a conjunction of the values (prop in the interval) is used
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        save (bool): if True output is pickled
        debug (bool): if True extensive print will be used
        progress (Tkinter element): progress bar
        quantitative (bool): if True return how far is the point from satisfying / not satisfying the constraints
        sort (Bool): tag whether the params are non-decreasing (CASE STUDY SPECIFIC SETTING)
        parallel (Bool): flag to run this in parallel mode

    """
    start_time = time()
    global glob_sort
    global glob_space
    global glob_debug
    global glob_compress
    global glob_constraints

    assert isinstance(space, RefinedSpace)
    if debug:
        silent = False

    if isinstance(parallel, int):
        pool_size = parallel
    else:
        pool_size = multiprocessing.cpu_count() - 1

    ## Convert z3 functions
    for index, constraint in enumerate(constraints):
        if is_this_z3_function(constraint):
            constraints[index] = translate_z3_function(constraint)

    ## Convert constraints for quantitative sampling
    if quantitative:
        constraints = copy(constraints)
        constraints = list(map(normalise_constraint, constraints))

        ## Split constraints into two pairs ((left, mid)(mid, right)) or ((left, right), None)
        constraints = split_constraints(constraints)

    parameter_values = []
    if debug:
        print("space.params", space.params)
        print("space.region", space.region)
        print("sample_size", sample_size)

    ## Create list of parameter values to sample
    for index in range(len(space.params)):
        parameter_values.append(np.linspace(space.region[index][0], space.region[index][1], sample_size, endpoint=True))
    sampling = create_matrix(sample_size, len(space.params))
    parameter_values = cartesian_product(*parameter_values)

    if not silent:
        print("sampling here")
        print("sample_size", sample_size)
        print("space.params", space.params)
        print("parameter_values", parameter_values)
    if debug:
        print("sampling", sampling)

    del sampling
    glob_sort = sort
    glob_space = space
    glob_debug = debug
    glob_compress = compress
    glob_constraints = constraints

    print(colored(f"Sampling initialisation took {round(time() - start_time, 4)} seconds", "yellow"))

    ## ACTUAL SAMPLING
    if parallel and not quantitative:
        with multiprocessing.Pool(pool_size) as p:
            sat_list = list(p.map(check_sample, parameter_values))
            ## TODO check how to alter progress when using Pool

        ## TODO this can be optimised by putting two lists separately
        for index, item in enumerate(parameter_values):
            if glob_compress:
                if sat_list[index]:
                    space.add_sat_samples([list(item)])
                elif not sat_list[index]:
                    space.add_unsat_samples([list(item)])
                else:
                    ## skipped point
                    pass
            else:
                if False in sat_list[index]:
                    space.add_unsat_samples([list(item)])
                elif True in sat_list[index]:
                    space.add_sat_samples([list(item)])
                else:
                    ## skipped point
                    pass

    elif parallel and quantitative:
        with multiprocessing.Pool(pool_size) as p:
            dist_list = list(p.map(sample_sat_degree, parameter_values))

        for index, item in enumerate(parameter_values):
            space.add_degree_samples({tuple(item): dist_list[index]})

    elif not quantitative:
        for index, item in enumerate(parameter_values):
            check_sample(item)
            if progress:
                progress(index / len(parameter_values))
        space = glob_space
    else:
        for index, item in enumerate(parameter_values):
            space.add_degree_samples({tuple(item): sample_sat_degree(item)})

            if progress:
                progress(index / len(parameter_values))

    ## Setting flag to not visualise sat if no unsat and vice versa
    space.gridsampled = True

    space.sampling_took(time() - start_time)
    space.title = f"using grid_size:{sample_size}"
    print(colored(f"Sampling took {round(time()-start_time, 4)} seconds", "yellow"))

    if parallel:
        if quantitative:
            return dist_list
        else:
            return sat_list

