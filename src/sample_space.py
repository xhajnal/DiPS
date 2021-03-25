import multiprocessing
import warnings
from copy import copy
from functools import partial
from time import time
from typing import Iterable

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

global glob_space
global glob_debug
global glob_compress
global glob_constraints


def check_sample(parameter_value, save_memory=False, silent=False):
    """ Checks whether constraints are satisfied in the given point """
    ## If sort constraint is not sat we simply skipp the point and not put it in the space.samples
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

        try:
            is_sat = eval(constraint)
        except Exception as err:
            if not silent:
                print(colored(f"An error occurred while evaluating parameter point {parameter_value} and constraint number {constraint_index +1}", "red"))
                print(colored(f"   {err}", "red"))
                print(colored("   skipping this point", "red"))
            if glob_compress:
                return None
            else:
                is_sat = None
        if glob_compress and not is_sat:
            ## Skip evaluating other point as one of the constraint is not sat
            # print(f"{parameter_value} unsat")
            # print(f"new space {glob_space}")
            ## TODO, the following line works only for the sequential version
            if not save_memory:
                glob_space.add_unsat_samples([list(parameter_value)])
            # print("False")
            return False
        sat_list.append(is_sat)
    if glob_compress:
        ## TODO, the following line works only for the sequential version
        glob_space.add_sat_samples([list(parameter_value)])
        # print("True")
        return True
    else:
        return sat_list


def sample_sat_degree(parameter_value):
    """ Computes satisfaction degree of constraints in the given point """
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
        elif constraint[0] is None:
            dist = eval(f"{eval(constraint[2])} - {mid}")
        else:
            dist = min(eval(f"{mid} - {eval(constraint[0])}"), eval(f"{eval(constraint[2])} - {mid}"))

        distance_list.append(dist)

    if glob_compress:
        return sum(distance_list)
    else:
        return distance_list


def sample_space(space, constraints, sample_size, boundaries=False, compress=False, silent=True, save=False, debug: bool = False,
                 progress=False, quantitative=False, parallel=True, save_memory=False, stop_on_unknown=False):
    """ Samples the space in **sample_size** samples in each dimension and saves if the point is in respective interval

    Args:
        space: (space.RefinedSpace): space
        constraints  (list of strings): array of properties
        sample_size (int): number of samples in dimension
        boundaries (list of intervals): subspace to sample, False for default region of space
        compress (bool): if True, only a conjunction of the values (prop in the interval) is used
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        save (bool): if True output is pickled
        debug (bool): if True extensive print will be used
        progress (Tkinter element): progress bar
        quantitative (bool): if True return how far is the point from satisfying / not satisfying the constraints
        parallel (Bool): flag to run this in parallel mode
        save_memory (Bool): if True saves only sat samples

    """
    start_time = time()
    global glob_space
    global glob_debug
    global glob_compress
    global glob_constraints

    assert isinstance(space, RefinedSpace)
    # print(debug)
    # print(silent)
    if debug:
        silent = False

    if parallel is True:
        pool_size = multiprocessing.cpu_count() - 1
    elif parallel > 1:
        pool_size = min(parallel, multiprocessing.cpu_count() - 1)
    elif parallel == 1:
        pool_size = 1

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
        if boundaries:
            parameter_values.append(np.linspace(boundaries[index][0], boundaries[index][1], sample_size, endpoint=True))
        else:
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
    glob_space = space
    glob_debug = debug
    glob_compress = compress
    glob_constraints = constraints

    print(colored(f"Sampling initialisation took {round(time() - start_time, 4)} seconds", "yellow")) if not silent else None

    ## ACTUAL SAMPLING
    if parallel and not quantitative:
        ## Parallel sampling
        if stop_on_unknown:
            ## TODO implement this, very good idea on paper
            raise NotImplementedError("this optimisation is not implemented so far, please use option stop_on_unknown=False")
            current = None
            with multiprocessing.Pool(pool_size) as p:
                check_samplee = partial(check_sample, silent=silent)
                results = [p.apply_async(check_samplee, item).get() for item in parameter_values]

            print(results)
            return results
            #     for item in parameter_values:
            #         print("item", item)
            #         result = p.apply_async(check_sample, item)
            #         print("result, ", result)
            #         if current is None:
            #             if result is True:
            #                 current = "safe"
            #             else:
            #                 current = "unsafe"
            #         elif current == "safe":
            #             if result is False:
            #                 current = "neither"
            #                 p.terminate()
            #         elif current == "unsafe":
            #             if result is True:
            #                 current = "neither"
            #                 p.terminate()
            # return current
        else:
            with multiprocessing.Pool(pool_size) as p:
                check_samplee = partial(check_sample, silent=silent)
                sat_list = list(p.map(check_samplee, parameter_values))
                ## TODO check how to alter progress when using Pool

        ## TODO this can be optimised by putting two lists separately
        for index, item in enumerate(parameter_values):
            if glob_compress:
                if sat_list[index]:
                    space.add_sat_samples([list(item)])
                elif not sat_list[index]:
                    if not save_memory:
                        space.add_unsat_samples([list(item)])
                else:
                    ## skipped point
                    pass
            else:
                if False in sat_list[index]:
                    if not save_memory:
                        space.add_unsat_samples([list(item)])
                elif True in sat_list[index]:
                    space.add_sat_samples([list(item)])
                else:
                    ## skipped point
                    pass

    elif parallel and quantitative:
        ## Parallel quantitative sampling
        with multiprocessing.Pool(pool_size) as p:
            dist_list = list(p.map(sample_sat_degree, parameter_values))

        for index, item in enumerate(parameter_values):
            space.add_degree_samples({tuple(item): dist_list[index]})

    elif not quantitative:
        ## Sequential sampling
        for index, item in enumerate(parameter_values):
            check_sample(item, save_memory, silent=silent)
            if progress:
                progress(index / len(parameter_values))
        space = glob_space
    else:
        ## Sequential quantitative sampling
        for index, item in enumerate(parameter_values):
            space.add_degree_samples({tuple(item): sample_sat_degree(item)})

            if progress:
                progress(index / len(parameter_values))

    ## Setting flag to not visualise sat if no unsat and vice versa
    space.gridsampled = True

    space.sampling_took(time() - start_time)
    space.title = f"using grid_size:{sample_size}"
    print(colored(f"Sampling took {round(time()-start_time, 4)} seconds", "yellow")) if not silent else None

    if parallel:
        if quantitative:
            return dist_list
        else:
            return sat_list
    else:
        if quantitative:
            return True
        else:
            return True


def sample_region(region, params, constraints, sample_size, boundaries=False, compress=False, silent=True, save=False,
                  debug=False, progress=False, quantitative=False, parallel=True, save_memory=False, stop_on_unknown=False):
    """ Samples the space in **sample_size** samples in each dimension and saves if the point is in respective interval

    Args:
        region: (Rectangle): region to be sampled
        params: (list of strings): parameters
        constraints  (list of strings): array of properties
        sample_size (int): number of samples in dimension
        boundaries (list of intervals): subspace to sample, False for default region of space
        compress (bool): if True, only a conjunction of the values (prop in the interval) is used
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        save (bool): if True output is pickled
        debug (bool): if True extensive print will be used
        progress (Tkinter element): progress bar
        quantitative (bool): if True return how far is the point from satisfying / not satisfying the constraints
        parallel (Bool): flag to run this in parallel mode
        save_memory (Bool): if True saves only sat samples

    """
    start_time = time()
    global glob_space
    global glob_debug
    global glob_compress
    global glob_constraints

    ## TODO this can be optimised using check_sample without using space
    glob_space = RefinedSpace(region, params)

    assert isinstance(region, Iterable)
    if debug:
        silent = False

    if parallel is True:
        pool_size = multiprocessing.cpu_count() - 1
    elif parallel > 1:
        pool_size = min(parallel, multiprocessing.cpu_count() - 1)
    elif parallel == 1:
        pool_size = 1

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
        print("space.params", params)
        print("space.region", region)
        print("sample_size", sample_size)

    ## Create list of parameter values to sample
    for index in range(len(params)):
        if boundaries:
            parameter_values.append(np.linspace(boundaries[index][0], boundaries[index][1], sample_size, endpoint=True))
        else:
            parameter_values.append(np.linspace(region[index][0], region[index][1], sample_size, endpoint=True))
    sampling = create_matrix(sample_size, len(params))
    parameter_values = cartesian_product(*parameter_values)

    if not silent:
        print("sampling here")
        print("sample_size", sample_size)
        print("params", params)
        print("parameter_values", parameter_values)
    if debug:
        print("sampling", sampling)

    del sampling
    glob_debug = debug
    glob_compress = compress
    glob_constraints = constraints

    print(colored(f"Sampling initialisation took {round(time() - start_time, 4)} seconds", "yellow")) if not silent else None

    ## ACTUAL SAMPLING
    if parallel and not quantitative:
        ## Parallel sampling
        if stop_on_unknown:
            ## TODO implement this, very good idea on paper
            raise NotImplementedError("this optimisation is not implemented so far, please use option stop_on_unknown=False")
            current = None
            with multiprocessing.Pool(pool_size) as p:
                check_samplee = partial(check_sample, silent=silent)
                results = [p.apply_async(check_samplee, item).get() for item in parameter_values]

            print(results)
            return results
            #     for item in parameter_values:
            #         print("item", item)
            #         result = p.apply_async(check_sample, item)
            #         print("result, ", result)
            #         if current is None:
            #             if result is True:
            #                 current = "safe"
            #             else:
            #                 current = "unsafe"
            #         elif current == "safe":
            #             if result is False:
            #                 current = "neither"
            #                 p.terminate()
            #         elif current == "unsafe":
            #             if result is True:
            #                 current = "neither"
            #                 p.terminate()
            # return current
        else:
            with multiprocessing.Pool(pool_size) as p:
                sat_list = list(p.map(check_sample, parameter_values))

    elif parallel and quantitative:
        ## Parallel quantitative sampling
        with multiprocessing.Pool(pool_size) as p:
            dist_list = list(p.map(sample_sat_degree, parameter_values))

    elif not quantitative:
        ## Sequential sampling
        sat_list, unsat_list = [], []
        for index, item in enumerate(parameter_values):
            spam = check_sample(item, save_memory, silent=silent)
            if spam is True:
                sat_list.append(item)
            elif spam is False:
                unsat_list.append(item)
            else:
                pass
                # warnings.filterwarnings("ignore")
                # raise Warning(f"Checking {parameter_values} resulted in unexpected value: {spam}")
            if progress:
                progress(index / len(parameter_values))
    else:
        ## Sequential quantitative sampling
        dist_map = {}
        for index, item in enumerate(parameter_values):
            dist_map[item] = sample_sat_degree(item)

            if progress:
                progress(index / len(parameter_values))

    ## Setting flag to not visualise sat if no unsat and vice versa
    print(colored(f"Sampling took {round(time()-start_time, 4)} seconds", "yellow")) if not silent else None

    if parallel:
        if quantitative:
            return dist_list
        else:
            return sat_list
    else:
        if quantitative:
            return dist_map
        else:
            return sat_list, unsat_list

