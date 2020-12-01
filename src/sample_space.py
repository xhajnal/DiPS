import multiprocessing
import os
import re
from copy import copy
from time import time, strftime, localtime
import numpy as np
from termcolor import colored

## Importing my code
from common.convert import normalise_constraint, split_constraints
from common.files import pickle_dump
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


# def bar(parameter_value, constraints, sort, space, debug, compress, return_dict):
#     """ Private method of sample_space """
#     ## If sort constraint is not sat we simply skipp the point and not put it in the space.samples
#     parameter_value = tuple(parameter_value)
#     if sort:
#         if (parameter_value != np.sort(parameter_value)).any():
#             return
#     for param in range(len(space.params)):
#         locals()[space.params[param]] = float(parameter_value[param])
#         if debug:
#             print("type(locals()[space.params[param]])", type(locals()[space.params[param]]))
#             print(f"locals()[space.params[param]] = {space.params[param]} = {float(parameter_value[param])}")
#
#     ## By default it is True
#     space.add_sat_samples(parameter_value)
#     return_dict[parameter_value] = True
#
#     ## For each constraint (inequality - interval bound)
#     for constraint_index, constraint in enumerate(constraints):
#         if debug:
#             print(f"constraints[{constraint_index}]", constraint)
#             print(f"eval(constraints[{constraint_index}])", eval(constraint))
#
#         is_sat = eval(constraint)
#         if compress and not is_sat:
#             ## Skip evaluating other point as one of the constraint is not sat
#             space.add_unsat_samples(parameter_value)
#             return_dict[parameter_value] = False
#             break


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
            return False
        sat_list.append(is_sat)
    if glob_compress:
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
                 progress=False, quantitative=False, sort=False, parallel=True):
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

    Returns:
        (dict) of point to list of Bools whether f(point) in interval[index]
        if quantitative
        (dict) of point to list of numbers, sum of distances to satisfy constraints
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

    ## TODO maybe normalise constraints before
    ## check whether constraints are in normal form
    if quantitative:
        constraints = copy(constraints)
        constraints = list(map(normalise_constraint, constraints))

        ## Split constraints into two pairs ((left, mid)(mid, right)) or ((left, right), None)
        constraints = split_constraints(constraints)

        ##

        # for constraint in constraints:
        #     if len(re.findall(">", constraint)) >= 1:
        #         raise Exception("Constraints", "Please rewrite constraints using < / <= instead of > / >=")

    ## Convert z3 functions
    for index, constraint in enumerate(constraints):
        if is_this_z3_function(constraint):
            constraints[index] = translate_z3_function(constraint)

    parameter_values = []
    parameter_indices = []
    if debug:
        print("space.params", space.params)
        print("space.region", space.region)
        print("sample_size", sample_size)
    for index in range(len(space.params)):
        parameter_values.append(np.linspace(space.region[index][0], space.region[index][1], sample_size, endpoint=True))
        parameter_indices.append(np.asarray(range(0, sample_size)))

    sampling = create_matrix(sample_size, len(space.params))
    if not silent:
        print("sampling here")
        print("sample_size", sample_size)
        print("space.params", space.params)
    if debug:
        print("sampling", sampling)
    parameter_values = cartesian_product(*parameter_values)
    parameter_indices = cartesian_product(*parameter_indices)

    # if (len(space.params) - 1) == 0:
    #    parameter_values = linspace(0, 1, sample_size, endpoint=True)[newaxis, :].T
    if not silent:
        print("parameter_values", parameter_values)
        print("parameter_indices", parameter_indices)
        # print("a sample_space:", sampling[0][0])
    parameter_index = 0
    ## For each parametrisation eval the constraints
    print(colored(f"Sampling initialisation took {round(time() - start_time, 4)} seconds", "yellow"))

    ## Set variables for multiprocessing
    if parallel:
        del sampling
        del parameter_indices
        del parameter_index

        glob_sort = sort
        glob_space = space
        glob_debug = debug
        glob_compress = compress
        glob_constraints = constraints

    if parallel and not quantitative:
        with multiprocessing.Pool(5) as p:
            sat_list = list(p.map(check_sample, parameter_values))

        # print(a)
        # print(parameter_values)

        ## TODO this can be optimised by putting two lists separately
        for index, item in enumerate(parameter_values):
            if sat_list[index]:
                space.add_sat_samples([item])
            elif not sat_list[index]:
                space.add_unsat_samples([item])
            else:
                ## skipped point
                pass

        # Implementations with Processes
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        # processes = []
        # for index, parameter_value in enumerate(parameter_values):
        #     # print("item ", item, "index", index)
        #     processes.append(multiprocessing.Process(target=bar, args=(parameter_value, constraints, sort, space, debug, compress, return_dict)))
        # print(colored(f"With making processes it took {round(time() - start_time, 4)} seconds", "yellow"))
        # for p in processes:
        #     p.start()
        #
        # for p in processes:
        #     p.join()
        #
        # print(return_dict)
        # print(return_dict.keys())
        #
        # for item in return_dict.keys():
        #     if return_dict[item]:
        #         space.add_sat_samples([item])
        #     else:
        #         space.add_unsat_samples([item])
    elif parallel and quantitative:
        # raise NotImplementedError("Parallel quantitative sampling is not implemented yet")

        with multiprocessing.Pool(5) as p:
            dist_list = list(p.map(sample_sat_degree, parameter_values))

        for index, item in enumerate(parameter_values):
            space.add_degree_samples({tuple(item): dist_list[index]})

    else:
    for index, parameter_value in enumerate(parameter_values):
        ## For each parameter set the current sample_space point value
        if progress:
            progress(index / len(parameter_values))
        if sort:
            if (parameter_value != np.sort(parameter_value)).any():
                continue
        for param in range(len(space.params)):
            locals()[space.params[param]] = float(parameter_value[param])
            if debug:
                print("type(locals()[space.params[param]])", type(locals()[space.params[param]]))
                print(f"locals()[space.params[param]] = {space.params[param]} = {float(parameter_value[param])}")
        ## print("parameter_value", parameter_value)
        # print(str(parameter_value))
        # print(type(parameter_value))
        ## print("parameter_index", parameter_indices[i])
        ## print(type(parameter_indices[i]))
        ## print("sampling", sampling)
        ## print("sampling[0][0]", sampling[0, 0])
        # sampling[0][0] = [[0.], [True]]

        ## print("sampling[0][0][0]", sampling[0][0][0])
        ## print("sampling[0][0][0]", type(sampling[0][0][0]))

        ## print("here")
        ## print(tuple(parameter_indices[i]))
        ## print(sampling[tuple(parameter_indices[i])])
        # sampling[0, 0] = 9
        # sampling[0, 0] = True

        sampling[tuple(parameter_indices[parameter_index])][0] = list(parameter_value)

        satisfaction_list = []
        ## Only for quantitative
        distance_list = []
        ## For each constraint (inequality - interval bound)
        for constraint_index, constraint in enumerate(constraints):
            # print(constraint)
            # print("type(constraint[index])", type(constraint))
            # for param in range(len(space.params)):
            #     print(space.params[param], parameter_value[param])
            #     print("type(space.params[param])", type(space.params[param]))
            #     print("type(parameter_value[param])", type(parameter_value[param]))

            if debug:
                print(f"constraints[{constraint_index}]", constraint)
                print(f"eval(constraints[{constraint_index}])", eval(constraint))

            if not quantitative:
                is_sat = eval(constraint)
                satisfaction_list.append(is_sat)
                ## Skips evaluating other point as one of the constraint is not sat
                if compress and not is_sat:
                    break
            else:
                ## Two interval bounds
                ## TODO this may be expensive and can be optimised by changing the constraints once in beginning
                if len(re.findall("<", constraint)) == 2:
                    # print(constraint)
                    ## LEFT SIDE
                    left_side = constraint.split("<")[:2]

                    check_left_sat = "<".join(left_side)
                    is_left_sat = eval(check_left_sat)
                    left_side = "-".join(left_side)
                    left_side = left_side.replace("=", "")

                    if is_left_sat:
                        left_distance = abs(eval(left_side))
                    else:
                        left_distance = -abs(eval(left_side))
                    # print("left distance", left_distance)

                    ## RIGHT SIDE
                    right_side = constraint.split("<")[1:]
                    check_right_sat = "<".join(right_side)
                    if check_right_sat[0] == " ":
                        check_right_sat = check_right_sat[1:]
                    if check_right_sat[0] == "=":
                        check_right_sat = check_right_sat[1:]
                    is_right_sat = eval(check_right_sat)

                    right_side = "-".join(right_side)
                    right_side = right_side.replace("=", "")
                    if is_right_sat:
                        right_distance = abs(eval(right_side))
                    else:
                        right_distance = -abs(eval(right_side))
                    # print("right distance", right_distance)
                    distance = round(min(left_distance, right_distance), 16)
                else:
                    ## Single interval bound
                    # print(constraint)
                    input = constraint.split("<")
                    check_sat = "<".join(input)
                    is_sat = eval(check_sat)
                    # print("check_sat", check_sat, is_sat)
                    input = "-".join(input)
                    input = input.replace("=", "")
                    # print(input)
                    if is_sat:
                        distance = round(abs(eval(input)), 16)
                    else:
                        distance = -round(abs(eval(input)), 16)
                # print("sat degree", distance)
                satisfaction_list.append(True if distance >= 0 else False)
                distance_list.append(float(distance))

            ## print("cycle")
            ## print(sampling[tuple(parameter_indices[i])])
        if quantitative:
            if compress:
                space.add_degree_samples({tuple(parameter_value): sum(distance_list)})
            else:
                space.add_degree_samples({tuple(parameter_value): distance_list})

        if False in satisfaction_list:
            # print("adding unsat", sampling[tuple(parameter_indices[i])][0])
            space.add_unsat_samples([sampling[tuple(parameter_indices[parameter_index])][0]])
            if compress:
                sampling[tuple(parameter_indices[parameter_index])][1] = False
            else:
                sampling[tuple(parameter_indices[parameter_index])][1] = satisfaction_list
        else:
            # print("adding sat", sampling[tuple(parameter_indices[i])][0])
            space.add_sat_samples([sampling[tuple(parameter_indices[parameter_index])][0]])
            if compress:
                sampling[tuple(parameter_indices[parameter_index])][1] = True
            else:
                sampling[tuple(parameter_indices[parameter_index])][1] = satisfaction_list

        parameter_index = parameter_index + 1

    ## Setting flag to not visualise sat if no unsat and vice versa
    space.gridsampled = True

    ## Saving the sampled space as pickled dictionary
    if save is True:
        save = str(strftime("%d-%b-%Y-%H-%M-%S", localtime()))
    try:
        pickle_dump(sampling, os.path.join(refinement_results, ("Sampled_space_" + save).split(".")[0] + ".p"))
    except UnboundLocalError as err:
        pass

    space.sampling_took(time() - start_time)
    space.title = f"using grid_size:{sample_size}"
    print(colored(f"Sampling took {round(time()-start_time, 4)} seconds", "yellow"))

    try:
        return sampling
    except UnboundLocalError as err:
        return True
