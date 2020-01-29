import os
import pickle
from time import time, strftime, localtime
import numpy as np

from common.math import create_matrix, cartesian_product
from common.z3 import is_this_z3_function, translate_z3_function
from common.config import load_config

spam = load_config()
results_dir = spam["results"]
refinement_results = spam["refinement_results"]
refine_timeout = spam["refine_timeout"]
z3_path = spam["z3_path"]
del spam


def sample(space, constraints, sample_size, compress=False, silent=True, save=False, debug: bool = False, progress=False):
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

    Returns:
        (dict) of point to list of Bools whether f(point) in interval[index]
    """
    if debug:
        silent = False

    ## Convert z3 functions
    for index, constraint in enumerate(constraints):
        if is_this_z3_function(constraint):
            constraints[index] = translate_z3_function(constraint)

    start_time = time()
    parameter_values = []
    parameter_indices = []
    if debug:
        print("space.params", space.params)
        print("space.region", space.region)
        print("sample_size", sample_size)
    for param in range(len(space.params)):
        parameter_values.append(np.linspace(space.region[param][0], space.region[param][1], sample_size, endpoint=True))
        parameter_indices.append(np.asarray(range(0, sample_size)))

    sampling = create_matrix(sample_size, len(space.params))
    if not silent:
        print("sampling here")
        print("sample_size", sample_size)
        print("space.params", space.params)
        print("sampling", sampling)
    parameter_values = cartesian_product(*parameter_values)
    parameter_indices = cartesian_product(*parameter_indices)

    # if (len(space.params) - 1) == 0:
    #    parameter_values = linspace(0, 1, sample_size, endpoint=True)[newaxis, :].T
    if not silent:
        print("parameter_values", parameter_values)
        print("parameter_indices", parameter_indices)
        # print("a sample:", sampling[0][0])
    parameter_index = 0
    ## For each parametrisation eval the constraints
    for index, parameter_value in enumerate(parameter_values):
        ## For each parameter set the current sample point value
        if progress:
            progress(index / len(parameter_values))
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

        satisfied_list = []
        ## For each constraint (inequality - interval bound)
        for index, constraint in enumerate(constraints):
            # print(constraint)
            # print("type(constraint[index])", type(constraint))
            # for param in range(len(space.params)):
            #     print(space.params[param], parameter_value[param])
            #     print("type(space.params[param])", type(space.params[param]))
            #     print("type(parameter_value[param])", type(parameter_value[param]))

            if debug:
                print(f"constraints[{index}]", constraint)
                print(f"eval(constraints[{index}])", eval(constraint))

            satisfied_list.append(eval(constraint))

            ## print("cycle")
            ## print(sampling[tuple(parameter_indices[i])])

        if False in satisfied_list:
            # print("adding unsat", sampling[tuple(parameter_indices[i])][0])
            space.add_unsat_samples([sampling[tuple(parameter_indices[parameter_index])][0]])
            if compress:
                sampling[tuple(parameter_indices[parameter_index])][1] = False
            else:
                sampling[tuple(parameter_indices[parameter_index])][1] = satisfied_list
        else:
            # print("adding sat", sampling[tuple(parameter_indices[i])][0])
            space.add_sat_samples([sampling[tuple(parameter_indices[parameter_index])][0]])
            if compress:
                sampling[tuple(parameter_indices[parameter_index])][1] = True
            else:
                sampling[tuple(parameter_indices[parameter_index])][1] = satisfied_list
        parameter_index = parameter_index + 1

    ## Setting flag to not visualise sat if no unsat and vice versa
    space.gridsampled = True

    ## Saving the sampled space as pickled dictionary
    if save:
        if save is True:
            save = str(strftime("%d-%b-%Y-%H-%M-%S", localtime()))
        pickle.dump(sampling, open(os.path.join(refinement_results, ("Sampled_space_" + save).split(".")[0] + ".p"), "wb"))

    space.sampling_took(time() - start_time)
    return sampling