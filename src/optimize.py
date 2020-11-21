import scipy
import numpy as np
from scipy.optimize import least_squares

## Importing my code
from termcolor import colored

from common.mathematics import weight_list
from common.my_z3 import is_this_z3_function, translate_z3_function

global params
global functions
global data_point


def dist(param_point):
    """ Computes the distance between functions and data point.

    Args:
        param_point (list): point in parameter space

    Returns:
        (list): of distances of the function from the data point
    """
    for index, param in enumerate(params):
        globals()[str(param)] = float(param_point[index])
    result = []
    for index, function in enumerate(functions):
        ## Function value - data point
        try:
            result.append(abs(eval(function) - float(data_point[index])))
        except TypeError as err:
            str(function)
            result.append(abs(eval(function) - float(data_point[index])))
            # if isinstance(function, sympy.core.numbers.Float):
            #     function = float(function)
            #     result.append(abs(eval(function) - float(data_point[index])))
            # else:
            #     print(colored(f"function {function}, type {type(function)}", "red"))
            #     print(colored(f"data_point[index] {data_point[index]}, type {type(data_point[index])}", "red"))
            #     print(colored(eval(function), "red"))
            #     raise err
    # print("semiresult", result)
    return np.array(result)


def weighted_dist(param_point, weights):
    return weight_list(dist(param_point), weights)


def optimize(functions: [list], params: [list], param_intervals: [list], data_point: [list], weights=False, debug=False):
    """ Search for parameter values minimizing the distance of function to data.

    Args:
        functions (list): of functions to be optimized
        params (list): of functions parameters
        param_intervals (list): of intervals of functions parameters
        data_point (list): of values of functions to be optimized
        weights (list): of weights to multiply the respective distance with
        debug (bool): if True extensive print will be used

    Returns:
        (list): [point of parameter space with the least distance, values of functions in the point, the distance between the data and functions values]
    """

    assert len(functions) == len(data_point)
    assert len(params) == len(param_intervals)
    if weights:
        assert len(weights) == len(data_point)

    ## Convert z3 functions
    for index, function in enumerate(functions):
        if is_this_z3_function(function):
            functions[index] = translate_z3_function(function)

    ## Get the average value of parameter intervals
    x0 = np.array(list(map(lambda lst: (lst[0] + lst[1]) / 2, param_intervals)))

    globals()["functions"] = functions
    # print("globals()[functions]", globals()["functions"])
    globals()["params"] = params
    # print("globals()[params]", globals()["params"])
    globals()["data_point"] = data_point
    # print("globals()[data_point]", globals()["data_point"])

    bounds = [[], []]
    for interval in param_intervals:
        bounds[0].append(interval[0])
        bounds[1].append(interval[1])
    # print("bounds", bounds)

    if debug:
        verbose = 2
    else:
        verbose = 0

    if weights:
        res = scipy.optimize.least_squares(weighted_dist, x0, bounds=bounds, args=[weights], verbose=verbose)
    else:
        res = scipy.optimize.least_squares(dist, x0, bounds=bounds, verbose=verbose)
    # print(res.x)

    ## VALUES OF PARAMS, VALUES OF FUNCTIONS, DISTANCE
    # print("point", list(res.x))
    ## function_values = res.fun
    ## for index, item in enumerate(function_values):
    ##     function_values[index] = function_values[index] + data_point[index]
    if debug:
        print(res)
        print("params", params)
        ## !!! THIS IS NOT UPDATED RESULT
        # spam = []
        # for item in params:
        #     spam.append(globals()[item])
        # print("param values", spam)
        print("param values", list(res.x))
        print("function values", list(map(eval, functions)))
        print("distance values", list(res.fun))
        print("total distance according to scipy - not transformed", res.cost)
        ## !!! THIS IS NOT UPDATED RESULT
        # print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
        # print("L2 distance", (sum([(x - y)**2 for x, y in zip(list(map(eval, functions)), data_point)]))**(1/2))

        for index, value in enumerate(params):
            globals()[str(value)] = res.x[index]
        print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
        print("L2 distance", (sum([(x - y)**2 for x, y in zip(list(map(eval, functions)), data_point)]))**(1/2))
    return list(res.x), list(map(eval, functions)), (2*res.cost)**(1/2)
