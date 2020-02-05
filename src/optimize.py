import scipy
import numpy as np
from scipy.optimize import least_squares

## Importing my code
from common.z3 import is_this_z3_function, translate_z3_function


def dist(x):
    """ Computes the distance between functions and data point.

    Args:
        x (list): point in parameter space

    Returns:
        (list): of distances of the function from the data point
    """
    # global functions
    # global data_point
    for index, param in enumerate(globals()["params"]):
        globals()[str(param)] = float(x[index])
    result = []
    for index, function in enumerate(globals()["functions"]):
        ## Function value - data point
        result.append(eval(function) - float(globals()["data_point"][index]))
    # print("semiresult", result)
    return np.array(result)


def optimize(functions: [list], params: [list], param_intervals: [list], data_point: [list]):
    """ Search for parameter values minimizing the distance of function to data.

    Args:
        functions (list): of functions to be optimized
        params (list): of functions parameters
        param_intervals (list): of intervals of functions parameters
        data_point (list): of values of functions to be optimized

    Returns:
        (list): [point of parameter space with the least distance, values of functions in the point, the distance between the data and functions values]
    """

    ## Convert z3 functions
    for index, function in enumerate(functions):
        if is_this_z3_function(function):
            functions[index] = translate_z3_function(function)

    ## Get the average value of parameter intervals
    x0 = np.array(list(map(lambda lst: (lst[0]+lst[1])/2, param_intervals)))

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

    res = scipy.optimize.least_squares(dist, x0, bounds=bounds)
    # print(res.x)

    function_values = []
    for polynome in functions:
        function_values.append(eval(polynome))

    ## VALUES OF PARAMS, VALUES OF FUNCTIONS, DISTANCE
    return res.x, function_values,  sum([abs(x - y) for x, y in zip(function_values, data_point)])
