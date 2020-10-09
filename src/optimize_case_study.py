from math import sqrt

import scipy
import numpy as np
from scipy.optimize import least_squares, NonlinearConstraint, LinearConstraint, Bounds

## Importing my code
from termcolor import colored

from common.my_z3 import is_this_z3_function, translate_z3_function


def dist_l1(param_point):
    """ L1 norm distance """
    spam = dist(param_point)
    spam = list(map(lambda a: abs(float(a)), spam))
    return sum(spam)


def dist_l2(param_point):
    """ L2 norm distance """
    spam = dist(param_point)
    spam = list(map(lambda a: float(a*a), spam))
    return sqrt(sum(spam))


def dist2(param_point):
    return sum(dist(param_point))


def dist(param_point):
    """ Computes the distance between functions and data point.

    Args:
        param_point (list): point in parameter space

    Returns:
        (list): of distances of the function from the data point
    """
    # global functions
    # global data_point
    for index, param in enumerate(globals()["params"]):
        globals()[str(param)] = float(param_point[index])
    result = []
    for index, function in enumerate(globals()["functions"]):
        ## Function value - data point
        # print(f'{index} {abs(eval(function) - float(globals()["data_point"][index]))}')
        result.append(abs(eval(function) - float(globals()["data_point"][index])))
    # print("semiresult", result)
    return np.array(result)


def dist_single(param_point, parameter_values):
    """ Computes the distance between functions and data point.

    Args:
        param_point (list): point in parameter space
        parameter_values (list): values of other not fitting parameters

    Returns:
        (list): of distances of the function from the data point
    """
    # global functions
    # global data_point
    indexx = len(parameter_values)
    for index, param in enumerate(globals()["params"]):
        if index == indexx:
            globals()[str(param)] = float(param_point)
        else:
            try:
                globals()[str(param)] = parameter_values[index]
            except IndexError:
                pass
    result = [eval(globals()["functions"][indexx]) - float(globals()["data_point"][indexx])]
    # if indexx == len(globals()["functions"]) - 2:
    #     result[0] = result[0] + eval(globals()["functions"][indexx+1]) - float(globals()["data_point"][indexx+1])
    # for index, function in enumerate():
    #     ## Function value - data point
    #     # print(f'{index} {abs(eval(function) - float(globals()["data_point"][index]))}')
    #     if index == indexx:
    #         result.append(abs(eval(function) - float(globals()["data_point"][index])))
    #     else:
    #         break
    # # print("semiresult", result)
    return np.array(result)


def optimize_case_study(functions: [list], params: [list], param_intervals: [list], data_point: [list], sort=False, debug=False):
    """ Search for parameter values minimizing the distance of function to data.

    Args:
        functions (list): of functions to be optimized
        params (list): of functions parameters
        param_intervals (list): of intervals of functions parameters
        data_point (list): of values of functions to be optimized
        sort (bool): sort (Bool): tag whether the params are non-decreasing (CASE STUDY SPECIFIC SETTING)
        debug (bool): if True extensive print will be used

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

    # If accept only non-decreasing params
    if sort:
        low_bounds = []
        upper_bounds = []
        for interval in param_intervals:
            low_bounds.append(interval[0])
            upper_bounds.append(interval[1])
        bounds = Bounds(low_bounds, upper_bounds)
        if debug:
            print("bounds", bounds)

        ## THE USABLE SET OF METHODS
        methods = ["COBYLA", "SLSQP", "trust-constr"]

        ## Example for 10 bees
        # lc_10 = LinearConstraint(
        #     [[1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1]],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0])
        const = []
        for i in range(len(params)-1):
            a = np.zeros(len(params))
            a[i] = 1
            a[i+1] = -1
            const.append(a)
        low_const = list(map(lambda x: -x, np.ones(len(params) - 1)))
        high_const = np.zeros(len(params) - 1)
        if debug:
            print("const", const)
            print("low_const", low_const)
            print("high_const", high_const)
        lc = LinearConstraint(const, low_const, high_const)

        results = dict()
        for method in methods:
            # print("method", method)
            # res = scipy.optimize.minimize(dist_l2, x0, method=method, options={"maxiter": 100}, bounds=bounds, constraints=[lc])
            res = scipy.optimize.minimize(dist_l2, x0, method=method, bounds=bounds, constraints=[lc])
            results[method] = res

        for key in results.keys():
            print(colored(key, "red"))
            print(results[key])
            print()
            ## Find the method with minimal distance
            if results[key].fun < res.fun:
                res = results[key]

        if debug:
            print("param names", params)
            print("param values", list(res.x))
            print("function values", list(map(eval, functions)))
            for index, value in enumerate(params):
                locals()[str(value)] = res.x[index]
            print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
            print("L2 distance", (sum([(x - y) ** 2 for x, y in zip(list(map(eval, functions)), data_point)])) ** (1 / 2))

        ## parameter point, function values, distance
        return list(res.x), list(map(eval, functions)), res.fun
    else:
        bounds = [[], []]
        for interval in param_intervals:
            bounds[0].append(interval[0])
            bounds[1].append(interval[1])
        # print("bounds", bounds)

        # print(x0)
        # print(dist(x0))
        res = scipy.optimize.least_squares(dist, x0, bounds=bounds)
        if debug:
            print(res)
            print("param names", params)
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
            print("L2 distance",
                  (sum([(x - y) ** 2 for x, y in zip(list(map(eval, functions)), data_point)])) ** (1 / 2))
        return list(res.x), list(map(eval, functions)), (2 * res.cost) ** (1 / 2)
