from math import sqrt

import scipy
import numpy as np
from scipy.optimize import least_squares, NonlinearConstraint, LinearConstraint, Bounds

## Importing my code
from termcolor import colored

from common.my_z3 import is_this_z3_function, translate_z3_function
from common.mathematics import weight_list
from optimize import dist
import optimize


def dist_l1(param_point):
    """ L1 norm distance """

    optimize.functions = globals()["functions"]
    optimize.params = globals()["params"]
    optimize.data_point = globals()["data_point"]

    spam = dist(param_point)
    spam = list(map(lambda a: abs(float(a)), spam))
    return sum(spam)


def weighted_dist_l1(param_point, weights):
    """ L1 weighted distance """
    optimize.functions = globals()["functions"]
    optimize.params = globals()["params"]
    optimize.data_point = globals()["data_point"]

    spam = weight_list(dist(param_point), weights)
    return sum(spam)


def dist_l2(param_point):
    """ L2 norm distance """
    optimize.functions = globals()["functions"]
    optimize.params = globals()["params"]
    optimize.data_point = globals()["data_point"]

    spam = dist(param_point)
    spam = list(map(lambda a: float(a*a), spam))
    return sqrt(sum(spam))


def weighted_dist_l2(param_point, weights):
    """ L2 weighted distance """
    spam = dist(param_point)
    spam = list(map(lambda a: float(a * a), spam))
    spam = weight_list(spam, weights)
    return sqrt(sum(spam))


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


def optimize_case_study(functions: [list], params: [list], param_intervals: [list], data_point: [list], weights=False, sort=False, debug=False):
    """ Search for parameter values minimizing the distance of function to data.

    Args:
        functions (list): of functions to be optimized
        params (list): of functions parameters
        param_intervals (list): of intervals of functions parameters
        data_point (list): of values of functions to be optimized
        weights (list): of weights to multiply the respective distance with
        sort (bool): sort (Bool): tag whether the params are non-decreasing (CASE STUDY SPECIFIC SETTING)
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
    x0 = np.array(list(map(lambda lst: (lst[0]+lst[1])/2, param_intervals)))

    optimize.functions = functions
    globals()["functions"] = functions
    # print("globals()[functions]", globals()["functions"])
    optimize.params = params
    globals()["params"] = params
    # print("globals()[params]", globals()["params"])
    optimize.data_point = data_point
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
            if weights:
                res = scipy.optimize.minimize(weighted_dist_l2, x0, args=weights, method=method, bounds=bounds, constraints=[lc])
            else:
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
            print("data values", data_point)
            for index, value in enumerate(params):
                locals()[str(value)] = res.x[index]
            print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
            print("L2 distance", (sum([(x - y) ** 2 for x, y in zip(list(map(eval, functions)), data_point)])) ** (1 / 2))

        ## parameter point, function values, distance
        for index, item in enumerate(params):
            locals()[item] = list(res.x)[index]
        return list(res.x), list(map(eval, functions)), res.fun
    else:
        # bounds = [[], []]
        # for interval in param_intervals:
        #     bounds[0].append(interval[0])
        #     bounds[1].append(interval[1])
        # print("bounds", bounds)
        bounds = param_intervals

        # print(x0)
        # print(dist(x0))
        ## THE USABLE SET OF METHODS
        # methods = ["trf", "dogbox"]  ## lm cannot handle bounds
        methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'trust-exact', 'trust-krylov']  # 'Powell', 'Nelder-Mead', 'CG', 'BFGS', Newton-CG, COBYLA, dogleg, trust-ncg  cannot handle bounds

        results = dict()
        for method in methods:
            # res = scipy.optimize.least_squares(dist, x0, method=method, bounds=bounds, verbose=verbose)
            try:
                if weights:
                    spam = scipy.optimize.minimize(weighted_dist_l2, x0, args=weights, method=method, bounds=bounds)
                else:
                    spam = scipy.optimize.minimize(dist_l2, x0, method=method, bounds=bounds)
            except ValueError as err:
                if debug:
                    print(err)
                continue

            res = spam
            results[method] = res

            if debug:
                ## Update param point
                for index, item in enumerate(params):
                    locals()[item] = list(res.x)[index]

                print(res)
                print("param names", params)
                ## !!! THIS IS NOT UPDATED RESULT
                # spam = []
                # for item in params:
                #     spam.append(globals()[item])
                # print("param values", spam)
                print("param values", list(res.x))
                print("function values", list(map(eval, functions)))
                print("data values", data_point)
                print("total distance according to scipy - not transformed", res.fun)
                ## !!! THIS IS NOT UPDATED RESULT
                # print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
                # print("L2 distance", (sum([(x - y)**2 for x, y in zip(list(map(eval, functions)), data_point)]))**(1/2))

                for index, value in enumerate(params):
                    globals()[str(value)] = res.x[index]
                print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
                print("L2 distance",
                      (sum([(x - y) ** 2 for x, y in zip(list(map(eval, functions)), data_point)])) ** (1 / 2))

        for key in results.keys():
            print(colored(key, "red"))
            print(results[key])
            print()
            ## Find the method with minimal distance
            if results[key].fun < res.fun:
                res = results[key]

        ## Update param point
        for index, item in enumerate(params):
            locals()[item] = list(res.x)[index]
        return list(res.x), list(map(eval, functions)), res.fun
