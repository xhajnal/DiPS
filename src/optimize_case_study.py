from math import sqrt

import scipy
import numpy as np
from scipy.optimize import least_squares, NonlinearConstraint

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
        ## Vector of parameter values and function values to store
        parameter_values = []
        function_values = []
        bounds = [[], []]
        bounds[0].append(param_intervals[0][0])
        bounds[1].append(param_intervals[0][1])
        x0 = [x0[0]]
        distance = 0

        ## For each function we run least_squares separately
        for indexx, function in enumerate(functions):
            ## Non decreasing values by setting lower bound
            for index, value in enumerate(parameter_values):
                if index >= 1:
                    bounds = [[], []]
                    bounds[0].append(parameter_values[-1])
                    bounds[1].append(param_intervals[index][1])
                    ## x0 = [(parameter_values[-1] + param_intervals[index][1])/2]
                    x0 = [parameter_values[-1]]
            try:
                res = scipy.optimize.least_squares(dist_single, x0, bounds=bounds, args=[parameter_values])
            except ValueError as error:
                print(error)
                print("point: ", x0)
                print("bounds: ", bounds)
            print(res)
            function_values.append(float(res.fun))
            parameter_values.append(float(res.x))
            distance = distance + res.cost

        parameter_values = parameter_values[:-1]
        # print(list(res.x), list(function_values), res.cost)

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
                globals()[str(value)] = parameter_values
            print("L1 distance", sum([abs(x - y) for x, y in zip(list(map(eval, functions)), data_point)]))
            print("L2 distance",
                  (sum([(x - y) ** 2 for x, y in zip(list(map(eval, functions)), data_point)])) ** (1 / 2))
        return parameter_values, list(map(eval, functions)), (2 * distance) ** (1 / 2)

        #
        # bounds = []
        # for interval in param_intervals:
        #     bounds.append(interval)
        # # print("bounds", bounds)
        #
        # def is_nondecreasing(param_point):
        #     if np.array_equal(param_point, sorted(param_point)):
        #         return 0
        #     else:
        #         return -1
        #
        # # constraints = [{'type': 'eq', 'fun': is_nondecreasing}]
        #
        # ## THE USABLE SET OF METHODS = ["COBYLA", "SLSQP", "trust-constr"]
        #
        # results = dict()
        # for method in ["SLSQP"]:
        #     # print("method", method)
        #     nlc = NonlinearConstraint(is_nondecreasing, 0 - 1e-12, 0 + 1e-12, keep_feasible=True)
        #     res = scipy.optimize.minimize(dist_l2, x0, method=method, options={"maxiter": 1000}, bounds=bounds, constraints=nlc)
        #     # print(res)
        #     results[method] = res
        #
        # for key in results.keys():
        #     print(colored(key, "grey"))
        #     print(results[key])
        #     print()
        #
        # results = dict()
        # for method in ["SLSQP"]:
        #     # print("method", method)
        #     nlc = NonlinearConstraint(is_nondecreasing, 0 - 1e-12, 0 + 1e-12, keep_feasible=False)
        #     res = scipy.optimize.minimize(dist_l2, x0, method=method, options={"maxiter": 50000}, bounds=bounds, constraints=nlc)
        #     # print(res)
        #     results[method] = res
        #
        # for key in results.keys():
        #     print(colored(key, "red"))
        #     print(results[key])
        #     print()
        #
        # results = dict()
        # for method in ["SLSQP"]:
        #     # print("method", method)
        #     constraints = []
        #     constraints.append({'type': 'ineq', 'fun': is_nondecreasing})
        #     res = scipy.optimize.minimize(dist_l2, x0, method=method, options={"maxiter": 1000}, bounds=bounds, constraints=constraints)
        #     # print(res)
        #     results[method] = res
        #
        # for key in results.keys():
        #     print(colored(key, "blue"))
        #     print(results[key])
        #     print()
        #
        # results = dict()
        # for method in ["SLSQP"]:
        #     # print("method", method)
        #     constraints = []
        #     constraints.append({'type': 'eq', 'fun': is_nondecreasing})
        #     res = scipy.optimize.minimize(dist_l2, x0, method=method, options={"maxiter": 1000}, bounds=bounds, constraints=constraints)
        #     # print(res)
        #     results[method] = res
        #
        # for key in results.keys():
        #     print(colored(key, "green"))
        #     print(results[key])
        #     print()
        #
        # function_values = []
        # for index, param in enumerate(globals()["params"]):
        #     globals()[str(param)] = float(res.x[index])
        # for index, function in enumerate(globals()["functions"]):
        #     function_values.append(eval(function))
        # return list(res.x), list(function_values), res.fun
    else:
        bounds = [[], []]
        for interval in param_intervals:
            bounds[0].append(interval[0])
            bounds[1].append(interval[1])
        # print("bounds", bounds)

        #print(x0)
        #print(dist(x0))
        res = scipy.optimize.least_squares(dist, x0, bounds=bounds)
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
            print("L2 distance",
                  (sum([(x - y) ** 2 for x, y in zip(list(map(eval, functions)), data_point)])) ** (1 / 2))
        return list(res.x), list(map(eval, functions)), (2 * res.cost) ** (1 / 2)
