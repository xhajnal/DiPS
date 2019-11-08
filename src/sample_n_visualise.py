import random
from mpmath import mpi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections as mc
from matplotlib.ticker import MaxNLocator
from termcolor import colored
from z3 import *

from common.z3 import is_this_z3_function, translate_z3_function

workspace = os.path.dirname(__file__)
sys.path.append(os.path.join(workspace, '../src/'))
# sys.path.append(os.path.dirname(__file__))
from load import find_param
from common.document_wrapper import DocumentWrapper
from common.math import cartesian_product

wraper = DocumentWrapper(width=60)


def get_param_values(parameters, size_q, intervals=False, debug=False):
    """ Creates linearly sampled parameter space from the given parameter intervals and number of samples

    Args
    ----------
    parameters: (list) of parameters to sample
    size_q: (int) sample size in each parameter
    intervals: (Bool) if False (0,1) interval is used
    debug: (Bool) if debug extensive output is provided
    """
    parameter_values = []
    for param in range(len(parameters)):
        if intervals:
            if debug:
                print(f"Parameter index:{param} with intervals: [{intervals[param][0]},{intervals[param][1]}]")
            parameter_values.append(np.linspace(intervals[param][0], intervals[param][1], size_q, endpoint=True))
        else:
            parameter_values.append(np.linspace(0, 1, size_q, endpoint=True))
    parameter_values = cartesian_product(*parameter_values)
    if (len(parameters) - 1) == 0:
        parameter_values = np.linspace(0, 1, size_q, endpoint=True)[np.newaxis, :].T
    if debug:
        print("Parameter_values: ", parameter_values)
    return parameter_values


def eval_and_show(functions, parameter_value, parameters=False, data=False, data_intervals=False, cumulative=False, debug=False, where=False):
    """ Creates bar plot of evaluation of given functions for given point in parameter space

    Args
    ----------
    functions: (list of strings) list of rational functions
    parameter_value: (list of floats) array of param values
    parameters: (list of strings) parameter names (used for faster eval)
    data: (list) Data comparison next to respective function
    data_intervals: (list) intervals obtained from the data to check if the function are within the intervals
    cumulative: (Bool) if True cdf instead of pdf is visualised
    debug: (Bool) if debug extensive output is provided
    where: (Tuple/List) : output matplotlib sources to output created figure
    """

    ## Convert z3 functions
    for index, function in enumerate(functions):
        if is_this_z3_function(function):
            functions[index] = translate_z3_function(function)

    if not parameters:
        parameters = set()
        for polynome in functions:
            parameters.update(find_param(polynome, debug))
        parameters = sorted(list(parameters))
    if debug:
        print("Parameters: ", parameters)

    title = "Rational functions sampling \n parameter values:"
    values = []
    add = 0
    for param in range(len(parameters)):
        if debug:
            print("Parameters[param]", parameters[param])
            print("Parameter_value[param]", parameter_value[param])
        globals()[parameters[param]] = parameter_value[param]
        title = "{} {}={},".format(title, parameters[param], parameter_value[param])
    title = title[:-1]

    title = f"{title}\n values: "
    for polynome in functions:
        expression = eval(polynome)
        if debug:
            print(polynome)
            print("Eval ", polynome, expression)
        if cumulative:
            ## Add sum of all values
            add = add + expression
            values.append(add)
            title = f"{title} {add} , "
            del expression
        else:
            values.append(expression)
            title = f"{title} {expression} ,"
    title = title[:-2]
    if data:
        title = f"{title}\n Comparing with the data: \n{data}"
        if cumulative:
            for index in range(1, len(data)):
                data[index] = data[index] + data[index - 1]

    if where:
        fig = where[0]
        ax = where[1]
        plt.autoscale()
        ax.autoscale()
    else:
        fig, ax = plt.subplots()
    width = 0.2
    ax.set_ylabel(f'{("Value", "Cumulative value")[cumulative]}')
    if data:
        ax.set_xlabel('Rational function indices (blue), Data point indices (red)')
        if data_intervals:
            functions_inside_of_intervals = []
            for index in range(len(data)):
                if values[index] in data_intervals[index]:
                    functions_inside_of_intervals.append(True)
                else:
                    functions_inside_of_intervals.append(False)
            title = f"{title} \n Function value within the respective interval:\n {functions_inside_of_intervals} \n Intervals: {data_intervals}"
    else:
        ax.set_xlabel('Rational function indices')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if debug:
        print("title: \n", title)
    ax.set_title(wraper.fill(title))
    if debug:
        print("Len(fun_list): ", len(functions))
        print("values:", values)
        print("range:", range(1, len(values) + 1))
        print("title", title)
    ax.bar(range(1, len(values) + 1), values, width, color='b')
    if data:
        ax.bar(list(map(lambda x: x + width, range(1, len(data) + 1))), data, width, color='r')

    if where:
        return fig, ax
    else:
        plt.show()
    return values


def sample_dictionary_funs(dictionary, size_q, keys=None, debug=False):
    """ Returns a dictionary of function values for sampled parametrisations

    Args
    ----------
    dictionary: dictionary of list of functions
    size_q: (int) sample size in each parameter
    keys: (list) dictionary keys to be used
    debug: (Bool) if debug extensive output is provided

    Returns
    ----------
    Array of [agents_quantity, function index, [parameter values], function value]

    """
    arr = []
    if debug:
        print("Inside of sample_n_visualise.sample_dictionary_funs()")
        print("Dictionary of functions: ", dictionary)

    ## Initialisation
    sampling = {}
    if keys is None:
        keys = dictionary.keys()

    ## For
    for key in keys:
        array = sample_list_funs(dictionary[key], size_q, debug=debug)
        sampling[key] = array
    return sampling


def sample_list_funs(functions, size_q, parameters=False, intervals=False, silent=False, debug=False):
    """ Returns a list of function values for sampled parametrisations

    Args
    ----------
    functions: (list of functions) to be sampled
    size_q: (int) sample size in each parameter
    parameters: (list of strings) parameter names (used for faster eval)
    intervals: (list of pairs of numbers) intervals of parameters
    silent: (Bool) if silent command line output is set to minimum
    debug: (Bool) if debug extensive output is provided

    Returns
    ----------
    Array of [function index, [parameter values], function value]

    """
    arr = []
    if debug:
        print("Inside of sample_n_visualise.sample_list_funs()")
        print("List_fun: ", functions)
        print("Intervals: ", intervals)

    for index, polynome in enumerate(functions):
        if is_this_z3_function(polynome):
            functions[index] = translate_z3_function(polynome)
        if debug:
            print("Polynome: ", polynome)

        if parameters:
            fun_parameters = parameters
        else:
            fun_parameters = set()
            fun_parameters.update(find_param(polynome, debug))

            ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
            # if len(parameters) < N:
            #     parameters.update(find_param(polynome, debug))
            if debug:
                print("Parameters: ", fun_parameters)
            fun_parameters = sorted(list(fun_parameters))

        if debug:
            print("Sorted parameters: ", fun_parameters)

        parameter_values = get_param_values(fun_parameters, size_q, intervals=intervals, debug=debug)

        for parameter_value in parameter_values:
            if debug:
                print("Parameter_value: ", parameter_value)
            a = [functions.index(polynome)]
            for param_index in range(len(fun_parameters)):
                a.append(parameter_value[param_index])
                if debug:
                    print("Parameter[param]: ", fun_parameters[param_index])
                    print("Parameter_value[param]: ", parameter_value[param_index])
                locals()[fun_parameters[param_index]] = float(parameter_value[param_index])

            print("polynome", polynome)

            value = eval(polynome)
            if debug:
                print("Eval ", polynome, value)
            a.append(value)
            arr.append(a)
    return arr


def visualise(dic_fun, agents_quantities, size_q, cumulative=False, debug=False, show_all_in_one=False, where=False):
    """ Creates bar plot of probabilities of i successes for sampled parametrisation

    Args
    ----------
    dic_fun: (dictionary N -> list of rational functions)
    size_q: (int) sample size in each parameter
    agents_quantities: (int) pop sizes to be used
    cumulative: (Bool) if True cdf instead of pdf is visualised
    debug: (Bool) if debug extensive output is provided
    show_all_in_one: (Bool) if True all plots are put into one window
    where: (Tuple/List) : output matplotlib sources to output created figure
    """

    for N in agents_quantities:
        parameters = set()
        for index, polynome in enumerate(dic_fun[N]):
            if is_this_z3_function(polynome):
                dic_fun[N][index] = translate_z3_function(polynome)

            if debug:
                print("Polynome: ", polynome)
            parameters.update(find_param(polynome, debug))

            ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
            # if len(parameters) < N:
            #    parameters.update(find_param(polynome, debug))
        if debug:
            print("Parameters: ", parameters)
        parameters = sorted(list(parameters))
        if debug:
            print("Sorted parameters: ", parameters)

        parameter_values = get_param_values(parameters, size_q, debug)

        for parameter_value in parameter_values:
            if debug:
                print("Parameter_value: ", parameter_value)
            add = 0
            a = [N, dic_fun[N].index(polynome)]
            if N == 0:
                title = f"Rational functions sampling \n parameters:"
            else:
                title = f"Rational functions sampling \n N={N}, parameters:"
            for param in range(len(parameters)):
                a.append(parameter_value[param])
                if debug:
                    print("Parameter[param]: ", parameters[param])
                    print("Parameter_value[param]: ", parameter_value[param])
                globals()[parameters[param]] = parameter_value[param]
                title = "{} {}={},".format(title, parameters[param], parameter_value[param])
            title = title[:-1]
            if debug:
                print("Eval ", polynome, eval(polynome))
            for polynome in dic_fun[N]:
                value = eval(polynome)
                if cumulative:
                    ## Add sum of all values
                    add = add + value
                    a.append(add)
                    del value
                else:
                    a.append(value)

            # print(a)
            fig, ax = plt.subplots()
            width = 0.2
            ax.set_ylabel('Value')
            ax.set_xlabel('Rational function indices')
            ax.set_title(wraper.fill(title))
            # print(title)
            rects1 = ax.bar(range(len(dic_fun[N])), a[len(parameters) + 2:], width, color='b')
            plt.show()


## SOURCE: https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
def visualise_by_param(hyper_rectangles, title="", where=False):
    """
    Visualises domain intervals of each dimension in a plot.

    Args
    ----------
    hyper_rectangles: list of (hyper)rectangles
    title: (String) title used for the Figure
    where: (Tuple/List) : output matplotlib sources to output created figure
    """
    from sympy import Interval

    if hyper_rectangles:
        lines = []
        intervals = []
        for i in range(len(hyper_rectangles[0])):
            intervals.append([])
            for j in range(len(hyper_rectangles)):
                # print(hyper_rectangles_sat[j][i])
                intervals[i].append(Interval(hyper_rectangles[j][i][0], hyper_rectangles[j][i][1]))
                if len(intervals[i]) == 2:
                    intervals[i] = [intervals[i][0].union(intervals[i][1])]
                lines.append([(i + 1, hyper_rectangles[j][i][0]), (i + 1, hyper_rectangles[j][i][1])])
                # print([(i+1, hyper_rectangles_sat[j][i][0]), (i+1, hyper_rectangles_sat[j][i][1])])
        c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

        lc = mc.LineCollection(lines, color='g', linewidths=2)

        if where:
            fig, ax = where
        else:
            fig, ax = plt.subplots()

        ax.set_xlabel('Parameter indices')
        ax.set_ylabel('Parameter values')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if title:
            ax.set_title(wraper.fill(title))
        else:
            ax.set_title(wraper.fill("Domain in which respective parameter belongs to in the given space"))

        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        print("Intervals: ", intervals)
        if where:
            return fig, ax
        else:
            plt.show()
    else:
        print(colored("Given space is empty, no intervals to be visualised", "red"))
        if where:
            return None


def heatmap(function, region, sampling_sizes, posttitle="", where=False, parameters=False):
    """ Creates 2D heatmap plot of sampled points of given function

    Args
    ----------
    function: (string) function to be analysed
    region: (list of intervals) boundaries of parameter space to be sampled
    sampling_sizes: (int) tuple of sample size of respective parameter
    posttitle: (string) A string to be put after the title
    where: (Tuple/List) : output matplotlib sources to output created figure
    parameters: (list): list of parameters

    Example
    ----------
    heatmap("p+q",[[0,1],[3,4]],[5,5])
    """

    ## Convert z3 function
    if is_this_z3_function(function):
        function = translate_z3_function(function)

    if not parameters:
        parameters = sorted(list(find_param(function)))
    # print(parameters)
    if len(parameters) != 2:
        raise Exception(f"Number of parameters of given function is not equal to 2 but {len(parameters)}")

    arr = np.zeros((sampling_sizes[0] * sampling_sizes[1], 3))

    # f = lambda locals()[parameters[0]),locals()[parameters[1]): fun

    ii = -1
    jj = -1
    for i in np.linspace(region[0][0], region[0][1], sampling_sizes[0], endpoint=True):
        ii += 1
        # print("ii: ",ii)
        locals()[parameters[0]] = i
        for j in np.linspace(region[1][0], region[1][1], sampling_sizes[1], endpoint=True):
            jj += 1
            # print("jj: ",jj)
            locals()[parameters[1]] = j
            arr[jj, 0] = round(i, 2)
            arr[jj, 1] = round(j, 2)
            arr[jj, 2] = eval(function)
    # print(arr)
    # d = pd.DataFrame(arr, columns=["p","q","E"])
    heatmap_data = pd.DataFrame(arr, columns=[parameters[0], parameters[1], "E"])
    # d = d.pivot("p", "q", "E")
    heatmap_data = heatmap_data.pivot(parameters[0], parameters[1], "E")

    if where:
        f, ax = plt.subplots()
        ax = sns.heatmap(heatmap_data)
        title = f"Heatmap \n{posttitle}"
        ax.set_title(wraper.fill(title))
        ax.invert_yaxis()
        return f
    else:
        ax = sns.heatmap(heatmap_data)
        title = f"Heatmap of the parameter space \n function: {function}"
        ax.set_title(wraper.fill(title))
        ax.invert_yaxis()
        plt.show()


def visualise_sampled_by_param(hyper_rectangles, sample_size):
    """
    Visualises sampled hyperspace by connecting the values in each dimension.

    Args
    ----------
    hyper_rectangles: (list of hyperrectangles)
    sample_size: (int): number of points to be sampled
    """
    if hyper_rectangles:
        fig, ax = plt.subplots()
        ## Creates values of the horizontal axis
        x_axis = []
        i = 0
        for dimension in hyper_rectangles[0]:
            i = i + 1
            x_axis.append(i)
        ## Get values of the vertical axis for respective line
        for sample in range(sample_size):
            rectangle = random.randint(0, len(hyper_rectangles) - 1)
            # print(rectangle)
            values = []
            # print(hyper_rectangles[rectangle])
            for dimension in range(len(hyper_rectangles[rectangle])):
                # print(hyper_rectangles[rectangle][dimension])
                values.append(random.uniform(hyper_rectangles[rectangle][dimension][0],
                                             hyper_rectangles[rectangle][dimension][1]))
            ax.scatter(x_axis, values)
            ax.plot(x_axis, values)
        ax.set_xlabel("Parameter indices")
        ax.set_ylabel("Parameter values")
        ax.set_title("Sample points of the given hyperspace")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.autoscale()
        ax.margins(0.1)
        plt.show()
    else:
        print(colored("Given space is empty.", "red"))
