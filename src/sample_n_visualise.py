import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections as mc
from matplotlib.ticker import MaxNLocator, FixedLocator
from termcolor import colored

## Importing my code
from common.convert import to_sympy_intervals
from common.my_z3 import is_this_z3_function, translate_z3_function
from load import find_param
from common.document_wrapper import DocumentWrapper
from common.mathematics import cartesian_product

wraper = DocumentWrapper(width=60)


def bar_err_plot(data, intervals=None, titles=""):
    """ Creates bar plot (with errors)

    Args:
        data (list of floats): values to barplot
        intervals (list of tuples or list of floats or Intervals): if False (0,1) interval is used, if [] no intervals used
        titles (list of strings): (xlabel, ylabel, title)
    """
    if intervals is None:
        intervals = []
    if titles == "":
        if intervals:
            titles = ["Data indices", "Data values", "Data barplot with data intervals as error bars"]
        else:
            titles = ["Data indices", "Data values", "Data barplot"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(FixedLocator(range(1, len(data)+1)))
    if intervals == []:
        ax.bar(list(range(1, len(data) + 1)), data, color='r')
    else:
        ## Uniformize data intervals
        intervals = to_sympy_intervals(intervals)
        errors = [[], []]
        for index, item in enumerate(intervals):
            errors[0].append(float(abs(data[index] - item.start)))
            errors[1].append(float(abs(data[index] - item.end)))

        ax.bar(list(range(1, len(data) + 1)), data, yerr=errors, color='r', capsize=10)

    ax.set_xticks(list(range(1, len(data) + 1)))
    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])
    ax.set_title(titles[2])
    fig.show()


def get_param_values(parameters, sample_size, intervals=False, debug: bool = False):
    """ Creates linearly sampled parameter space from the given parameter intervals and number of samples

    Args:
        parameters (list): of parameters to sample
        sample_size (int): sample size in each parameter
        intervals (list of tuples or False): if False (0,1) interval is used
        debug (bool): if debug extensive output is provided
    """
    parameter_values = []
    for param in range(len(parameters)):
        if intervals:
            if debug:
                print(f"Parameter index:{param} with intervals: [{intervals[param][0]},{intervals[param][1]}]")
            parameter_values.append(np.linspace(intervals[param][0], intervals[param][1], sample_size, endpoint=True))
        else:
            parameter_values.append(np.linspace(0, 1, sample_size, endpoint=True))
    parameter_values = cartesian_product(*parameter_values)
    if (len(parameters) - 1) == 0:
        parameter_values = np.linspace(0, 1, sample_size, endpoint=True)[np.newaxis, :].T
    if debug:
        print("Parameter_values: ", parameter_values)
    return parameter_values


def eval_and_show(functions, parameter_value, parameters=False, data=False, data_intervals=False, cumulative=False, debug: bool = False, where=False):
    """ Creates bar plot of evaluation of given functions for given point in parameter space

    Args:
        functions (list of strings): list of rational functions
        parameter_value: (list of numbers) array of param values
        parameters (list of strings): parameter names (used for faster eval)
        data (list of floats): Data comparison next to respective function
        data_intervals (list of Intervals): intervals obtained from the data to check if the function are within the intervals
        cumulative (bool): if True cdf instead of pdf is visualised
        debug (bool): if debug extensive output is provided
        where (tuple or list): output matplotlib sources to output created figure
    """
    ## Convert z3 functions
    for index, function in enumerate(functions):
        if is_this_z3_function(function):
            functions[index] = translate_z3_function(function)

    if not parameters:
        parameters = set()
        for function in functions:
            parameters.update(find_param(function, debug))
        parameters = sorted(list(parameters))
    if debug:
        print("Parameters: ", parameters)

    if data:
        ## Check the sizes of data and functions
        if len(data) != len(functions):
            raise Exception(f"Number of data points, {len(data)}, is not equal to number of functions, {len(functions)}.")
        title = "Rational functions and data \n Parameter values:"
    else:
        title = "Rational functions \n Parameter values:"
    function_values = []
    add = 0
    for param in range(len(parameters)):
        if debug:
            print("Parameters[param]", parameters[param])
            print("Parameter_value[param]", parameter_value[param])
        globals()[parameters[param]] = parameter_value[param]
        title = f"{title} {parameters[param]}={parameter_value[param]},"
    title = title[:-1]

    if data_intervals:
        ## Check the sizes of data and functions
        if len(data_intervals) != len(functions):
            raise Exception(f"Number of data intervals, {len(data_intervals)}, is not equal to number of functions, {len(functions)}.")
        ## Uniformize the intervals
        data_intervals = to_sympy_intervals(data_intervals)

    title = f"{title}\n Function values: "
    for function in functions:
        expression = eval(function)
        if debug:
            print(function)
            print("Eval ", function, expression)
        if cumulative:
            ## Add sum of all values
            add = add + expression
            function_values.append(add)
            title = f"{title} {add} , "
            del expression
        else:
            function_values.append(expression)
            title = f"{title} {expression} ,"
    title = title[:-2]
    if data:
        # data_to_str = str(data).replace(" ", "\u00A0") does not work
        title = f"{title}\n Data values: {str(data)[1:-1]}"
        if cumulative:
            for index in range(1, len(data)):
                data[index] = data[index] + data[index - 1]
        distance = 0
        for index in range(len(data)):
            try:
                distance = distance + (eval(functions[index]) - data[index])**2
            except IndexError as error:
                raise Exception(f"Unable to show the intervals on the plot. Number of data point ({len(data)}) is not equal to number of functions ({len(functions)}).")
        title = f"{title}\n L2 Distance: {distance}"

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
        ax.set_xlabel('Rational function indices (blue bars), Data point indices (red bars)')
        # if data_intervals:
        #     functions_inside_of_intervals = []
        #     for index in range(len(data)):
        #         try:
        #             if function_values[index] in data_intervals[index]:
        #                 functions_inside_of_intervals.append(True)
        #             else:
        #                 functions_inside_of_intervals.append(False)
        #         except IndexError as error:
        #             raise Exception(f"Unable to show the intervals on the plot. Number of data intervals ({len(data_intervals)}) is not equal to number of functions ({len(functions)}).")
        #
        #     # functions_inside_of_intervals_to_str = str(functions_inside_of_intervals).replace(" ", "\u00A0") - does not work
        #     title = f"{title} \n Function value within the respective interval: {functions_inside_of_intervals}"
    else:
        ax.set_xlabel('Rational function indices')
    ax.xaxis.set_major_locator(FixedLocator(range(1, len(function_values)+1)))
    ax.bar(range(1, len(function_values) + 1), function_values, width, color='b', label="function")
    if data:
        if data_intervals:
            # np.array(list(map(lambda x: np.array([x.start, x.end]), data_intervals))) # wrong shape (N,2) instead of (2,N)
            errors = [[], []]
            for index, item in enumerate(data_intervals):
                errors[0].append(float(abs(data[index] - item.start)))
                errors[1].append(float(abs(data[index] - item.end)))

            ax.bar(list(map(lambda x: x + width, range(1, len(data) + 1))), data, width, yerr=errors, color='r', capsize=10, label="data")
            title = f"{title}\n Data intervals visualised as error bars."
        else:
            ax.bar(list(map(lambda x: x + width, range(1, len(data) + 1))), data, width, color='r', label="data")
    ax.set_title(wraper.fill(title))
    # fig.legend()
    if debug:
        print("Len(fun_list): ", len(functions))
        print("values:", function_values)
        print("range:", range(1, len(function_values) + 1))
        print("title", title)
    if where:
        return fig, ax
    else:
        plt.show()
    return function_values


def sample_dictionary_funs(dictionary, sample_size, keys=None, debug: bool = False):
    """ Returns a dictionary of function values for sampled parametrisations

    Args:
        dictionary: dictionary of list of functions
        sample_size (int): sample size in each parameter
        keys (list): dictionary keys to be used
        debug (bool): if debug extensive output is provided

    Returns:
        (array): [agents_quantity, function index, [parameter values], function value]

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
        array = sample_list_funs(dictionary[key], sample_size, debug=debug)
        sampling[key] = array
    return sampling


def sample_list_funs(functions, sample_size, parameters=False, intervals=False, silent: bool = False, debug: bool = False):
    """ Returns a list of function values for sampled parametrisations

    Args:
        functions: (list of functions) to be sampled
        sample_size (int): sample size in each parameter
        parameters (list of strings): parameter names (used for faster eval)
        intervals (list of pairs of numbers): intervals of parameters
        silent (bool): if silent command line output is set to minimum
        debug (bool): if debug extensive output is provided

    Returns:
        (array): [function index, [parameter values], function value]

    """
    arr = []
    if debug:
        print("Inside of sample_n_visualise.sample_list_funs()")
        print("List_fun: ", functions)
        print("Intervals: ", intervals)

    for index, polynomial in enumerate(functions):
        if is_this_z3_function(polynomial):
            functions[index] = translate_z3_function(polynomial)
        if debug:
            print("Polynomial: ", polynomial)

        if parameters:
            fun_parameters = parameters
        else:
            fun_parameters = set()
            fun_parameters.update(find_param(polynomial, debug))

            ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
            # if len(parameters) < N:
            #     parameters.update(find_param(polynomial, debug))
            if debug:
                print("Parameters: ", fun_parameters)
            fun_parameters = sorted(list(fun_parameters))

        if debug:
            print("Sorted parameters: ", fun_parameters)

        parameter_values = get_param_values(fun_parameters, sample_size, intervals=intervals, debug=debug)

        for parameter_value in parameter_values:
            if debug:
                print("Parameter_value: ", parameter_value)
            a = [functions.index(polynomial)]
            for param_index in range(len(fun_parameters)):
                a.append(parameter_value[param_index])
                if debug:
                    print("Parameter[param]: ", fun_parameters[param_index])
                    print("Parameter_value[param]: ", parameter_value[param_index])
                locals()[fun_parameters[param_index]] = float(parameter_value[param_index])

            # print("polynomial", polynomial)

            value = eval(polynomial)
            if debug:
                print("Eval ", polynomial, value)
            a.append(value)
            arr.append(a)
    return arr


def visualise(dic_fun, indices, sample_size, cumulative=False, debug: bool = False, show_all_in_one=False, where=False):
    """ Creates bar plot of probabilities of i successes for sampled parametrisation

    Args:
        dic_fun (dictionary index -> list of rational functions)
        sample_size (int): sample size in each parameter
        indices (list of ints): list of indices to show
        cumulative (bool): if True cdf instead of pdf is visualised
        debug (bool): if debug extensive output is provided
        show_all_in_one (bool): if True all plots are put into one window
        where (tuple/list): output matplotlib sources to output created figure
    """

    for index in indices:
        parameters = set()
        for index, expression in enumerate(dic_fun[index]):
            if is_this_z3_function(expression):
                dic_fun[index][index] = translate_z3_function(expression)

            if debug:
                print("Polynomial: ", expression)
            parameters.update(find_param(expression, debug))

            ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
            # if len(parameters) < index:
            #    parameters.update(find_param(expression, debug))
        if debug:
            print("Parameters: ", parameters)
        parameters = sorted(list(parameters))
        if debug:
            print("Sorted parameters: ", parameters)

        parameter_values = get_param_values(parameters, sample_size, debug)

        for parameter_value in parameter_values:
            if debug:
                print("Parameter_value: ", parameter_value)
            add = 0
            a = [index, dic_fun[index].index(expression)]
            if index == 0:
                title = f"Rational functions sampling \n parameters:"
            else:
                title = f"Rational functions sampling \n index={index}, parameters:"
            for param in range(len(parameters)):
                a.append(parameter_value[param])
                if debug:
                    print("Parameter[param]: ", parameters[param])
                    print("Parameter_value[param]: ", parameter_value[param])
                globals()[parameters[param]] = parameter_value[param]
                title = "{} {}={},".format(title, parameters[param], parameter_value[param])
            title = title[:-1]
            if debug:
                print("Eval ", expression, eval(expression))
            for expression in dic_fun[index]:
                value = eval(expression)
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
            rects1 = ax.bar(range(len(dic_fun[index])), a[len(parameters) + 2:], width, color='b')
            plt.show()


## SOURCE: https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
def visualise_by_param(hyper_rectangles, colour='g', title="", where=False):
    """
    Visualises domain intervals of each dimension in a plot.

    Args:
        hyper_rectangles (list of (hyper)rectangles)
        colour (string): colour of the lines in the figure
        title (string): title used for the figure
        where (tuple/list): output matplotlib sources to output created figure

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

        lc = mc.LineCollection(lines, color=colour, linewidths=2)

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


def heatmap(function, region, sampling_sizes, posttitle="", where=False, parameters=False, verbose=False):
    """ Creates 2D heatmap plot of sampled points of given function

    Args:
        function (string): function to be analysed
        region (list of intervals): boundaries of parameter space to be sampled
        sampling_sizes (list of ints): tuple of sample size of respective parameter
        posttitle (string): A string to be put after the title
        where (tuple/list): output matplotlib sources to output created figure
        parameters (list): list of parameters
        verbose (bool): will input maximum information

    Example:
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

    vmin = min(arr[:, 2])
    vmax = max(arr[:, 2])

    if where:
        f, ax = plt.subplots()
        ax = sns.heatmap(heatmap_data, vmin=vmin, vmax=vmax, annot=verbose)
        title = f"Heatmap \n{posttitle}"
        ax.set_title(wraper.fill(title))
        ax.invert_yaxis()
        return f
    else:
        ax = sns.heatmap(heatmap_data, vmin=vmin, vmax=vmax, annot=verbose)
        title = f"Heatmap of the parameter space \n function: {function}"
        ax.set_title(wraper.fill(title))
        ax.invert_yaxis()
        plt.show()


def visualise_sampled_by_param(hyper_rectangles, sample_size):
    """
    Visualises sampled hyperspace by connecting the values in each dimension.

    Args:
        hyper_rectangles (list of hyperrectangles):
        sample_size (int): number of points to be sampled
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
