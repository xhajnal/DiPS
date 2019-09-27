import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from matplotlib import collections as mc
from matplotlib.ticker import MaxNLocator

workspace = os.path.dirname(__file__)
sys.path.append(os.path.join(workspace, '../src/'))
# sys.path.append(os.path.dirname(__file__))
from load import find_param
from miscellaneous import DocumentWrapper


wraper = DocumentWrapper(width=60)


def cartesian_product(*arrays):
    """ Returns a product of given list of arrays
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def get_param_values(parameters, size_q, intervals=False, debug=False):
    parameter_values = []
    for param in range(len(parameters)):
        if intervals:
            if debug:
                print(f"parameter index:{param} with intervals: [{intervals[param][0]},{intervals[param][1]}]")
            parameter_values.append(np.linspace(intervals[param][0], intervals[param][1], size_q, endpoint=True))
        else:
            parameter_values.append(np.linspace(0, 1, size_q, endpoint=True))
    parameter_values = cartesian_product(*parameter_values)
    if (len(parameters) - 1) == 0:
        parameter_values = np.linspace(0, 1, size_q, endpoint=True)[np.newaxis, :].T
    if debug:
        print("parameter_values", parameter_values)
    return parameter_values


def eval_and_show(fun_list, parameter_value, cumulative=False, debug=False, give_back=False, where=False):
    """ Creates bar plot of probabilities of i successes for given parametrisation

    Args
    ----------
    fun_list: (list of strings) list of rational functions
    parameter_value: (list of floats) array of param values
    cumulative: (Bool) if True cdf instead of pdf is visualised
    debug: (Bool) if debug extensive output is provided
    give_back: (Bool) if True the plot is not shown but returned
    where: (Tuple/List) : output matplotlib sources to output created figure
    """
    parameters = set()
    for polynome in fun_list:
        parameters.update(find_param(polynome, debug))
    parameters = sorted(list(parameters))
    if debug:
        print("parameters", parameters)

    title = "Rational functions sampling \n parameter values:"
    a = []
    add = 0
    for param in range(len(parameters)):
        if debug:
            print("parameters[param]", parameters[param])
            print("parameter_value[param]", parameter_value[param])
        globals()[parameters[param]] = parameter_value[param]
        title = "{} {}={},".format(title, parameters[param], parameter_value[param])
    title = title[:-1]

    title = f"{title}\n values: "
    for polynome in fun_list:
        if debug:
            print("eval ", polynome, eval(polynome))
        if cumulative:
            ## Add sum of all values
            value = eval(polynome)
            add = add + value
            a.append(add)
            title = f"{title} {add} , "
            del value
        else:
            a.append(eval(polynome))
            title = f"{title} {eval(polynome)} ,"
    title = title[:-2]
    if where:
        fig = where[0]
        ax = where[1]
        plt.autoscale()
        ax.autoscale()
    else:
        fig, ax = plt.subplots()
    width = 0.2
    ax.set_ylabel('Value')
    ax.set_xlabel('Rational function indices')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if debug:
        print(title)
    ax.set_title(wraper.fill(title))
    if debug:
        print("len(fun_list)", len(fun_list))
    ax.bar(range(1, len(fun_list)+1), a, width, color='b')
    if give_back:
        return fig, ax
    plt.show()
    return a


## OLD VERSION NOW USED FOR BACK COMPATIBILITY
def sample(dic_fun, agents_quantities, size_q, debug=False):
    """ Returns probabilities of i successes for sampled parametrisation

    Args
    ----------
    dic_fun: (dictionary N -> list of polynomials)
    size_q: (int) sample size in each parameter
    agents_quantities: (int) pop sizes to be used
    debug: (Bool) if debug extensive output is provided

    Returns
    ----------
    Array of [N, i, parameters, value]

    """
    arr = []
    if debug:
        print("Inside of sample_n_visualise.sample()")
        print("dic_fun", dic_fun)
    for N in agents_quantities:
        for polynome in dic_fun[N]:
            if debug:
                print("polynome", polynome)
            parameters = set()
            parameters.update(find_param(polynome, debug))

            ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
            # if len(parameters) < N:
            #     parameters.update(find_param(polynome, debug))
            if debug:
                print("parameters", parameters)
            parameters = sorted(list(parameters))
            if debug:
                print("parameters", parameters)

            parameter_values = get_param_values(parameters, size_q, debug)

            for parameter_value in parameter_values:
                if debug:
                    print("parameter_value", parameter_value)
                a = [N, dic_fun[N].index(polynome)]
                for param in range(len(parameters)):
                    a.append(parameter_value[param])
                    if debug:
                        print("parameters[param]", parameters[param])
                        print("parameter_value[param]", parameter_value[param])
                    globals()[parameters[param]] = parameter_value[param]
                if debug:
                    print("eval ", polynome, eval(polynome))
                a.append(eval(polynome))
                arr.append(a)
    return arr


def sample_fun(list_fun, size_q, intervals=False, debug=False, silent=False):
    """ Returns probabilities of i successes for sampled parametrisation

    Args
    ----------
    list_fun: (list of polynomials)
    size_q: (int) sample size in each parameter
    intervals: (list of pairs of numbers) intervals of parameters
    debug: (Bool) if debug extensive output is provided
    silent: (Bool) if silent command line output is set to minimum

    Returns
    ----------
    Array of [N, i, parameters, value]

    """
    arr = []
    if debug:
        print("Inside of sample_n_visualise.sample()")
        print("list_fun", list_fun)
        print("intervals", intervals)
    for polynome in list_fun:
        if debug:
            print("polynome", polynome)
        parameters = set()
        parameters.update(find_param(polynome, debug))

        ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
        # if len(parameters) < N:
        #     parameters.update(find_param(polynome, debug))
        if debug:
            print("parameters", parameters)
        parameters = sorted(list(parameters))
        if debug:
            print("parameters", parameters)

        parameter_values = get_param_values(parameters, size_q, intervals=intervals, debug=debug)

        for parameter_value in parameter_values:
            if debug:
                print("parameter_value", parameter_value)
            a = [list_fun.index(polynome)]
            for param in range(len(parameters)):
                a.append(parameter_value[param])
                if debug:
                    print("parameters[param]", parameters[param])
                    print("parameter_value[param]", parameter_value[param])
                globals()[parameters[param]] = parameter_value[param]
            if debug:
                print("eval ", polynome, eval(polynome))
            a.append(eval(polynome))
            arr.append(a)
    return arr


def visualise(dic_fun, agents_quantities, size_q, cumulative=False, debug=False, show_all_in_one=False, where=False):
    """ Creates bar plot of probabilities of i successes for sampled parametrisation

    Args
    ----------
    dic_fun: (dictionary N -> list of polynomials)
    size_q: (int) sample size in each parameter
    agents_quantities: (int) pop sizes to be used
    cumulative: (Bool) if True cdf instead of pdf is visualised
    debug: (Bool) if debug extensive output is provided
    show_all_in_one: (Bool) if True all plots are put into one window
    where: (Tuple/List) : output matplotlib sources to output created figure
    """

    for N in agents_quantities:
        parameters = set()
        for polynome in dic_fun[N]:
            if debug:
                print("polynome", polynome)
            parameters.update(find_param(polynome, debug))

            ## THIS THING IS WORKING ONLY FOR THE CASE STUDY
            # if len(parameters) < N:
            #    parameters.update(find_param(polynome, debug))
        if debug:
            print("parameters", parameters)
        parameters = sorted(list(parameters))
        if debug:
            print("parameters", parameters)

        parameter_values = get_param_values(parameters, size_q, debug)

        for parameter_value in parameter_values:
            if debug:
                print("parameter_value", parameter_value)
            add = 0
            a = [N, dic_fun[N].index(polynome)]
            if N == 0:
                title = f"Rational functions sampling \n parameters:"
            else:
                title = f"Rational functions sampling \n N={N}, parameters:"
            for param in range(len(parameters)):
                a.append(parameter_value[param])
                if debug:
                    print("parameters[param]", parameters[param])
                    print("parameter_value[param]", parameter_value[param])
                globals()[parameters[param]] = parameter_value[param]
                title = "{} {}={},".format(title, parameters[param], parameter_value[param])
            title = title[:-1]
            if debug:
                print("eval ", polynome, eval(polynome))
            for polynome in dic_fun[N]:
                if cumulative:
                    ## Add sum of all values
                    value = eval(polynome)
                    add = add + value
                    a.append(add)
                    del value
                else:
                    a.append(eval(polynome))

            print(a)
            fig, ax = plt.subplots()
            width = 0.2
            ax.set_ylabel('Value')
            ax.set_xlabel('Rational function indices')
            ax.set_title(wraper.fill(title))
            print(title)
            rects1 = ax.bar(range(len(dic_fun[N])), a[len(parameters) + 2:], width, color='b')
            plt.show()


def visualise_byparam(hyper_rectangles):
    """
    Visualises domain intervals of each dimension in a plot.

    Args
    ----------
    hyper_rectangles: list of (hyper)rectangles
    """
    from sympy import Interval

    ## https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
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

        fig, ax = pl.subplots()

        ax.set_xlabel('Parameter indices')
        ax.set_ylabel('Parameter values')
        ax.set_title("Domain in which respective parameter belongs to in the given space")

        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        print(intervals)
    else:
        print("Given space is empty, no intervals to be visualised")


def heatmap(fun, region, sampling_sizes, posttitle="", where=False, parameters=False):
    """ Creates 2D heatmap plot of sampled points of given function

    Args
    ----------
    fun: dictionary N -> list of polynomials
    region: (list of intervals) boundaries of parameter space to be sampled
    sampling_sizes: (int) tuple of sample size of respective parameter
    posttitle: (string) A string to be put after the title
    where: (Tuple/List) : output matplotlib sources to output created figure
    parameters (list): list of parameters

    Example
    ----------
    heatmap("p+q",[[0,1],[3,4]],[5,5])
    """
    if not parameters:
        parameters = sorted(list(find_param(fun)))
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
            arr[jj, 2] = eval(fun)
    # print(arr)
    # d = pd.DataFrame(arr, columns=["p","q","E"])
    d = pd.DataFrame(arr, columns=[parameters[0], parameters[1], "E"])
    # d = d.pivot("p", "q", "E")
    d = d.pivot(parameters[0], parameters[1], "E")

    if where:
        f, ax = plt.subplots()
        ax = sns.heatmap(d)
        title = f"Heatmap \n{posttitle}"
        ax.set_title(wraper.fill(title))
        return f
    else:
        ax = sns.heatmap(d)
        title = "Heatmap of the parameter space"
        ax.set_title(wraper.fill(title))
        plt.show()


def visualise_sampled_byparam(hyper_rectangles, sample_size):
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
        ax.autoscale()
        ax.margins(0.1)
        plt.show()
    else:
        print("Given space is empty")
