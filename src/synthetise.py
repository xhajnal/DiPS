import datetime
import os
import re
import socket
import sys
import pickle
from time import strftime, localtime, time
import platform
from collections.abc import Iterable

from termcolor import colored
from math import log
import itertools

import numpy as np
from sympy import Interval
from mpmath import mpi
from matplotlib.patches import Rectangle
import unittest


from dreal import logical_and, logical_or, logical_not, Variable, CheckSatisfiability, Box
# from dreal import *

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param
from space import RefinedSpace
from space import get_rectangle_volume
from sample_n_visualise import cartesian_product

import configparser

config = configparser.ConfigParser()

# print(os.getcwd())
workspace = os.path.dirname(__file__)
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)

config.read("../config.ini")
# config.sections()

refine_timeout = int(config.get("paths", "refine_timeout"))

refinement_results = config.get("paths", "refinement_results")
if not os.path.exists(refinement_results):
    os.makedirs(refinement_results)

# import struct
# print("You are running "+ str(struct.calcsize("P") * 8)+"bit Python, please verify that installed z3 is compatible")
# print("path: ", os.environ["PATH"])

# print(os.environ["PATH"])

try:
    from z3 import *
    os.chdir(cwd)
except:
    z3_path = config.get("paths", "z3_path")

    if not os.path.exists(z3_path):
        raise OSError("Directory does not exist: " + str(z3_path))
    print("z3_path", z3_path)
    os.chdir(cwd)

    ##  Add z3 to PATH
    if '/' in z3_path:
        z3_path_short = '/'.join(z3_path.split("/")[:-1])
    elif '\\' in z3_path:
        z3_path_short = '\\'.join(z3_path.split("\\")[:-1])
    else:
        print("Warning: Could not set path to add to the PATH, please add it manually")

    if "PATH" not in os.environ:
        os.environ["PATH"] = z3_path
    else:
        if z3_path_short not in os.environ["PATH"]:
            if z3_path_short.replace("/", "\\") not in os.environ["PATH"]:
                if "wind" in platform.system().lower():
                    os.environ["PATH"] = os.environ["PATH"] + ";" + z3_path_short
                else:
                    os.environ["PATH"] = os.environ["PATH"] + ":" + z3_path_short
    sys.path.append(z3_path)

    ## Add z3 to PYTHON PATH
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = z3_path
    else:
        if z3_path not in os.environ["PYTHONPATH"]:

            if "wind" in platform.system().lower():
                os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ";" + z3_path
            else:
                os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + z3_path

    ## Add z3 to LDLIB PATH
    if "wind" not in platform.system().lower():
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":" + z3_path_short
        else:
            os.environ["LD_LIBRARY_PATH"] = z3_path_short

    ##  Add z3 to other variables
    os.environ["Z3_LIBRARY_PATH"] = z3_path_short
    os.environ["Z3_LIBRARY_DIRS"] = z3_path_short

    ## Try to import z3 from a given folder
    try:
        from z3 import *
        # print(os.getcwd())
        # import subprocess
        # subprocess.call(["python", "example.py"])
    except:
        raise Exception("could not load z3 from: ", z3_path)

## Try to run z3
try:
    p = Real('p')
except NameError:
    raise Exception("z3 not loaded properly")


class Queue:
    ## Constructor creates a list
    def __init__(self):
        self.queue = list()

    ## Adding elements to queue
    def enqueue(self, data):
        ## Checking to avoid duplicate entry (not mandatory)
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False

    ## Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return "Queue Empty!"

    ## Getting the size of the queue
    def size(self):
        return len(self.queue)

    ## Printing the elements of the queue
    def print_queue(self):
        return self.queue


def ineq_to_props(funcs, intervals, silent=True):
    """ Converts inequalities of the function given by the interval to properties

    Example: ["x+3"],[[0,1]] ->  ["x+3>=0","x+3<=1"]

    Args
    ----------
    funcs:  (list of strings) array of functions
    intervals: (list of intervals) array of pairs, low and high bound
    silent: (Bool): if silent printed output is set to minimum
    """

    if len(funcs) is not len(intervals):
        if not silent:
            print(colored(f"len of functions {len(funcs)} and intervals {len(intervals)} does not correspond", "red"))
        return False

    spam = []
    for index in range(len(funcs)):
        if isinstance(intervals[index], Interval):
            spam.append(funcs[index] + ">=" + str(intervals[index].start))
            spam.append(funcs[index] + "<=" + str(intervals[index].end))
        else:
            spam.append(funcs[index] + ">=" + str(intervals[index][0]))
            spam.append(funcs[index] + "<=" + str(intervals[index][1]))
    return spam


def props_to_ineq(props, silent=True, debug=False):
    """ Converts properties to functions and inequalities if possible

    Example: ["x+3>=0","x+3<=1"] -> ["x+3"],[[0,1]]

    Args
    ----------
    props:  (list of strings) properties to be converted
    silent: (Bool): if silent printed output is set to minimum
    debug: (Bool) if True extensive print will be used
    """
    if len(props) % 2:
        if not silent:
            print(colored("Number of properties is not even, some interval will be invalid", "red"))
        return False
    funcs = []
    intervals = []
    is_odd = False
    index = 0

    for prop in props:
        spam = "None"
        if debug:
            print(f"property {index + 1} before splitting", prop)
        try:
            prop = prop.replace("<=", "<").replace(">=", "<").replace("=>", "<").replace("=<", "<").replace(">", "<")
            spam = prop.split("<")
        except AttributeError:
            print()
        if debug:
            print(f"property {index+1} after splitting", spam)
        if len(spam) <= 1:
            if not silent:
                print(colored(f"Property {index+1} is not in a form of inequality", "red"))
            return False
        elif len(spam) > 2:
            if not silent:
                print(colored(f"Property {index+1} has more than one inequality sign", "red"))
            return False
        else:
            try:
                ## The righthandside is number
                float(spam[1])
                if debug:
                    print("righthandside ", float(spam[1]))
            except ValueError:
                spam = [f"{spam[0]} -( {spam[1]})", 0]

            ## If we are at odd position check
            if is_odd:
                if debug:
                    print("is odd")
                    print("funcs[-1]", funcs[-1])
                    print("spam[0]", spam[0])
                ## whether the previous function is the same as the new one
                if funcs[-1] == spam[0]:
                    # if yes, add the other value of the interval
                    if debug:
                        print("Adding value")
                        print("len(funcs)", len(funcs))
                        print("[intervals[len(funcs)-1], spam[1]]", [intervals[len(funcs)-1], spam[1]])
                    intervals[len(funcs)-1] = [intervals[len(funcs)-1], spam[1]]
            else:
                funcs.append(spam[0])
                intervals.append(spam[1])
        is_odd = not is_odd
        index = index + 1

    ## Sort the intervals
    index = 0
    for interval_index in range(len(intervals)):
        if len(intervals[interval_index]) is not 2:
            if not silent:
                print(colored(f"Property {index + 1} does not have proper number of boundaries", "red"))
            return False
        if debug:
            print("sorted([float(intervals[interval_index][0]), float(intervals[interval_index][1])])", sorted([float(intervals[interval_index][0]), float(intervals[interval_index][1])]))
        intervals[interval_index] = sorted([float(intervals[interval_index][0]), float(intervals[interval_index][1])])
        if debug:
            print("Interval(intervals[interval_index][0], intervals[interval_index][1]) ", Interval(intervals[interval_index][0], intervals[interval_index][1]))
        intervals[interval_index] = Interval(intervals[interval_index][0], intervals[interval_index][1])
        index = index + 1

    if debug:
        print("funcs: ", funcs)
        print("intervals: ", intervals)

    return funcs, intervals


def to_interval(points):
    """ Transforms the set of points into set of intervals

    Args
    ----------
    points: (list of pairs) which are the points
    """
    intervals = []
    for dimension in range(len(points[0])):
        interval = [points[0][dimension], points[0][dimension]]
        for point in range(len(points)):
            if interval[0] > points[point][dimension]:
                interval[0] = points[point][dimension]
            if interval[1] < points[point][dimension]:
                interval[1] = points[point][dimension]
        intervals.append(interval)
    return intervals


def is_in(region1, region2):
    """Returns yes if the interval1 is in the other interval, returns False otherwise

    Args
    ----------
    region1: (list of pairs) (hyper)space defined by the regions
    region2: (list of pairs) (hyper)space defined by the regions
    """
    if len(region1) is not len(region2):
        print("The intervals does not have the same size")
        return False

    for dimension in range(len(region1)):
        if mpi(region1[dimension]) not in mpi(region2[dimension]):
            return False
    return True


def refine_by(region1, region2, debug=False):
    """Returns the first (hyper)space refined/spliced by the second (hyperspace) into orthogonal subspaces

    Args
    ----------
    region1: (list of pairs) (hyper)space defined by the regions
    region2: (list of pairs) (hyper)space defined by the regions
    debug: (Bool): if True extensive print will be used
    """

    if not is_in(region2, region1):
        raise Exception("the first interval is not within the second, it cannot be refined/spliced properly")

    region1 = copy.deepcopy(region1)
    regions = []
    ## for each dimension trying to cut of the space
    for dimension in range(len(region2)):
        ## LEFT
        if region1[dimension][0] < region2[dimension][0]:
            sliced_region = copy.deepcopy(region1)
            sliced_region[dimension][1] = region2[dimension][0]
            if debug:
                print("left ", sliced_region)
            regions.append(sliced_region)
            region1[dimension][0] = region2[dimension][0]
            if debug:
                print("new intervals", region1)

        ## RIGHT
        if region1[dimension][1] > region2[dimension][1]:
            sliced_region = copy.deepcopy(region1)
            sliced_region[dimension][0] = region2[dimension][1]
            if debug:
                print("right ", sliced_region)
            regions.append(sliced_region)
            region1[dimension][1] = region2[dimension][1]
            if debug:
                print("new intervals", region1)

    # print("region1 ", region1)
    regions.append(region1)
    return regions


def check_unsafe(region, props, silent=False, called=False):
    """ Check if the given region is unsafe or not using z3.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of functions (rational functions in the case of Markov Chains)
    silent: (Bool): if silent printed output is set to minimum
    called: (Bool): if called updates the global variables (use when calling it directly)
    """
    ## Initialisation
    if not silent:
        print("checking unsafe", region, "current time is", datetime.datetime.now())

    # p = Real('p')
    # print(p)
    # print(type(p))

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        globals()["parameters"] = sorted(list( globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        for param in parameters:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]

    s = Solver()

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        # print("globals()[param]", globals()[param])
        # print("region[j][0]", region[j][0])
        s.add(globals()[param] >= region[j][0])
        s.add(globals()[param] <= region[j][1])
        j = j + 1

    ## Adding properties to solver
    for i in range(0, len(props)):
        # print(props[i])
        s.add(eval(props[i]))

    if s.check() == sat:
        return s.model()
    else:
        space.add_red(region)
        return True


def check_safe(region, props, silent=False, called=False):
    """ Check if the given region is safe or not using z3.

    It means whether for all parametrisations in **region** every property(prop) is evaluated within the given
    **interval**, otherwise it is not safe and counterexample is returned.

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    silent: (Bool): if silent printed output is set to minimum
    called: (Bool): if called updates the global variables (use when calling it directly)
    """
    # initialisation
    if not silent:
        print("checking safe", region, "current time is", datetime.datetime.now())

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']

        for param in globals()["parameters"]:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]

    s = Solver()

    # if not silent:
    #    print("with parameters", globals()["parameters"])

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        # print("globals()[param]", globals()[param])
        # print("region[j][0]", region[j][0])
        s.add(globals()[param] >= region[j][0])
        s.add(globals()[param] <= region[j][1])
        j = j + 1

    ## Adding properties to solver
    formula = Not(eval(props[0]))
    for i in range(1, len(props)):
        formula = Or(formula, Not(eval(props[i])))
    s.add(formula)

    # print(s.check_unsafe())
    # return s.check_unsafe()
    if s.check() == unsat:
        space.add_green(region)
        return True
    else:
        return s.model()


def check_unsafe_dreal(region, props, delta=0.001, silent=False, called=False):
    """ Check if the given region is unsafe or not using dreal.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of functions (rational functions in the case of Markov Chains)
    delta:  (Number): allowed error
    silent: (Bool): if silent printed output is set to minimum
    called: (Bool): if called updates the global variables (use when calling it directly)
    """
    ## Initialisation
    if not silent:
        print("checking unsafe", region, "current time is", datetime.datetime.now())

    # p = Real('p')
    # print(p)
    # print(type(p))

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        globals()["parameters"] = sorted(list( globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        for param in parameters:
            globals()[param] = Variable(param)
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]

    # if not silent:
    #    print("with parameters", globals()["parameters"])

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        # print("globals()[param]", globals()[param])
        # print("region[j][0]", region[j][0])
        if j == 0:
            f_sat = logical_and(globals()[param] >= region[j][0], globals()[param] <= region[j][1])
        else:
            f_sat = logical_and(f_sat, globals()[param] >= region[j][0])
            f_sat = logical_and(f_sat, globals()[param] <= region[j][1])
        j = j + 1

    ## Adding properties to solver
    for i in range(0, len(props)):
        # print(props[i])
        f_sat = logical_and(f_sat, eval(props[i]))

    result = CheckSatisfiability(f_sat, 0.001)

    if result is not None:
        return result
    else:
        space.add_red(region)
        return True


def check_safe_dreal(region, props, delta=0.001, silent=False, called=False):
    """ Check if the given region is safe or not using dreal

    It means whether for all parametrisations in **region** every property(prop) is evaluated within the given
    **interval**, otherwise it is not safe and counterexample is returned.

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    delta:  (Number): allowed error
    silent: (Bool): if silent printed output is set to minimum
    called: (Bool): if called updates the global variables (use when calling it directly)
    """
    # initialisation
    if not silent:
        print("checking safe", region, "current time is", datetime.datetime.now())

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']

        for param in globals()["parameters"]:
            globals()[param] = Variable(param)
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]

    # if not silent:
    #    print("with parameters", globals()["parameters"])

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        # print("globals()[param]", globals()[param])
        # print("region[j][0]", region[j][0])
        if j == 0:
            f_sat = logical_and(globals()[param] >= region[j][0], globals()[param] <= region[j][1])
        else:
            f_sat = logical_and(f_sat, globals()[param] >= region[j][0])
            f_sat = logical_and(f_sat, globals()[param] <= region[j][1])
        j = j + 1

    ## Adding properties to solver
    formula = logical_not(eval(props[0]))
    for i in range(1, len(props)):
        formula = logical_or(formula, logical_not(eval(props[i])))
    f_sat = logical_and(f_sat, formula)

    result = CheckSatisfiability(f_sat, 0.001)

    if result is None:
        space.add_green(region)
        return True
    else:
        return result


def check_deeper(region, props, n, epsilon, coverage, silent, version, size_q=False, debug=False, save=False, title="", where=False):
    """ Refining the parameter space into safe and unsafe regions with respective alg/method

    Args
    ----------
    region: (list of intervals/space) array of pairs, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    version: (Int): version of the algorithm to be used
    size_q: (Int): number of samples in dimension used for presampling
    debug: (Bool): if True extensive print will be used
    save: (Bool): if True output is stored
    title: (string): text to be added in Figure titles
    where: (Tuple/List) : output matplotlib sources to output created figure
    """

    ## INITIALISATION
    ## Save file
    if save is True:
        # save = f"{},{n},{epsilon},{coverage},{version}"
        save = strftime("%d-%b-%Y-%H-%M-%S", localtime())
        # save = os.path.join(refinement_results, str(strftime("%d-%b-%Y-%H:%M:%S", localtime())))
        if debug:
            print(save)

    ## Parameters
    globals()["parameters"] = set()
    for polynome in props:
        globals()["parameters"].update(find_param(polynome))
    globals()["parameters"] = sorted(list(globals()["parameters"]))
    parameters = globals()["parameters"]
    if debug:
        print("parameters", parameters)

    ## If the given region is space
    ## TBD correct this
    if not isinstance(region, list):
        space = region
        globals()["space"] = space
        del region
        # print(type(space))
        region = space.region
        # print(region)

        ## Not required since space with no parameters cannot be created
        # if not space.params:
        #     space.params = parameters

        ## Check whether the the set of params is equal
        print("space.params", space.params)
        print("parameters", parameters)
        if not sorted(space.params) == sorted(parameters):
            raise Exception("The set of parameters of the given space and properties does not correspond")

    ## If the region is just list of intervals - a space is to be created
    else:
        ### Regions
        ### Taking care of unchangeable tuples
        for interval_index in range(len(region)):
            region[interval_index] = [region[interval_index][0], region[interval_index][1]]

        ### Params
        if not isinstance(props, Iterable):
            raise Exception("Given properties are not iterable, to use single property use list of length 1")

        globals()["space"] = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[], title=title)
        space = globals()["space"]

        globals()["default_region"] = copy.deepcopy(region)

    if not silent:
        print("the area is: ", space.region)
        print("the volume of the whole area is:", space.get_volume())
        print()

    start_time = time()

    if debug:
        print("region", region)
        print("props", props)

    ## PRESAMPLING HERE
    if size_q:
        if version == 1:
            print("Using presampled DFS method")
        elif version == 2:
            print("Using presampled BFS method")
            globals()["que"] = Queue()
        elif version == 3:
            print("Using presampled BFS method with passing examples")
            globals()["que"] = Queue()
        elif version == 4:
            print("Using presampled BFS method with passing examples and counterexamples")
            globals()["que"] = Queue()
        elif version == 5:
            print("Using presampled interval arithmetic")
            globals()["que"] = Queue()
        else:
            print(colored("Chosen version not found", "red"))
            return

        # globals()["space"] = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])

        # funcs, intervals = props_to_ineq(props)

        to_be_searched = sample(space, props, size_q, compress=True, silent=not debug, save=save)

        if debug:
            print(type(to_be_searched))
            print("sampled space: ", to_be_searched)

        ## PARSE SAT POINTS
        sat_points = []

        print(to_be_searched)
        while not isinstance(to_be_searched[0][1], type(True)):
            to_be_searched = list(itertools.chain.from_iterable(to_be_searched))

        if debug:
            print(type(to_be_searched))
            print("unfolded sampled space: ", to_be_searched)
            print("an element from sampled space:", to_be_searched[0])

        for point in to_be_searched:
            if point[1] is True:
                sat_points.append(point[0])
        if debug:
            print("satisfying points: ", sat_points)

        if debug and save:
            print("I am showing sampling_sat_"+str(save))
        space.show(red=False, green=False, sat_samples=True, unsat_samples=False, save=save, where=where)

        ## COMPUTING THE ORTHOGONAL HULL OF SAT POINTS
        ## Initializing the min point and max point as the first point
        if sat_points:
            sat_min = copy.deepcopy(sat_points[0])
            if debug:
                print("initial min", sat_min)
            sat_max = copy.deepcopy(sat_points[0])
            if debug:
                print("initial max", sat_max)

            ## TBD - POSSIBLE OPTIMISATION HERE DOING IT IN THE REVERSE ORDER AND STOPPING IF A BORDER OF THE REGION IS ADDED
            for point in sat_points:
                if debug:
                    print(point)
                for dimension in range(0, len(sat_points[0])):
                    if debug:
                        print(point[dimension])
                    if point[dimension] < sat_min[dimension]:
                        if debug:
                            print("current point:", point[dimension], "current min:", sat_min[dimension], "change min")
                        sat_min[dimension] = point[dimension]
                    if point[dimension] > sat_max[dimension]:
                        if debug:
                            print("current point:", point[dimension], "current max:", sat_max[dimension], "change max")
                        sat_max[dimension] = point[dimension]
            if debug:
                print(f"Points bordering the sat hull are: {sat_min}, {sat_max}")

            if is_in(region, to_interval([sat_min, sat_max])):
                print("The orthogonal hull of sat points actually covers the whole region")
            else:
                ## SPLIT THE WHITE REGION INTO 3-5 AREAS (in 2D) (DEPENDING ON THE POSITION OF THE HULL)

                ## THIS FIXING WORKS ONLY FOR THE UNIFORM SAMPLING
                spam = to_interval([sat_min, sat_max])
                for interval_index in range(len(spam)):
                    ## increase the space to the left
                    spam[interval_index][0] = max(region[interval_index][0], spam[interval_index][0] - (region[interval_index][1]-region[interval_index][0])/(size_q-1))
                    ## increase the space to the right
                    spam[interval_index][1] = min(region[interval_index][1], spam[interval_index][1] + (region[interval_index][1] - region[interval_index][0]) / (size_q - 1))
                print(f"Intervals bordering the sat hull are: {spam}")
                if debug:
                    print(colored("I was here", 'red'))
                space.remove_white(region)
                regions = refine_by(region, spam, debug)
                for subregion in regions:
                    space.add_white(subregion)
        else:
            print("No sat points in the samples")

        ## PARSE UNSAT POINTS
        unsat_points = []
        for point in to_be_searched:
            if point[1] is False:
                unsat_points.append(point[0])
        if debug:
            print("unsatisfying points: ", unsat_points)
        if debug and save:
            print("I am showing sampling_unsat_"+str(save))

        space.show(red=False, green=False, sat_samples=False, unsat_samples=True, save=save, where=where)

        ## If there is only the default region to be refined in the whitespace
        if len(space.get_white()) == 1:
            ## COMPUTING THE ORTHOGONAL HULL OF UNSAT POINTS
            ## Initializing the min point and max point as the first point

            if unsat_points:
                unsat_min = copy.deepcopy(unsat_points[0])
                if debug:
                    print("initial min", unsat_min)
                unsat_max = copy.deepcopy(unsat_points[0])
                if debug:
                    print("initial max", unsat_max)

                ## TBD - POSSIBLE OPTIMISATION HERE DOING IT IN THE REVERSE ORDER AND STOPPING IF A BORDER OF THE REGION IS ADDED
                for point in unsat_points:
                    if debug:
                        print(point)
                    for dimension in range(0, len(unsat_points[0])):
                        if debug:
                            print(point[dimension])
                        if point[dimension] < unsat_min[dimension]:
                            if debug:
                                print("current point:", point[dimension], "current min:", unsat_min[dimension], "change min")
                            unsat_min[dimension] = point[dimension]
                        if point[dimension] > unsat_max[dimension]:
                            if debug:
                                print("current point:", point[dimension], "current max:", unsat_max[dimension], "change max")
                            unsat_max[dimension] = point[dimension]
                if debug:
                    print(f"Intervals bordering the unsat hull are:: {unsat_min},{unsat_max}")

                if is_in(region, to_interval([unsat_min, unsat_max])):
                    print("The orthogonal hull of unsat points actually covers the whole region")
                else:
                    ## SPLIT THE WHITE REGION INTO 3-5 AREAS (in 2D) (DEPENDING ON THE POSITION OF THE HULL)
                    if debug:
                        print(colored("I was here", 'red'))
                        print("space white", space.get_white())
                    space.remove_white(region)
                    regions = refine_by(region, to_interval([unsat_min, unsat_max]))
                    for subregion in regions:
                        space.add_white(subregion)
            else:
                print("No unsat points in the samples")

        ## Make a copy of white space
        white_space = space.get_white()

        # print(globals()["parameters"])
        # print(space.params)
        ## Setting the param back to z3 definition
        if debug:
            print("region now", region)
            print("space white", white_space)

        print("Presampling resulted in splicing the region into these subregions: ", white_space)
        print(f"It took {socket.gethostname()} {round(time() - start_time)} second(s)")
        print()

    ## Choosing version/algorithm here
    ## If using z3 initialise the parameters
    if version <= 4:
        index = 0
        for param in space.params:
            if space.types[index] == "Real":
                globals()[param] = Real(param)
            elif space.types[index] == "Int":
                globals()[param] = Int(param)
            elif space.types[index] == "Bool":
                globals()[param] = Bool(param)
            elif space.types[index][0] == "BitVec":
                globals()[param] = BitVec(param, space.types[index][1])
            else:
                print(colored(f"Type of parameter {param} which was set as {space.types[index]} does not correspond with any known type", "red"))
                raise TypeError(f"Type of parameter {param} which was set as {space.types[index]} does not correspond with any known type")
            index = index + 1

    ## Iterating through the regions
    white_space = space.get_white()
    if len(white_space) is 1:
        if version == 1:
            print("Using DFS method")
            private_check_deeper(region, props, n, epsilon, coverage, silent)
        elif version == 2:
            print("Using BFS method")
            globals()["que"] = Queue()
            private_check_deeper_queue(region, props, n, epsilon, coverage, silent)
        elif version == 3:
            print("Using BFS method with passing examples")
            globals()["que"] = Queue()
            private_check_deeper_queue_checking(region, props, n, epsilon, coverage, silent, None)
        elif version == 4:
            print("Using BFS method with passing examples and counterexamples")
            globals()["que"] = Queue()
            private_check_deeper_queue_checking_both(region, props, n, epsilon, coverage, silent, None)
        elif version == 5:
            print("Using interval arithmetic")
            globals()["que"] = Queue()

            egg = props_to_ineq(props, debug=False)
            if not egg:
                return space
            if not silent:
                print("props", props)
                print("converted_intervals", egg)

            private_check_deeper_interval(region, egg[0], egg[1], n, epsilon, coverage, silent)
        else:
            print(colored("Chosen version not found", "red"))
    else:
        for rectangle in white_space:
            start_time = time()
            ## To get more similar result substituting the number of splits from the max_depth
            if debug:
                print("max_depth = ", max(1, n-(int(log(len(white_space), 2)))))
                print("refining", rectangle)

            ## THE PROBLEM IS THAT COVERAGE IS COMPUTED FOR THE WHOLE SPACE NOT ONLY FOR THE GIVEN REGION
            rectangle_size = get_rectangle_volume(rectangle)

            # print("rectangle", rectangle, " props", props, "silent", silent)
            # print("current coverage", space.get_coverage())
            # print("whole area", space.get_volume())
            # print("rectangle_size", rectangle_size)

            ## Setting the coverage as lower value between desired coverage and the proportional expected coverage
            next_coverage = min(coverage, (space.get_coverage() + (rectangle_size / space.get_volume())*coverage))
            # print("next coverage", next_coverage)

            if debug:
                print("region", rectangle)
                print("props", props)

            if version == 1:
                print(f"Using DFS method to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper(rectangle, props, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent)
            elif version == 2:
                print(f"Using BFS method to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_queue(rectangle, props, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent)
            elif version == 3:
                print(f"Using BFS method with passing examples to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_queue_checking(rectangle, props, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, None)
            elif version == 4:
                print(f"Using BFS method with passing examples and counterexamples to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_queue_checking_both(rectangle, props, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, None)
            elif version == 5:
                print(f"Using interval method to solve spliced rectangle number {white_space.index(rectangle)+1}")

                egg = props_to_ineq(props, debug=False)
                if not egg:
                    return space
                private_check_deeper_interval(rectangle, egg[0], egg[1], max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent)
            else:
                print(colored("Chosen version not found", "red"))
                return space

            ## Showing the step refinements of respective rectangles from the white space
            ## If the visualisation of the space did not succeed space_shown = (None, error message)
            space_shown = space.show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version} \n It took {socket.gethostname()} {round(time() - start_time)} second(s)", save=save, where=where)
            print()
            if space.get_coverage() >= coverage:
                break

            ## OLD REFINEMENT HERE
            # # to_be_searched = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], , compress=True, silent=False)

            # to_be_searched = refine_into_rectangles(to_be_searched, silent=False)

            # print("to_be_searched: ", to_be_searched)
            # globals()["que"] = Queue()

            # for rectangle in to_be_searched:
            #     print(rectangle)
            #     # print("safe", space.get_green())
            #     print("unsafe", space.get_red())
            #     space.add_white(rectangle)
            #     private_check_deeper_interval(rectangle, props, 0, epsilon, coverage, silent)

    ## VISUALISATION
    if not size_q:
        ## If the visualisation of the space did not succeed space_shown = (None, error message)
        space_shown = space.show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version} \n It took {socket.gethostname()} {round(time() - start_time)} second(s)", save=save, where=where)
    print("result coverage is: ", space.get_coverage())
    if where:
        if space_shown[0] is None:
            return space, space_shown[1]
        else:
            return space


def private_check_deeper(region, props, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    """

    ## TBD check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            # if not silent:
            #    print(f"hyperrectangle {region} too small, skipped")
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            # if not silent:
            #    print(f"rectangle {region} too small, skipped")
            return f"rectangle {region} too small, skipped"
        else:
            # if not silent:
            #    print(f"interval {region} too small, skipped")
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    if space.get_coverage() > coverage:
        return "coverage ", space.get_coverage(), " is above the threshold"

    # HERE MAY ADDING THE MODEL
    if check_unsafe(region, props, silent) is True:
        result = "unsafe"
    elif check_safe(region, props, silent) is True:
        result = "safe"
    else:
        result = "unknown"

    # print("result",result)
    # print("result",result)def
    if result == "safe" or result == "unsafe":
        space.remove_white(region)
    if n == 0:
        # print("[",p_low,",",p_high ,"],[",q_low,",",q_high ,"]",result)
        if not silent:
            print("maximal recursion reached here with coverage:",
                  space.get_coverage())
        return result
    else:
        if not (
                result == "safe" or result == "unsafe"):  ## Here is necessary to check only 3 of 4, since this line check 1 segment
            ## Find max interval
            index, maximum = 0, 0
            for i in range(len(region)):
                value = region[i][1] - region[i][0]
                if value > maximum:
                    index = i
                    maximum = value
            low = region[index][0]
            high = region[index][1]
            foo = copy.deepcopy(region)
            foo[index] = (low, low + (high - low) / 2)
            foo2 = copy.deepcopy(region)
            foo2[index] = (low + (high - low) / 2, high)
            space.remove_white(region)
            space.add_white(foo)  # add this region as white
            space.add_white(foo2)  # add this region as white
            # print("white area",globals()["hyper_rectangles_white"])
            if silent:
                private_check_deeper(foo, props, n - 1, epsilon, coverage, silent)
                if space.get_coverage() > coverage:
                    return f"coverage {space.get_coverage()} is above the threshold"
                private_check_deeper(foo2, props, n - 1, epsilon, coverage, silent)
            else:
                print(n, foo, space.get_coverage(),
                      private_check_deeper(foo, props, n - 1, epsilon, coverage, silent))
                if space.get_coverage() > coverage:
                    return f"coverage {space.get_coverage()} is above the threshold"
                print(n, foo2, space.get_coverage(),
                      private_check_deeper(foo2, props, n - 1, epsilon, coverage, silent))
    return result


def private_check_deeper_queue(region, props, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    """

    ## TBD check consitency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print(f"hyperrectangle {region} too small, skipped")
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print(f"rectangle {region} too small, skipped")
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print(f"interval {region} too small, skipped")
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    ## HERE I CAN APPEND THE VALUE OF EXAMPLE AND COUNTEREXAMPLE
    # print("hello check =",check_unsafe(region,prop,silent))
    # print("hello check safe =",check_safe(region,prop,n_samples,silent))
    if check_unsafe(region, props, silent) is True:
        result = "unsafe"
    elif check_safe(region, props, silent) is True:
        result = "safe"
    else:
        result = "unknown"

    if result == "safe" or result == "unsafe":
        # print("removing region:", region)
        space.remove_white(region)

    if not silent:
        if result is "safe":
            print(n, region, colored(f"{space.get_coverage()} {result}", "green"))
        elif result is "unsafe":
            print(n, region, colored(f"{space.get_coverage()} {result}", "red"))
        else:
            print(n, region, space.get_coverage(), result)

    # print("hello")
    if n == 0:
        return
    if result == "safe" or result == "unsafe":
        return

    ## Find maximum interval
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Add calls to the Queue
    # print("adding",[copy.deepcopy(foo),prop,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo),prop,n-1,epsilon,coverage,silent]))
    # print("adding",[copy.deepcopy(foo2),prop,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo2),prop,n-1,epsilon,coverage,silent]))
    globals()["que"].enqueue([copy.deepcopy(foo), props, n - 1, epsilon, coverage, silent])
    globals()["que"].enqueue([copy.deepcopy(foo2), props, n - 1, epsilon, coverage, silent])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue(*que.dequeue())


def private_check_deeper_queue_checking(region, props, n, epsilon, coverage, silent, model=None):
    """ THIS IS OBSOLETE METHOD, HERE JUST TO BE COMPARED WITH THE NEW ONE

    Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    model: (example,counterexample) of the satisfaction in the given region
    """

    ## TBD check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print(f"hyperrectangle {region} too small, skipped")
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print(f"rectangle {region} too small, skipped")
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print(f"interval {region} too small, skipped")
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    if model is None:
        example = check_unsafe(region, props, silent)
        # counterexample = check_safe(region,prop,silent)
    elif model[0] is None:
        example = check_unsafe(region, props, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        if not silent:
            print(n, region, colored(f"{space.get_coverage()} unsafe", "red"))
        return
    elif check_safe(region, props, silent) is True:
        space.remove_white(region)
        if not silent:
            print(n, region, colored(f"{space.get_coverage()} safe", "red"))
        return
    else:  ## unknown
        if not silent:
            print(n, region, space.get_coverage(), example)

    if n == 0:
        return

    example_points = re.findall(r'[0-9/]+', str(example))
    # counterexample_points= re.findall(r'[0-9/]+', str(counterexample))
    # print(example_points)
    # print(counterexample_points)

    ## Find maximum dimension an make a cut
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    model_low = [9, 9]
    model_high = [9, 9]
    if float(eval(example_points[index])) > low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = example
    else:
        model_low[0] = example
        model_high[0] = None
    ## Overwrite if equal
    if float(eval(example_points[index])) == low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = None

    ## Add calls to the Queue
    globals()["que"].enqueue(
        [copy.deepcopy(foo), props, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue(
        [copy.deepcopy(foo2), props, n - 1, epsilon, coverage, silent, model_high])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking(*que.dequeue())


def private_check_deeper_queue_checking_both(region, props, n, epsilon, coverage, silent, model=None):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    model: (example, counterexample) of the satisfaction in the given region
    """

    ## TBD check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print(f"hyperrectangle {region} too small, skipped")
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print(f"rectangle {region} too small, skipped")
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print(f"interval {region} too small, skipped")
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    ## Resolving if the region safe/unsafe/unknown
    if model is None:
        example = check_unsafe(region, props, silent)
        counterexample = check_safe(region, props, silent)
    elif model[0] is None:
        example = check_unsafe(region, props, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]
    if model is not None:
        if model[1] is None:
            counterexample = check_safe(region, props, silent)
        else:
            if not silent:
                print("skipping check_safe at", region, "since counterexample", model[1])
            counterexample = model[1]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        if not silent:
            print(n, region, colored(space.get_coverage(), "green"), "unsafe")
        return
    elif counterexample is True:
        space.remove_white(region)
        if not silent:
            print(n, region, colored(space.get_coverage(), "green"), "safe")
        return
    else:  # Unknown
        if not silent:
            print(n, region, colored(space.get_coverage(), "blue"), (example, counterexample))

    if n == 0:
        return

    # print("example", example)
    example_points = re.findall(r'[0-9/]+', str(example))
    counterexample_points = re.findall(r'[0-9/]+', str(counterexample))
    # print("example_points", example_points)
    # print(counterexample_points)

    ## Find maximum dimension an make a cut
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    model_low = [9, 9]
    model_high = [9, 9]
    if float(eval(example_points[index])) > low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = example
    else:
        model_low[0] = example
        model_high[0] = None
    if float(eval(counterexample_points[index])) > low + (high - low) / 2:
        model_low[1] = None
        model_high[1] = counterexample
    else:
        model_low[1] = counterexample
        model_high[1] = None
    ## Overwrite if equal
    if float(eval(example_points[index])) == low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = None
    if float(eval(counterexample_points[index])) == low + (high - low) / 2:
        model_low[1] = None
        model_high[1] = None

    ## Add calls to the Queue
    globals()["que"].enqueue([copy.deepcopy(foo), props, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue([copy.deepcopy(foo2), props, n - 1, epsilon, coverage, silent, model_high])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking_both(*que.dequeue())


def color_margins(greater, smaller):
    """ Colors outside of the smaller region in the greater region as previously unsat

    Args
    ----------
    greater: (list of intervals) region in which the smaller region is located
    smaller: (list of intervals) smaller region which is not to be colored
    """
    # rectangles_sat.append(Rectangle((low_x,low_y), width, height, fc='g'))
    # print("greater ",greater)
    # print("smaller ",smaller)
    if greater is None or smaller is None:
        return

    ## If 1 dimensional coloring
    if len(smaller) == 1:
        ## Color 2 regions, to the left, to the right
        ## To the left
        globals()["rectangles_unsat_added"].append(
            Rectangle([greater[0][0], 0], smaller[0][0] - greater[0][0], 1, fc='r'))
        ## To the right
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][1], 0], greater[0][1] - smaller[0][1], 1, fc='r'))

    ## Else 2 dimensional coloring
    elif len(smaller) == 2:
        ## Color 4 regions, to the left, to the right, below, and above
        ## TBD
        globals()["rectangles_unsat_added"].append(
            Rectangle([greater[0][0], 0], smaller[0][0] - greater[0][0], 1, fc='r'))
        ## TBD
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][1], 0], greater[0][1] - smaller[0][1], 1, fc='r'))
        ## TBD
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][0], 0], smaller[0][1] - smaller[0][0], smaller[1][0], fc='r'))
        ## TBD
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][0], smaller[1][1]], smaller[0][1] - smaller[0][0], 1 - smaller[1][0], fc='r'))
    else:
        print("Error, trying to color more than 2 dimensional hyperrectangle")

    # re=[]
    # re.append(Rectangle((greater[0][0],0),smaller[0][0]-greater[0][0] , 1, fc='r'))
    # re.append(Rectangle((smaller[0][1],0),greater[0][1]-smaller[0][1] , 1, fc='r'))
    # re.append(Rectangle((smaller[0][0],0),smaller[0][1]-smaller[0][0] , smaller[1][0], fc='r'))
    # re.append(Rectangle((smaller[0][0],smaller[1][1]),smaller[0][1]-smaller[0][0] ,1- smaller[1][0], fc='r'))

    # fig = plt.figure()
    # pic = fig.add_subplot(111, aspect='equal')
    # pic.set_xlabel('p')
    # pic.set_ylabel('q')
    # pic.set_title("red = unsafe region, green = safe region, white = in between")
    # pc = PatchCollection(re,facecolor='r', alpha=0.5)
    # pic.add_collection(pc)


def check_deeper_iter(region, props, n, epsilon, coverage, silent):
    """ New Refining the parameter space into safe and unsafe regions with iterative method using alg1

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    """
    new_tresh = copy.deepcopy(region)

    ## Implement ordering of the props with intervals
    for i in range(len(props) - 1):
        if not silent:
            # print("white: ",globals()["hyper_rectangles_white"])
            print("check_deeper(", new_tresh, [props[i]], ")")
        check_deeper(new_tresh, [props[i]], n, epsilon, coverage, True, 1)

        new_tresh = []
        for interval_index in range(len(region)):
            minimum = 9001
            maximum = 0
            ## Iterate though green regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_sat"])):
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0] < minimum:
                    minimum = globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1] > maximum:
                    maximum = globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1]
            ## Iterate though white regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_white"])):
                if globals()["hyper_rectangles_white"][rectangle_index][interval_index][0] < minimum:
                    minimum = globals()["hyper_rectangles_white"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_white"][rectangle_index][interval_index][1] > maximum:
                    maximum = globals()["hyper_rectangles_white"][rectangle_index][interval_index][1]
            new_tresh.append((minimum, maximum))

        if not silent:
            print("Computed hull of nonred region is:", new_tresh)
        # globals()["hyper_rectangles_white"]=[new_tresh]
    globals()["default_region"] = None
    check_deeper(new_tresh, props, n, epsilon, coverage, True, 1)


def check_interval_in(region, props, intervals, silent=False, called=False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    intervals: (list of pairs/ sympy.Intervals): array of interval to constrain props
    silent: (Bool): if silent printed output is set to minimum
    called: (Bool): if called updates the global variables (use when calling it directly)
    """
    # print(f"props: {props}")
    if not silent:
        print("checking interval in", region, "current time is", datetime.datetime.now())

    if called:
        if not silent:
            print("CALLED")
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        parameters = sorted(list(parameters))
        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]

    ## Assign each parameter its interval
    i = 0
    for param in globals()["parameters"]:
        globals()[param] = mpi(region[i][0], region[i][1])
        i = i + 1

    ## Check that all prop are in its interval
    i = 0
    for prop in props:
        # print(eval(prop))
        # print(intervals[i])
        # print(float(intervals[i].start), float(intervals[i].end))
        # print(mpi(float(intervals[i].start), float(intervals[i].end)))

        ## TBD THIS CAN BE OPTIMISED
        try:
            interval = mpi(float(intervals[i].start), float(intervals[i].end))
        except AttributeError:
            interval = mpi(float(intervals[i][0]), float(intervals[i][1]))

        if not eval(prop) in interval:
            if not silent:
                print(f"property {props.index(prop) + 1} ϵ {eval(prop)}, which is not in the interval {interval}")
            return False
        else:
            if not silent:
                print(f"property {props.index(prop)+1} ϵ {eval(prop)} -- safe")

        i = i + 1

    space.add_green(region)
    return True


def check_interval_out(region, props, intervals, silent=False, called=False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): of functions
    intervals: (list of pairs/ sympy.Intervals): array of interval to constrain props
    silent: (Bool): if silent printed output is set to minimum
    called: (Bool): if called updates the global variables (use when calling it directly)
    """

    if not silent:
        print("checking interval_out", region, "current time is", datetime.datetime.now())

    if called:
        if not silent:
            print("CALLED")
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        parameters = sorted(list(parameters))
        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]
    ## Assign each parameter its interval
    i = 0
    for param in globals()["parameters"]:
        # print(f"param {} in interval {mpi(region[i][0], region[i][1])}")
        try:
            # print(mpi(region[i][0], region[i][1]))
            globals()[param] = mpi(region[i][0], region[i][1])
        except:
            print(f"Error occurred while region: {region}, with param {globals()[param]} of interval {mpi(region[i][0], region[i][1])}")

        i = i + 1

    ## Check that all prop are in its interval
    i = 0
    for prop in props:
        # print(eval(prop))
        # print(intervals[i])
        # print((intervals[i].start, intervals[i].end))
        # print(mpi(0,1) in mpi(0,2))
        # try:
        prop_eval = eval(prop)
        # except :
        #    raise ValueError("Error with prop: ", prop)

        ## TBD THIS CAN BE OPTIMISED
        try:
            ## print(intervals)
            interval = mpi(float(intervals[i].start), float(intervals[i].end))
        except AttributeError:
            interval = mpi(float(intervals[i][0]), float(intervals[i][1]))

        ## If there exists an intersection (neither of these interval is greater in all points)
        if not (prop_eval > interval or prop_eval < interval):
            if not silent:
                print(f"property {props.index(prop) + 1} ϵ {eval(prop)}, which is not outside of interval {interval}")
        else:
            space.add_red(region)
            if not silent:
                print(f"property {props.index(prop) + 1} ϵ {eval(prop)} -- unsafe")
            return True
        i = i + 1
    return False


def private_check_deeper_interval(region, props, intervals, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of pairs of numbers) list of intervals, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of properties
    intervals: (list of pairs/ sympy.Intervals): array of interval to constrain props
    n (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    """

    ## TBD check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## TBD
    # if presampled:
    #    while globals()["que"].size() > 0:
    #        private_check_deeper_interval(*que.dequeue())
    #    return

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print(f"hyperrectangle {region} too small, skipped")
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print(f"rectangle {region} too small, skipped")
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print(f"interval {region} too small, skipped")
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    ## Resolve the result
    # print("gonna check region: ", region)
    if check_interval_out(region, props, intervals, silent=silent, called=False) is True:
        result = "unsafe"
    elif check_interval_in(region, props, intervals, silent=silent, called=False) is True:
        result = "safe"
    else:
        result = "unknown"

    if result == "safe" or result == "unsafe":
        # print("removing region:", region)
        space.remove_white(region)

    if not silent:
        print(n, region, space.get_coverage(), result)
        print()

    if n == 0:
        return
    if result == "safe" or result == "unsafe":
        return

    ## Find maximum interval
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Add calls to the Queue
    # print("adding",[copy.deepcopy(foo),prop,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo),prop,n-1,epsilon,coverage,silent]))
    # print("adding",[copy.deepcopy(foo2),prop,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo2),prop,n-1,epsilon,coverage,silent]))
    globals()["que"].enqueue([copy.deepcopy(foo), props, intervals, n - 1, epsilon, coverage, silent])
    globals()["que"].enqueue([copy.deepcopy(foo2), props, intervals, n - 1, epsilon, coverage, silent])

    ## Execute the queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_interval(*que.dequeue())


def create_matrix(size_q, dim):
    """ Return **dim** dimensional array of length **size_q** in each dimension

    Args
    -------
    size_q: (int): number of samples in dimension
    dim: (int): number of dimensions

    """
    return np.array(private_create_matrix(size_q, dim, dim))


def private_create_matrix(size_q, dim, n_param):
    """ Return **dim** dimensional array of length **size_q** in each dimension

    Args
    -------
    size_q: (int): number of samples in dimension
    dim: (int): number of dimensions
    n_param: (int): dummy parameter

    @author: xtrojak, xhajnal
    """
    if dim == 0:
        point = []
        for i in range(n_param):
            point.append(0)
        return [point, 9]
    return [private_create_matrix(size_q, dim - 1, n_param) for _ in range(size_q)]


def sample(space, props, size_q, compress=False, silent=True, save=False):
    """ Samples the space in **size_q** samples in each dimension and saves if the point is in respective interval

    Args
    -------
    space: (space.RefinedSpace): space
    props:  (list of strings): array of properties
    size_q: (int): number of samples in dimension
    silent: (Bool): if silent printed output is set to minimum
    compress: (Bool): if True, only a conjunction of the values (prop in the interval) is used
    save: (Bool): if True output is pickled

    Returns
    --------
    A map from point to list of Bool whether f(point) in interval[index]

    """
    parameter_values = []
    parameter_indices = []
    for param in range(len(space.params)):
        parameter_values.append(np.linspace(space.region[param][0], space.region[param][1], size_q, endpoint=True))
        parameter_indices.append(np.asarray(range(0, size_q)))

    sampling = create_matrix(size_q, len(space.params))
    if not silent:
        print("sampling here")
        print("size_q", size_q)
        print("space.params", space.params)
        print("sampling", sampling)
    parameter_values = cartesian_product(*parameter_values)
    parameter_indices = cartesian_product(*parameter_indices)

    # if (len(space.params) - 1) == 0:
    #    parameter_values = linspace(0, 1, size_q, endpoint=True)[newaxis, :].T
    if not silent:
        print("parameter_values", parameter_values)
        print("parameter_indices", parameter_indices)
        # print("a sample:", sampling[0][0])
    i = 0
    ## For each parametrisation
    for parameter_value in parameter_values:
        ## For each parameter
        for param in range(len(space.params)):
            locals()[space.params[param]] = float(parameter_value[param])
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

        sampling[tuple(parameter_indices[i])][0] = list(parameter_value)

        satisfied_list = []
        ## For each property,interval
        for index in range(len(props)):
            # print(props[index])
            # print("type(props[index])", type(props[index]))
            # for param in range(len(space.params)):
            #     print(space.params[param], parameter_value[param])
            #     print("type(space.params[param])", type(space.params[param]))
            #     print("type(parameter_value[param])", type(parameter_value[param]))

            spam = (eval(props[index]))
            if isinstance(spam, z3.z3.BoolRef):
                if z3.simplify(spam):
                    satisfied_list.append(True)
                else:
                    satisfied_list.append(False)
            else:
                if spam:
                    satisfied_list.append(True)
                else:
                    satisfied_list.append(False)

            ## print("cycle")
            ## print(sampling[tuple(parameter_indices[i])])

        if False in satisfied_list:
            # print("adding unsat", sampling[tuple(parameter_indices[i])][0])
            space.add_unsat_samples([sampling[tuple(parameter_indices[i])][0]])
            if compress:
                sampling[tuple(parameter_indices[i])][1] = False
            else:
                sampling[tuple(parameter_indices[i])][1] = satisfied_list
        else:
            # print("adding sat", sampling[tuple(parameter_indices[i])][0])
            space.add_sat_samples([sampling[tuple(parameter_indices[i])][0]])
            if compress:
                sampling[tuple(parameter_indices[i])][1] = True
            else:
                sampling[tuple(parameter_indices[i])][1] = satisfied_list
        i = i + 1

    ## Setting flag to not visualise sat if no unsat and vice versa
    space.gridsampled = True

    ## Saving the sampled space as pickled dictionary
    if save:
        if save is True:
            save = str(strftime("%d-%b-%Y-%H-%M-%S", localtime()))
        pickle.dump(sampling, open(os.path.join(refinement_results, ("Sampled_space_" + save).split(".")[0] + ".p"), "wb"))

    return sampling


def refine_into_rectangles(sampled_space, silent=True):
    """ Refines the sampled space into hyperrectangles such that rectangle is all sat or all unsat

    Args
    -------
    sampled_space: (space.RefinedSpace): space
    silent: (Bool): if silent printed output is set to minimum

    Yields
    --------
    Hyperectangles of length at least 2 (in each dimension)
    """
    size_q = len(sampled_space[0])
    dimensions = len(sampled_space.shape) - 1
    if not silent:
        print("\n refine into rectangles here ")
        print(type(sampled_space))
        print("shape", sampled_space.shape)
        print("space:", sampled_space)
        print("size_q:", size_q)
        print("dimensions:", dimensions)
    # find_max_rectangle(sampled_space, [0, 0])

    if dimensions == 2:
        parameter_indices = []
        for param in range(dimensions):
            parameter_indices.append(np.asarray(range(0, size_q)))
        parameter_indices = cartesian_product(*parameter_indices)
        if not silent:
            print(parameter_indices)
        a = []
        for point in parameter_indices:
            # print("point", point)
            result = find_max_rectangle(sampled_space, point, silent=silent)
            if result is not None:
                a.append(result)
        if not silent:
            print(a)
        return a
    else:
        print(f"Sorry, {dimensions} dimensions TBD")


def find_max_rectangle(sampled_space, starting_point, silent=True):
    """ Finds the largest hyperrectangles such that rectangle is all sat or all unsat from starting point in positive direction

    Args
    -------
    sampled_space: (space.RefinedSpace): space
    starting_point: (list of floats): a point in the space to start search in
    silent: (Bool): if silent printed output is set to minimum

    Returns
    --------
    triple(starting point, end point, is_sat)
    """
    size_q = len(sampled_space[0])
    dimensions = len(sampled_space.shape) - 1
    if dimensions == 2:
        index_x = starting_point[0]
        index_y = starting_point[1]
        length = 2
        start_value = sampled_space[index_x][index_y][1]
        if not silent:
            print("Dealing with 2D space at starting point", starting_point, "with start value", start_value)
        if start_value == 2:
            if not silent:
                print(starting_point, "already added, skipping")
            return
        if index_x >= size_q - 1 or index_y >= size_q - 1:
            if not silent:
                print(starting_point, "is at the border, skipping")
            sampled_space[index_x][index_y][1] = 2
            return

        ## While other value is found
        while True:
            ## print(index_x+length)
            ## print(sampled_space[index_x:index_x+length, index_y:index_y+length])
            values = list(
                map(lambda x: [y[1] for y in x], sampled_space[index_x:index_x + length, index_y:index_y + length]))
            # print(values)
            foo = []
            for x in values:
                for y in x:
                    foo.append(y)
            values = foo
            if not silent:
                print("Values found: ", values)
            if (not start_value) in values:
                length = length - 1
                if not silent:
                    print(f"rectangle [[{index_x},{index_y}],[{index_x + length},{index_y + length}]] does not satisfy all sat not all unsat")
                break
            elif index_x + length > size_q or index_y + length > size_q:
                if not silent:
                    print(f"rectangle [[{index_x},{index_y}],[{index_x + length},{index_y + length}]] is out of box, using lower value")
                length = length - 1
                break
            else:
                length = length + 1

        ## Mark as seen (only this point)
        sampled_space[index_x][index_y][1] = 2
        length = length - 1

        ## Skip if only this point safe/unsafe
        if length == 0:
            if not silent:
                print("Only single point found, skipping")
            return

        ## print((sampled_space[index_x, index_y], sampled_space[index_x+length-2, index_y+length-2]))

        # print(type(sampled_space))
        # place(sampled_space, sampled_space==False, 2)
        # print("new sampled_space: \n", sampled_space)

        # print("the space to be marked: \n", sampled_space[index_x:(index_x + length - 1), index_y:(index_y + length - 1)])
        if not silent:
            print("length", length)

        ## old result (in corner points format)
        # result = (sampled_space[index_x, index_y], sampled_space[index_x + length - 1, index_y + length - 1])

        ## new result (in region format)
        result = ([[sampled_space[index_x, index_y][0][0], sampled_space[index_x + length, index_y][0][0]],
                   [sampled_space[index_x, index_y][0][1], sampled_space[index_x, index_y + length][0][1]]])
        print(f"adding rectangle [[{index_x},{index_y}],[{index_x + length},{index_y + length}]] with value [{sampled_space[index_x, index_y][0]},{sampled_space[index_x + length, index_y + length][0]}]")

        ## OLD seen marking (seeting seen for all searched points)
        # place(sampled_space[index_x:(index_x + length), index_y:(index_y + length)],
        #      sampled_space[index_x:(index_x + length), index_y:(index_y + length)] == False, 2)
        # place(sampled_space[index_x:(index_x + length), index_y:(index_y + length)],
        #      sampled_space[index_x:(index_x + length), index_y:(index_y + length)] == True, 2)

        print("new sampled_space: \n", sampled_space)
        ## globals()["que"].enqueue([[index_x, index_x+length-2],[index_y, index_y+length-2]],start_value)
        return result
    else:
        print(f"Sorry, {dimensions} dimensions TBD")
