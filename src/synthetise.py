import datetime
import os
import re
import socket
import sys
import threading
import time
import platform
from collections.abc import Iterable
from termcolor import colored
from math import log
from numpy import prod
import itertools

import numpy as np
from sympy import Interval
from mpmath import mpi
from matplotlib.patches import Rectangle
import unittest

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
z3_path = config.get("paths", "z3_path")

if not os.path.exists(z3_path):
    raise OSError("Directory does not exist: " + str(z3_path))

os.chdir(cwd)

print("z3_path", z3_path)

# import struct
# print("You are running "+ str(struct.calcsize("P") * 8)+"bit Python, please verify that installed z3 is compatible")
# print("path: ", os.environ["PATH"])

# print(os.environ["PATH"])

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

## Try to import z3
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
except:
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
    def printQueue(self):
        return self.queue


def check_unsafe(region, props, intervals, silent=False, called=False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: (list of sympy.Interval): array of intervals to constrain properties
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

        space = RefinedSpace(copy.deepcopy(region), parameters, [], [])
    else:
        space = globals()["space"]

    s = Solver()

    # if not silent:
    #    print("with parameters", globals()["parameters"])

    ## Adding regional restrictions to solver
    # for param_index in range(len(space.params)):
    #     s.add(space.params[param_index] >= region[param_index][0])
    #     s.add(space.params[param_index] <= region[param_index][1])

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        # print("globals()[param]", globals()[param])
        # print("region[j][0]", region[j][0])
        s.add(globals()[param] >= region[j][0])
        s.add(globals()[param] <= region[j][1])
        j = j + 1

    ## Adding property in the interval restrictions to solver
    for i in range(0, len(props)):
        # if intervals[i]<100/n_samples:
        #    continue

        ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        # if  intervals[i]<0.01:
        #    continue

        s.add(eval(props[i]) >= intervals[i].start, eval(props[i]) <= intervals[i].end)
        # print(prop[i],intervals[i])

    if s.check() == sat:
        return s.model()
    else:
        space.add_red(region)
        return True


def check_safe(region, props, intervals, silent=False, called=False):
    """ Check if the given region is safe or not

    It means whether for all parametrisations in **region** every property(prop) is evaluated within the given
    **interval**, otherwise it is not safe and counterexample is returned.

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: (list of sympy.Interval): array of intervals to constrain properties
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

        space = RefinedSpace(copy.deepcopy(region), parameters, [], [])
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

    ## Adding property in the interval restrictions to solver

    formula = Or(Not(eval(props[0]) >= intervals[0].start), Not(eval(props[0]) <= intervals[0].end))

    ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
    # if intervals[0]<0.01:
    #    formula = Or(Not(eval(prop[0]) > intervals[0][0])), Not(eval(prop[0]) < intervals[0][1])))
    # else:
    #    formula = False

    for i in range(1, len(props)):
        # if intervals[i]<100/n_samples:
        #    continue

        ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        # if  intervals[i]<0.01:
        #    continue
        formula = Or(formula, Or(Not(eval(props[i]) >= intervals[i].start), Not(eval(props[i]) <= intervals[i].end)))
    s.add(formula)
    # print(s.check_unsafe())
    # return s.check_unsafe()
    if s.check() == unsat:
        space.add_green(region)
        return True
    else:
        return s.model()


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

    interval1: (list of pairs) (hyper)space defined by the regions
    interval2: (list of pairs) (hyper)space defined by the regions
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

    region1: (list of pairs) (hyper)space defined by the regions
    region2: (list of pairs) (hyper)space defined by the regions
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


def check_deeper(region, props, intervals, n, epsilon, coverage, silent, version, size_q=False, time_out=False, debug=False):
    """ Refining the parameter space into safe and unsafe regions with respective alg/method
    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    version: (Int): version of the algorithm to be used
    size_q: (Int): number of samples in dimension used for presampling
    time_out: (Int): time out in minutes
    debug: (Bool): if debug extensive print will be used
    """

    ## INITIALISATION
    ### Regions
    ### Taking care of unchangable tuples
    for interval_index in range(len(region)):
        region[interval_index] = [region[interval_index][0], region[interval_index][1]]

    ### Params
    if not isinstance(props, Iterable):
        raise Exception("Given properties are not iterable, to use single property use list of length 1")

    globals()["parameters"] = set()
    for polynome in props:
        globals()["parameters"].update(find_param(polynome))
    globals()["parameters"] = sorted(list(globals()["parameters"]))
    parameters = globals()["parameters"]

    globals()["space"] = RefinedSpace(copy.deepcopy(region), parameters, [], [])
    space = globals()["space"]

    globals()["default_region"] = copy.deepcopy(region)

    if not silent:
        print("the area is: ", space.region)
        print("the volume of the whole area is:", space.get_volume())
        print()

    ## Choosing version/algorithm here
    ## If using z3 initialise the parameters
    if version <= 4:
        for param in parameters:
            globals()[param] = Real(param)

    start_time = time.time()

    if debug:
        print("region", region)
        print("props", props)
        print("intervals", intervals)

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

        # globals()["space"] = RefinedSpace(copy.deepcopy(region), parameters, [], [])

        to_be_searched = sample(space, props, intervals, size_q, compress=True, silent=not debug)
        if debug:
            print(type(to_be_searched))
            print("sampled space: ", to_be_searched)

        ## PARSE SAT POINTS
        sat_points = []

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
                print(f"Fixed intervals bordering the sat hull are: {spam}")
                if debug:
                    print(colored("I was here", 'red'))
                space.remove_white(region)
                regions = refine_by(region, spam, debug)
                for subregion in regions:
                    space.add_white(subregion)
        else:
            print("No sat points in the samples")

        ## If there is only the default region to be refined in the whitespace
        if len(space.get_white()) == 1:
            ## PARSE UNSAT POINTS
            unsat_points = []
            for point in to_be_searched:
                if point[1] is False:
                    unsat_points.append(point[0])
            if debug:
                print("unsatisfying points: ", unsat_points)

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
                    print(f"Points bordering the unsat hull are: {unsat_min},{unsat_max}")

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
        white_space = copy.deepcopy(space.get_white())

        # print(globals()["parameters"])
        # print(space.params)
        ## Setting the param back to z3 definition
        if version <= 4:
            for param in parameters:
                globals()[param] = Real(param)

        if debug:
            print("region now", region)
            print("space white", white_space)

        print("Presampling resulted in splicing the region into these subregions: ", white_space)
        print(f"It took {socket.gethostname()} {round(time.time() - start_time)} second(s)")

        ## Iterating through the regions
        for rectangle in white_space:
            start_time = time.time()
            ## To get more similar result substituting the number of splits from the max_depth
            if debug:
                print("max_depth = ", max(1, n-(int(log(len(white_space), 2)))))
                print("refining", rectangle)

            ## THE PROBLEM IS THAT COVERAGE IS COMPUTED FOR THE WHOLE SPACE NOT ONLY FOR THE GIVEN REGION
            rectangle_size = []
            for interval in rectangle:
                rectangle_size.append(interval[1] - interval[0])
            rectangle_size = prod(rectangle_size)

            # print("rectangle", rectangle, " props", props, "intervals", intervals, "silent", silent)
            # print("current coverage", space.get_coverage())
            # print("whole area", space.get_volume())
            # print("rectangle_size", rectangle_size)

            ## Setting the coverage as lower value between desired coverage and the proportional expected coverage
            next_coverage = min(coverage, (space.get_coverage() + (rectangle_size / space.get_volume())*coverage))
            # print("next coverage", next_coverage)

            if debug:
                print("region", rectangle)
                print("props", props)
                print("intervals", intervals)

            if version == 1:
                print(f"Using DFS method to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper(rectangle, props, intervals, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, time_out=time_out)
            elif version == 2:
                print(f"Using BFS method to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_queue(rectangle, props, intervals, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, time_out=time_out)
            elif version == 3:
                print(f"Using BFS method with passing examples to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_queue_checking(rectangle, props, intervals, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, None, time_out=time_out)
            elif version == 4:
                print(f"Using BFS method with passing examples and counterexamples to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_queue_checking_both(rectangle, props, intervals, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, None, time_out=time_out)
            elif version == 5:
                print(f"Using interval method to solve spliced rectangle number {white_space.index(rectangle)+1}")
                private_check_deeper_interval(rectangle, props, intervals, max(1, n - (int(log(len(white_space), 2)))), epsilon, next_coverage, silent, time_out=time_out)
            else:
                print(colored("Chosen version not found", "red"))
                return

            ## Showing the step refinements of respective rectangles from the white space
            space.show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version} \n It took {socket.gethostname()} {round(time.time() - start_time)} second(s)")
            print()

        ## OLD REFINEMENT HERE
        # # to_be_searched = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], , compress=True, silent=False)

        # to_be_searched = refine_into_rectangles(to_be_searched, silent=False)

        # print("to_be_searched: ", to_be_searched)
        # globals()["que"] = Queue()

        # for rectangle in to_be_searched:
        #     print(rectangle)
        #     # print("safe", space.sat)
        #     print("unsafe", space.unsat)
        #     space.add_white(rectangle)
        #     private_check_deeper_interval(rectangle, props, intervals, 0, epsilon, coverage, silent)

    elif version == 1:
        print("Using DFS method")
        if time_out:
            print("using timeout", time_out)
            timeout(private_check_deeper, (region, props, intervals, n, epsilon, coverage, silent), timeout_duration=time_out, default=4)
        else:
            private_check_deeper(region, props, intervals, n, epsilon, coverage, silent, time_out=time_out)
    elif version == 2:
        print("Using BFS method")
        globals()["que"] = Queue()
        private_check_deeper_queue(region, props, intervals, n, epsilon, coverage, silent, time_out=time_out)
    elif version == 3:
        print("Using BFS method with passing examples")
        globals()["que"] = Queue()
        private_check_deeper_queue_checking(region, props, intervals, n, epsilon, coverage, silent, None, time_out=time_out)
    elif version == 4:
        print("Using BFS method with passing examples and counterexamples")
        print("rectangle", region)
        print("props", props)
        print("intervals", intervals)
        globals()["que"] = Queue()
        private_check_deeper_queue_checking_both(region, props, intervals, n, epsilon, coverage, silent, None, time_out=time_out)
    elif version == 5:
        print("Using interval arithmetic")
        globals()["que"] = Queue()
        private_check_deeper_interval(region, props, intervals, n, epsilon, coverage, silent, time_out=time_out)
    else:
        print(colored("Chosen version not found", "red"))

    ## VISUALISATION
    if not size_q:
        space.show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version} \n It took {socket.gethostname()} {round(time.time() - start_time)} second(s)")
    print("result coverage is: ", space.get_coverage())
    return space


def private_check_deeper(region, props, intervals, n, epsilon, coverage, silent, time_out=False):
    """ Refining the parameter space into safe and unsafe regions
    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props: (list of strings): array of polynomials
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    time_out: (Int): time out in minutes
    """

    ## TBD check consistency
    # print(region,prop,intervals,n,epsilon,coverage,silent)
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
    if check_unsafe(region, props, intervals, silent) is True:
        result = "unsafe"
    elif check_safe(region, props, intervals, silent) is True:
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
                private_check_deeper(foo, props, intervals, n - 1, epsilon, coverage, silent)
                if space.get_coverage() > coverage:
                    return f"coverage {space.get_coverage()} is above the threshold"
                private_check_deeper(foo2, props, intervals, n - 1, epsilon, coverage, silent)
            else:
                print(n, foo, space.get_coverage(),
                      private_check_deeper(foo, props, intervals, n - 1, epsilon, coverage, silent))
                if space.get_coverage() > coverage:
                    return f"coverage {space.get_coverage()} is above the threshold"
                print(n, foo2, space.get_coverage(),
                      private_check_deeper(foo2, props, intervals, n - 1, epsilon, coverage, silent))
    return result


def private_check_deeper_queue(region, props, intervals, n, epsilon, coverage, silent, time_out=False):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props: (list of strings): array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    time_out: (Int): time out in minutes
    """

    ## TBD check consitency
    # print(region,prop,intervals,n,epsilon,coverage,silent)
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
    # print("hello check =",check_unsafe(region,prop,intervals,silent))
    # print("hello check safe =",check_safe(region,prop,n_samples,silent))
    if check_unsafe(region, props, intervals, silent) is True:
        result = "unsafe"
    elif check_safe(region, props, intervals, silent) is True:
        result = "safe"
    else:
        result = "unknown"

    if result == "safe" or result == "unsafe":
        # print("removing region:", region)
        space.remove_white(region)

    if not silent:
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
    # print("adding",[copy.deepcopy(foo),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo),prop,intervals,n-1,epsilon,coverage,silent]))
    # print("adding",[copy.deepcopy(foo2),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo2),prop,intervals,n-1,epsilon,coverage,silent]))
    globals()["que"].enqueue([copy.deepcopy(foo), props, intervals, n - 1, epsilon, coverage, silent])
    globals()["que"].enqueue([copy.deepcopy(foo2), props, intervals, n - 1, epsilon, coverage, silent])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue(*que.dequeue())


def private_check_deeper_queue_checking(region, props, intervals, n, epsilon, coverage, silent, model=None, time_out=False):
    """ THIS IS OBSOLETE METHOD, HERE JUST TO BE COMPARED WITH THE NEW ONE

    Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props: (list of strings): array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    model: (example,counterexample) of the satisfaction in the given region
    time_out: (Int): time out in minutes
    """

    ## TBD check consitency
    # print(region,prop,intervals,n,epsilon,coverage,silent)
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
        example = check_unsafe(region, props, intervals, silent)
        # counterexample = check_safe(region,prop,intervals,silent)
    elif model[0] is None:
        example = check_unsafe(region, props, intervals, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        if not silent:
            print(n, region, space.get_coverage(), "unsafe")
        return
    elif check_safe(region, props, intervals, silent) is True:
        space.remove_white(region)
        if not silent:
            print(n, region, space.get_coverage(), "safe")
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
        [copy.deepcopy(foo), props, intervals, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue(
        [copy.deepcopy(foo2), props, intervals, n - 1, epsilon, coverage, silent, model_high])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking(*que.dequeue())


def private_check_deeper_queue_checking_both(region, props, intervals, n, epsilon, coverage, silent,
                                             model=None, time_out=False):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    model: (example, counterexample) of the satisfaction in the given region
    time_out: (Int): time out in minutes
    """

    ## TBD check consitency
    # print(region,prop,intervals,n,epsilon,coverage,silent)
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
        example = check_unsafe(region, props, intervals, silent)
        counterexample = check_safe(region, props, intervals, silent)
    elif model[0] is None:
        example = check_unsafe(region, props, intervals, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]
    if model is not None:
        if model[1] is None:
            counterexample = check_safe(region, props, intervals, silent)
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
    else:  ## unknown
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
    globals()["que"].enqueue([copy.deepcopy(foo), props, intervals, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue([copy.deepcopy(foo2), props, intervals, n - 1, epsilon, coverage, silent, model_high])

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


def check_deeper_iter(region, props, intervals, n, epsilon, coverage, silent, time_out=False):
    """ New Refining the parameter space into safe and unsafe regions with iterative method using alg1

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props (list of strings): array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    time_out: (Int): time out in minutes
    """
    new_tresh = copy.deepcopy(region)

    ## Implement ordering of the props with intervals
    for i in range(len(props) - 1):
        if not silent:
            # print("white: ",globals()["hyper_rectangles_white"])
            print("check_deeper(", new_tresh, [props[i]], [intervals[i]], ")")
        check_deeper(new_tresh, [props[i]], [intervals[i]], n, epsilon, coverage, True, 1)

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
    check_deeper(new_tresh, props, intervals, n, epsilon, coverage, True, 1)


def check_interval_in(region, props, intervals, silent=False, called=False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args
    ----------
    region: (list of intervals) low and high bound, defining the parameter space to be refined
    props: (list of strings) array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: (list of sympy.Interval): array of intervals to constrain properties
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
        space = RefinedSpace(copy.deepcopy(region), parameters, [], [])
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
        if not eval(prop) in mpi(float(intervals[i].start), float(intervals[i].end)):
            if not silent:
                print(f"property {props.index(prop) + 1} ϵ {eval(prop)}, which is not in the interval {mpi(float(intervals[i].start), float(intervals[i].end))}")
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
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props: (list of strings) array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: (list of sympy.Interval): array of intervals to constrain properties
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
        space = RefinedSpace(copy.deepcopy(region), parameters, [], [])
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

        interval = mpi(float(intervals[i].start), float(intervals[i].end))
        ## If there exists an intersection (neither of these interval is greater in all points)
        if not (prop_eval > interval or prop_eval < interval):
            if not silent:
                print(f"property {props.index(prop) + 1} ϵ {eval(prop)}, which is not outside of interval {mpi(float(intervals[i].start), float(intervals[i].end))}")
        else:
            space.add_red(region)
            if not silent:
                print(f"property {props.index(prop) + 1} ϵ {eval(prop)} -- unsafe")
            return True
        i = i + 1
    return False


def private_check_deeper_interval(region, props, intervals, n, epsilon, coverage, silent, presampled=False, time_out=False):
    """ Refining the parameter space into safe and unsafe regions

    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props: (list of strings) array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent printed output is set to minimum
    presampled: (Bool) if True, use presampled subregions to start refinement with
    time_out: (Int): time out in minutes
    """

    ## TBD check consistency
    # print(region,prop,intervals,n,epsilon,coverage,silent)
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
    if check_interval_out(region, props, intervals, silent, called=False) is True:
        result = "unsafe"
    elif check_interval_in(region, props, intervals, silent, called=False) is True:
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
    # print("adding",[copy.deepcopy(foo),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo),prop,intervals,n-1,epsilon,coverage,silent]))
    # print("adding",[copy.deepcopy(foo2),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.deepcopy(foo2),prop,intervals,n-1,epsilon,coverage,silent]))
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


def sample(space, props, intervals, size_q, compress=False, silent=True):
    """ Samples the space in **size_q** samples in each dimension and saves if the point is in respective interval

    Args
    -------
    space: (space.RefinedSpace): space
    props: (list of strings): array of functions (polynomials or general rational functions in the case of Markov Chains)
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    size_q: (int): number of samples in dimension
    silent: (Bool): if silent printed output is set to minimum
    compress: (Bool): if True, only a conjunction of the values (prop in the interval) is used

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
            globals()[space.params[param]] = parameter_value[param]
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
            if eval(props[index]) in mpi(float(intervals[index].start), float(intervals[index].end)):
                satisfied_list.append(True)
            else:
                satisfied_list.append(False)

            ## print("cycle")
            ## print(sampling[tuple(parameter_indices[i])])

        if compress:
            if False in satisfied_list:
                sampling[tuple(parameter_indices[i])][1] = False
            else:
                sampling[tuple(parameter_indices[i])][1] = True
        else:
            sampling[tuple(parameter_indices[i])][1] = satisfied_list
        i = i + 1
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


## CALL FUNCTION IN A SEPARATE THREAD WITH A GIVEN TIMEOUT
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    print("in timeout function, the timeout set to ", timeout_duration)
    it = InterruptableThread()
    ## added by matej
    # it.daemon = True

    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        it._stop()
        return default
    else:
        return it.result


class TimeoutError(Exception):
    pass


def timelimit(timeout):
    def internal(function):
        def internal2(*args, **kw):
            class Calculator(threading.Thread):
                def __init__(self):
                    threading.Thread.__init__(self)
                    self.result = None
                    self.error = None

                def run(self):
                    try:
                        self.result = function(*args, **kw)
                    except:
                        self.error = sys.exc_info()[0]

            c = Calculator()
            c.start()
            c.join(timeout)
            if c.isAlive():
                raise TimeoutError
            if c.error:
                raise c.error
            return c.result

        return internal2

    return internal


class TestLoad(unittest.TestCase):
    def test_Interval(self):
        print()
        print(colored("Interval test here", 'blue'))
        self.assertEqual(1.0 in mpi(1, 1), True)
        self.assertEqual(1.0 in mpi(1, 2), True)
        self.assertEqual(1.0 in mpi(0, 1), True)
        self.assertEqual(1.0 in mpi(0, 2), True)

        self.assertEqual(5.0 not in mpi(1, 1), True)
        self.assertEqual(5.0 not in mpi(1, 2), True)
        self.assertEqual(5.0 not in mpi(0, 1), True)
        self.assertEqual(5.0 not in mpi(0, 2), True)

        self.assertEqual(mpi(1, 1) in mpi(1, 1), True)
        self.assertEqual(mpi(1, 1) in mpi(1, 2), True)
        self.assertEqual(mpi(1, 1) in mpi(0, 1), True)
        self.assertEqual(mpi(1, 1) in mpi(0, 2), True)

        self.assertEqual(mpi(1, 2) in mpi(1, 3), True)
        self.assertEqual(mpi(1, 2) in mpi(0, 2), True)
        self.assertEqual(mpi(1, 2) in mpi(0, 3), True)
        self.assertEqual(mpi(1, 2) not in mpi(0, 1), True)
        self.assertEqual(mpi(1, 2) not in mpi(2, 3), True)
        self.assertEqual(mpi(1, 2) not in mpi(1.5, 2), True)

    def test_to_interval(self):
        print(colored("Checking transformation of a set of points into a set of intervals here", 'blue'))
        self.assertEqual(to_interval([(0, 2), (1, 3)]), [[0, 1], [2, 3]])
        self.assertEqual(to_interval([(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0), (0, 2, 3), (0, 0, 3), (1, 0, 3), (1, 2, 3)]), [[0, 1], [0, 2], [0, 3]])

    def test_is_in(self):
        print(colored("Checking if the first region is within the second one here", 'blue'))
        self.assertEqual(is_in([(1, 4)], [(1, 4)]), True)
        self.assertEqual(is_in([(1, 4)], [(0, 5)]), True)
        self.assertEqual(is_in([(1, 4)], [(0, 3)]), False)
        self.assertEqual(is_in([(1, 4)], [(2, 5)]), False)

        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 2), (1, 3)]), True)
        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 3), (1, 4)]), True)
        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 1), (1, 4)]), False)
        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 2), (1, 2)]), False)

    def test_check_single(self):
        print(colored("Check (un)safe with single properties here", 'blue'))
        ## IS IN
        ## def check_safe(region, props, intervals, silent=False, called=False):
        # check_deeper([(0, 1)], ["x"], [Interval(0, 1)], 0, 0.1, 1, True, 4)
        self.assertEqual(check_safe([(0, 1)], ["x"], [Interval(0, 1)], silent=True, called=True), True)

        # check_deeper([(1, 1)], ["x"], [Interval(0, 2)], 0, 0.1, 1, True, 4)
        self.assertEqual(check_safe([(1, 1)], ["x"], [Interval(0, 2)], silent=True, called=True), True)

        # check_deeper([(0, 1)], ["x"], [Interval(0.5, 3)], 10, 0.1, 1, True, 4)
        self.assertIsInstance(check_safe([(0, 1)], ["x"], [Interval(0.5, 3)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an counter example

        # check_deeper([(0, 1)], ["x"], [Interval(2, 3)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_safe([(0, 1)], ["x"], [Interval(2, 3)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an counter example

        # check_deeper([(1, 4)], ["x"], [Interval(2, 3)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_safe([(1, 4)], ["x"], [Interval(2, 3)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an counter example

        # check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], 0, 0.1, 1, True, 4)
        self.assertEqual(check_safe([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True, called=True), True)

        # check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_safe([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], silent=True, called="safe"),
                              z3.z3.ModelRef)  ## has an counter example

        ## IS OUT
        ## def check_unsafe(region, props, intervals, silent=False, called=False):
        # check_deeper([(0, 1)], ["x"], [Interval(0, 1)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_unsafe([(0, 1)], ["x"], [Interval(0, 1)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an example

        # check_deeper([(1, 1)], ["x"], [Interval(0, 2)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_unsafe([(1, 1)], ["x"], [Interval(0, 2)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an example

        # check_deeper([(0, 1)], ["x"], [Interval(0.5, 3)], 10, 0.1, 1, True, 4)
        self.assertIsInstance(check_unsafe([(0, 1)], ["x"], [Interval(0.5, 3)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an example

        # check_deeper([(0, 1)], ["x"], [Interval(2, 3)], 0, 0.1, 1, True, 4)
        self.assertEqual(check_unsafe([(0, 1)], ["x"], [Interval(2, 3)], silent=True, called=True), True)

        # check_deeper([(1, 4)], ["x"], [Interval(2, 3)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_unsafe([(1, 4)], ["x"], [Interval(2, 3)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an example

        # check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_unsafe([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an example

        # check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 0, 0.1, 1, True, 4)
        self.assertIsInstance(check_unsafe([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], silent=True, called=True),
                              z3.z3.ModelRef)  ## has an example

        self.assertEqual(check_unsafe([(0, 1)], ["x"], [Interval(2, 3)], silent=True, called=True), True)
        self.assertEqual(check_unsafe([(1, 1)], ["x"], [Interval(2, 3)], silent=True, called=True), True)
        self.assertIsInstance(check_unsafe([(0, 1)], ["x"], [Interval(1, 3)], silent=True, called=True), z3.z3.ModelRef)
        self.assertIsInstance(check_unsafe([(0, 3)], ["x"], [Interval(2, 3)], silent=True, called=True), z3.z3.ModelRef)
        self.assertIsInstance(check_unsafe([(1, 4)], ["x"], [Interval(2, 3)], silent=True, called=True), z3.z3.ModelRef)
        self.assertIsInstance(check_unsafe([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True, called=True), z3.z3.ModelRef)
        self.assertEqual(check_unsafe([(0, 1), (0, 1)], ["x+y"], [Interval(4, 5)], silent=True, called=True), True)

    def test_check_multiple(self):
        print(colored("Check (un)safe with multiple properties here", 'blue'))
        ## IS IN
        ## def check_safe(region, props, intervals, silent=False, called=False):
        self.assertEqual(
            check_safe([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 2)], silent=True, called=True), True)
        self.assertIsInstance(
            check_safe([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), z3.z3.ModelRef)

        ## !!!TRICKY
        self.assertIsInstance(
            check_safe([(0, 2)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), z3.z3.ModelRef)

        self.assertEqual(
            check_safe([(0, 1), (0, 1)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), True)
        self.assertIsInstance(
            check_safe([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), z3.z3.ModelRef)

        ## IS OUT
        ## def check_unsafe(region, props, intervals, silent=False, called=False):
        self.assertEqual(
            check_unsafe([(0, 1)], ["x", "2*x"], [Interval(2, 3), Interval(3, 4)], silent=True, called=True), True)
        self.assertIsInstance(
            check_unsafe([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), z3.z3.ModelRef)

        ## !!!TRICKY
        self.assertIsInstance(
            check_unsafe([(0, 2)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False, called=True), z3.z3.ModelRef)

        self.assertEqual(
            check_unsafe([(0, 1), (0, 1)], ["x", "y"], [Interval(2, 3), Interval(2, 3)], silent=True, called=True), True)
        self.assertIsInstance(
            check_unsafe([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), z3.z3.ModelRef)

    def test_check_interval_single(self):
        print(colored("Check interval (un)safe with single properties here", 'blue'))
        ## IS IN
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0, 1)], silent=True, called=True), True)
        self.assertEqual(check_interval_in([(1, 1)], ["x"], [Interval(0, 2)], silent=True, called=True), True)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0.5, 3)], silent=True, called=True), False)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(2, 3)], silent=True, called=True), False)
        self.assertEqual(check_interval_in([(1, 4)], ["x"], [Interval(2, 3)], silent=True, called=True), False)
        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True, called=True), True)
        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], silent=True, called=True), False)

        ## IS OUT
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(2, 3)], silent=True, called=True), True)
        self.assertEqual(check_interval_out([(1, 1)], ["x"], [Interval(2, 3)], silent=True, called=True), True)
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(1, 3)], silent=True, called=True), False)
        self.assertEqual(check_interval_out([(0, 3)], ["x"], [Interval(2, 3)], silent=True, called=True), False)
        self.assertEqual(check_interval_out([(1, 4)], ["x"], [Interval(2, 3)], silent=True, called=True), False)

        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True, called=True), False)
        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(4, 5)], silent=True, called=True), True)

    def test_check_interval_multiple(self):
        print(colored("Check interval (un)safe with multiple properties here", 'blue'))
        ## IS IN
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 2)], silent=True, called=True), True)
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), False)

        ## !!!TRICKY
        self.assertEqual(
            check_interval_in([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=True, called=True), False)

        self.assertEqual(
            check_interval_in([(0, 1), (0, 1)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=True,
                              called=True), True)
        self.assertEqual(
            check_interval_in([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=True,
                              called=True), False)

        ## IS OUT
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(2, 3), Interval(3, 4)], silent=True, called=True), True)
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), False)
        ## !!!TRICKY
        self.assertEqual(
            check_interval_out([(0, 2)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=True, called=True), False)

        self.assertEqual(
            check_interval_out([(0, 1), (0, 1)], ["x", "y"], [Interval(2, 3), Interval(2, 3)], silent=True,
                               called=True), True)
        self.assertEqual(
            check_interval_out([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=True,
                               called=True), False)

    def test_refine(self):
        print(colored("Refinement here", 'blue'))
        ## check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent, version)

        ## UNCOMMENT FOLLOWING to run this test
        # check_deeper([(0, 4)], ["x"], [Interval(0, 3)], 5, 0, 0.95, silent=False, version=5)
        # check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 5, 0, 0.95, silent=True, version=5)
        # check_deeper([(0, 0.5), (0, 0.5)], ["x+y"], [Interval(0, 1)], 5, 0, 0.95, silent=False, version=5)


        ## NORMAL TEST
        print(colored('Two-param test here', 'blue'))
        from load import create_intervals, get_f, load_pickled_data
        agents_quantities = [2, 3, 5, 10]
        f = get_f("./sem*[0-9].txt", "prism", True, agents_quantities)
        D3 = load_pickled_data("Data_two_param")

        coverage_thresh = 0.95
        alpha, n_samples, max_depth, min_rect_size, N, algorithm, v_p, v_q = 0.95, 100, 10, 1e-05, 2, 5, 0.028502714675268215, 0.03259111103419188

        space = check_deeper([(0, 1), (0, 0.9)], f[N],
                             create_intervals(alpha, n_samples, D3[("synchronous_", N, n_samples, v_p, v_q)]),
                             max_depth, min_rect_size, coverage_thresh, False, algorithm)
        print(colored('End of two-param test', 'blue'))

        print(colored('Multi-param test here', 'blue'))
        ## MULTIPARAM TEST
        f_multiparam = {10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        f_multiparam[10][0] = '(p - 1)**10'
        f_multiparam[10][1] = '10*p*(p - 1)**9*(q1 - 1)**9'
        f_multiparam[10][2] = '-5*p*(p - 1)**8*(2*p*q1**9 + 2*p*q1**8*q2 - 18*p*q1**8 + 2*p*q1**7*q2**2 - 18*p*q1**7*q2 + 72*p*q1**7 + 2*p*q1**6*q2**3 - 18*p*q1**6*q2**2 + 72*p*q1**6*q2 - 168*p*q1**6 + 2*p*q1**5*q2**4 - 18*p*q1**5*q2**3 + 72*p*q1**5*q2**2 - 168*p*q1**5*q2 + 252*p*q1**5 + 2*p*q1**4*q2**5 - 18*p*q1**4*q2**4 + 72*p*q1**4*q2**3 - 168*p*q1**4*q2**2 + 252*p*q1**4*q2 - 252*p*q1**4 + 2*p*q1**3*q2**6 - 18*p*q1**3*q2**5 + 72*p*q1**3*q2**4 - 168*p*q1**3*q2**3 + 252*p*q1**3*q2**2 - 252*p*q1**3*q2 + 168*p*q1**3 + 2*p*q1**2*q2**7 - 18*p*q1**2*q2**6 + 72*p*q1**2*q2**5 - 168*p*q1**2*q2**4 + 252*p*q1**2*q2**3 - 252*p*q1**2*q2**2 + 168*p*q1**2*q2 - 72*p*q1**2 + 2*p*q1*q2**8 - 18*p*q1*q2**7 + 72*p*q1*q2**6 - 168*p*q1*q2**5 + 252*p*q1*q2**4 - 252*p*q1*q2**3 + 168*p*q1*q2**2 - 72*p*q1*q2 + 18*p*q1 - 9*p*q2**8 + 72*p*q2**7 - 252*p*q2**6 + 504*p*q2**5 - 630*p*q2**4 + 504*p*q2**3 - 252*p*q2**2 + 72*p*q2 - 9*p - 2*q1**9 - 2*q1**8*q2 + 18*q1**8 - 2*q1**7*q2**2 + 18*q1**7*q2 - 72*q1**7 - 2*q1**6*q2**3 + 18*q1**6*q2**2 - 72*q1**6*q2 + 168*q1**6 - 2*q1**5*q2**4 + 18*q1**5*q2**3 - 72*q1**5*q2**2 + 168*q1**5*q2 - 252*q1**5 - 2*q1**4*q2**5 + 18*q1**4*q2**4 - 72*q1**4*q2**3 + 168*q1**4*q2**2 - 252*q1**4*q2 + 252*q1**4 - 2*q1**3*q2**6 + 18*q1**3*q2**5 - 72*q1**3*q2**4 + 168*q1**3*q2**3 - 252*q1**3*q2**2 + 252*q1**3*q2 - 168*q1**3 - 2*q1**2*q2**7 + 18*q1**2*q2**6 - 72*q1**2*q2**5 + 168*q1**2*q2**4 - 252*q1**2*q2**3 + 252*q1**2*q2**2 - 168*q1**2*q2 + 72*q1**2 - 2*q1*q2**8 + 18*q1*q2**7 - 72*q1*q2**6 + 168*q1*q2**5 - 252*q1*q2**4 + 252*q1*q2**3 - 168*q1*q2**2 + 72*q1*q2 - 18*q1)'
        f_multiparam[10][3] = '5*p*(p - 1)**7*(2*p**2*q1**8*q2 + 2*p**2*q1**7*q2**2 + 2*p**2*q1**7*q2*q3 - 18*p**2*q1**7*q2 + 2*p**2*q1**6*q2**3 + 2*p**2*q1**6*q2**2*q3 - 18*p**2*q1**6*q2**2 + 2*p**2*q1**6*q2*q3**2 - 18*p**2*q1**6*q2*q3 + 72*p**2*q1**6*q2 + 2*p**2*q1**5*q2**4 + 2*p**2*q1**5*q2**3*q3 - 18*p**2*q1**5*q2**3 + 2*p**2*q1**5*q2**2*q3**2 - 18*p**2*q1**5*q2**2*q3 + 72*p**2*q1**5*q2**2 + 2*p**2*q1**5*q2*q3**3 - 18*p**2*q1**5*q2*q3**2 + 72*p**2*q1**5*q2*q3 - 168*p**2*q1**5*q2 + 2*p**2*q1**4*q2**5 + 2*p**2*q1**4*q2**4*q3 - 18*p**2*q1**4*q2**4 + 2*p**2*q1**4*q2**3*q3**2 - 18*p**2*q1**4*q2**3*q3 + 72*p**2*q1**4*q2**3 + 2*p**2*q1**4*q2**2*q3**3 - 18*p**2*q1**4*q2**2*q3**2 + 72*p**2*q1**4*q2**2*q3 - 168*p**2*q1**4*q2**2 + 2*p**2*q1**4*q2*q3**4 - 18*p**2*q1**4*q2*q3**3 + 72*p**2*q1**4*q2*q3**2 - 168*p**2*q1**4*q2*q3 + 252*p**2*q1**4*q2 + 2*p**2*q1**3*q2**6 + 2*p**2*q1**3*q2**5*q3 - 18*p**2*q1**3*q2**5 + 2*p**2*q1**3*q2**4*q3**2 - 18*p**2*q1**3*q2**4*q3 + 72*p**2*q1**3*q2**4 + 2*p**2*q1**3*q2**3*q3**3 - 18*p**2*q1**3*q2**3*q3**2 + 72*p**2*q1**3*q2**3*q3 - 168*p**2*q1**3*q2**3 + 2*p**2*q1**3*q2**2*q3**4 - 18*p**2*q1**3*q2**2*q3**3 + 72*p**2*q1**3*q2**2*q3**2 - 168*p**2*q1**3*q2**2*q3 + 252*p**2*q1**3*q2**2 + 2*p**2*q1**3*q2*q3**5 - 18*p**2*q1**3*q2*q3**4 + 72*p**2*q1**3*q2*q3**3 - 168*p**2*q1**3*q2*q3**2 + 252*p**2*q1**3*q2*q3 - 252*p**2*q1**3*q2 + 2*p**2*q1**2*q2**7 + 2*p**2*q1**2*q2**6*q3 - 18*p**2*q1**2*q2**6 + 2*p**2*q1**2*q2**5*q3**2 - 18*p**2*q1**2*q2**5*q3 + 72*p**2*q1**2*q2**5 + 2*p**2*q1**2*q2**4*q3**3 - 18*p**2*q1**2*q2**4*q3**2 + 72*p**2*q1**2*q2**4*q3 - 168*p**2*q1**2*q2**4 + 2*p**2*q1**2*q2**3*q3**4 - 18*p**2*q1**2*q2**3*q3**3 + 72*p**2*q1**2*q2**3*q3**2 - 168*p**2*q1**2*q2**3*q3 + 252*p**2*q1**2*q2**3 + 2*p**2*q1**2*q2**2*q3**5 - 18*p**2*q1**2*q2**2*q3**4 + 72*p**2*q1**2*q2**2*q3**3 - 168*p**2*q1**2*q2**2*q3**2 + 252*p**2*q1**2*q2**2*q3 - 252*p**2*q1**2*q2**2 + 2*p**2*q1**2*q2*q3**6 - 18*p**2*q1**2*q2*q3**5 + 72*p**2*q1**2*q2*q3**4 - 168*p**2*q1**2*q2*q3**3 + 252*p**2*q1**2*q2*q3**2 - 252*p**2*q1**2*q2*q3 + 168*p**2*q1**2*q2 + 2*p**2*q1*q2**8 + 2*p**2*q1*q2**7*q3 - 18*p**2*q1*q2**7 + 2*p**2*q1*q2**6*q3**2 - 18*p**2*q1*q2**6*q3 + 72*p**2*q1*q2**6 + 2*p**2*q1*q2**5*q3**3 - 18*p**2*q1*q2**5*q3**2 + 72*p**2*q1*q2**5*q3 - 168*p**2*q1*q2**5 + 2*p**2*q1*q2**4*q3**4 - 18*p**2*q1*q2**4*q3**3 + 72*p**2*q1*q2**4*q3**2 - 168*p**2*q1*q2**4*q3 + 252*p**2*q1*q2**4 + 2*p**2*q1*q2**3*q3**5 - 18*p**2*q1*q2**3*q3**4 + 72*p**2*q1*q2**3*q3**3 - 168*p**2*q1*q2**3*q3**2 + 252*p**2*q1*q2**3*q3 - 252*p**2*q1*q2**3 + 2*p**2*q1*q2**2*q3**6 - 18*p**2*q1*q2**2*q3**5 + 72*p**2*q1*q2**2*q3**4 - 168*p**2*q1*q2**2*q3**3 + 252*p**2*q1*q2**2*q3**2 - 252*p**2*q1*q2**2*q3 + 168*p**2*q1*q2**2 + 2*p**2*q1*q2*q3**7 - 18*p**2*q1*q2*q3**6 + 72*p**2*q1*q2*q3**5 - 168*p**2*q1*q2*q3**4 + 252*p**2*q1*q2*q3**3 - 252*p**2*q1*q2*q3**2 + 168*p**2*q1*q2*q3 - 72*p**2*q1*q2 - 9*p**2*q2**8 - 9*p**2*q2**7*q3 + 72*p**2*q2**7 - 9*p**2*q2**6*q3**2 + 72*p**2*q2**6*q3 - 252*p**2*q2**6 - 9*p**2*q2**5*q3**3 + 72*p**2*q2**5*q3**2 - 252*p**2*q2**5*q3 + 504*p**2*q2**5 - 9*p**2*q2**4*q3**4 + 72*p**2*q2**4*q3**3 - 252*p**2*q2**4*q3**2 + 504*p**2*q2**4*q3 - 630*p**2*q2**4 - 9*p**2*q2**3*q3**5 + 72*p**2*q2**3*q3**4 - 252*p**2*q2**3*q3**3 + 504*p**2*q2**3*q3**2 - 630*p**2*q2**3*q3 + 504*p**2*q2**3 - 9*p**2*q2**2*q3**6 + 72*p**2*q2**2*q3**5 - 252*p**2*q2**2*q3**4 + 504*p**2*q2**2*q3**3 - 630*p**2*q2**2*q3**2 + 504*p**2*q2**2*q3 - 252*p**2*q2**2 - 9*p**2*q2*q3**7 + 72*p**2*q2*q3**6 - 252*p**2*q2*q3**5 + 504*p**2*q2*q3**4 - 630*p**2*q2*q3**3 + 504*p**2*q2*q3**2 - 252*p**2*q2*q3 + 72*p**2*q2 + 24*p**2*q3**7 - 168*p**2*q3**6 + 504*p**2*q3**5 - 840*p**2*q3**4 + 840*p**2*q3**3 - 504*p**2*q3**2 + 168*p**2*q3 - 24*p**2 - 4*p*q1**8*q2 - 4*p*q1**7*q2**2 - 4*p*q1**7*q2*q3 + 36*p*q1**7*q2 - 4*p*q1**6*q2**3 - 4*p*q1**6*q2**2*q3 + 36*p*q1**6*q2**2 - 4*p*q1**6*q2*q3**2 + 36*p*q1**6*q2*q3 - 144*p*q1**6*q2 - 4*p*q1**5*q2**4 - 4*p*q1**5*q2**3*q3 + 36*p*q1**5*q2**3 - 4*p*q1**5*q2**2*q3**2 + 36*p*q1**5*q2**2*q3 - 144*p*q1**5*q2**2 - 4*p*q1**5*q2*q3**3 + 36*p*q1**5*q2*q3**2 - 144*p*q1**5*q2*q3 + 336*p*q1**5*q2 - 4*p*q1**4*q2**5 - 4*p*q1**4*q2**4*q3 + 36*p*q1**4*q2**4 - 4*p*q1**4*q2**3*q3**2 + 36*p*q1**4*q2**3*q3 - 144*p*q1**4*q2**3 - 4*p*q1**4*q2**2*q3**3 + 36*p*q1**4*q2**2*q3**2 - 144*p*q1**4*q2**2*q3 + 336*p*q1**4*q2**2 - 4*p*q1**4*q2*q3**4 + 36*p*q1**4*q2*q3**3 - 144*p*q1**4*q2*q3**2 + 336*p*q1**4*q2*q3 - 504*p*q1**4*q2 - 4*p*q1**3*q2**6 - 4*p*q1**3*q2**5*q3 + 36*p*q1**3*q2**5 - 4*p*q1**3*q2**4*q3**2 + 36*p*q1**3*q2**4*q3 - 144*p*q1**3*q2**4 - 4*p*q1**3*q2**3*q3**3 + 36*p*q1**3*q2**3*q3**2 - 144*p*q1**3*q2**3*q3 + 336*p*q1**3*q2**3 - 4*p*q1**3*q2**2*q3**4 + 36*p*q1**3*q2**2*q3**3 - 144*p*q1**3*q2**2*q3**2 + 336*p*q1**3*q2**2*q3 - 504*p*q1**3*q2**2 - 4*p*q1**3*q2*q3**5 + 36*p*q1**3*q2*q3**4 - 144*p*q1**3*q2*q3**3 + 336*p*q1**3*q2*q3**2 - 504*p*q1**3*q2*q3 + 504*p*q1**3*q2 - 4*p*q1**2*q2**7 - 4*p*q1**2*q2**6*q3 + 36*p*q1**2*q2**6 - 4*p*q1**2*q2**5*q3**2 + 36*p*q1**2*q2**5*q3 - 144*p*q1**2*q2**5 - 4*p*q1**2*q2**4*q3**3 + 36*p*q1**2*q2**4*q3**2 - 144*p*q1**2*q2**4*q3 + 336*p*q1**2*q2**4 - 4*p*q1**2*q2**3*q3**4 + 36*p*q1**2*q2**3*q3**3 - 144*p*q1**2*q2**3*q3**2 + 336*p*q1**2*q2**3*q3 - 504*p*q1**2*q2**3 - 4*p*q1**2*q2**2*q3**5 + 36*p*q1**2*q2**2*q3**4 - 144*p*q1**2*q2**2*q3**3 + 336*p*q1**2*q2**2*q3**2 - 504*p*q1**2*q2**2*q3 + 504*p*q1**2*q2**2 - 4*p*q1**2*q2*q3**6 + 36*p*q1**2*q2*q3**5 - 144*p*q1**2*q2*q3**4 + 336*p*q1**2*q2*q3**3 - 504*p*q1**2*q2*q3**2 + 504*p*q1**2*q2*q3 - 336*p*q1**2*q2 - 4*p*q1*q2**8 - 4*p*q1*q2**7*q3 + 36*p*q1*q2**7 - 4*p*q1*q2**6*q3**2 + 36*p*q1*q2**6*q3 - 144*p*q1*q2**6 - 4*p*q1*q2**5*q3**3 + 36*p*q1*q2**5*q3**2 - 144*p*q1*q2**5*q3 + 336*p*q1*q2**5 - 4*p*q1*q2**4*q3**4 + 36*p*q1*q2**4*q3**3 - 144*p*q1*q2**4*q3**2 + 336*p*q1*q2**4*q3 - 504*p*q1*q2**4 - 4*p*q1*q2**3*q3**5 + 36*p*q1*q2**3*q3**4 - 144*p*q1*q2**3*q3**3 + 336*p*q1*q2**3*q3**2 - 504*p*q1*q2**3*q3 + 504*p*q1*q2**3 - 4*p*q1*q2**2*q3**6 + 36*p*q1*q2**2*q3**5 - 144*p*q1*q2**2*q3**4 + 336*p*q1*q2**2*q3**3 - 504*p*q1*q2**2*q3**2 + 504*p*q1*q2**2*q3 - 336*p*q1*q2**2 - 4*p*q1*q2*q3**7 + 36*p*q1*q2*q3**6 - 144*p*q1*q2*q3**5 + 336*p*q1*q2*q3**4 - 504*p*q1*q2*q3**3 + 504*p*q1*q2*q3**2 - 336*p*q1*q2*q3 + 144*p*q1*q2 + 9*p*q2**8 + 9*p*q2**7*q3 - 72*p*q2**7 + 9*p*q2**6*q3**2 - 72*p*q2**6*q3 + 252*p*q2**6 + 9*p*q2**5*q3**3 - 72*p*q2**5*q3**2 + 252*p*q2**5*q3 - 504*p*q2**5 + 9*p*q2**4*q3**4 - 72*p*q2**4*q3**3 + 252*p*q2**4*q3**2 - 504*p*q2**4*q3 + 630*p*q2**4 + 9*p*q2**3*q3**5 - 72*p*q2**3*q3**4 + 252*p*q2**3*q3**3 - 504*p*q2**3*q3**2 + 630*p*q2**3*q3 - 504*p*q2**3 + 9*p*q2**2*q3**6 - 72*p*q2**2*q3**5 + 252*p*q2**2*q3**4 - 504*p*q2**2*q3**3 + 630*p*q2**2*q3**2 - 504*p*q2**2*q3 + 252*p*q2**2 + 9*p*q2*q3**7 - 72*p*q2*q3**6 + 252*p*q2*q3**5 - 504*p*q2*q3**4 + 630*p*q2*q3**3 - 504*p*q2*q3**2 + 252*p*q2*q3 - 72*p*q2 + 2*q1**8*q2 + 2*q1**7*q2**2 + 2*q1**7*q2*q3 - 18*q1**7*q2 + 2*q1**6*q2**3 + 2*q1**6*q2**2*q3 - 18*q1**6*q2**2 + 2*q1**6*q2*q3**2 - 18*q1**6*q2*q3 + 72*q1**6*q2 + 2*q1**5*q2**4 + 2*q1**5*q2**3*q3 - 18*q1**5*q2**3 + 2*q1**5*q2**2*q3**2 - 18*q1**5*q2**2*q3 + 72*q1**5*q2**2 + 2*q1**5*q2*q3**3 - 18*q1**5*q2*q3**2 + 72*q1**5*q2*q3 - 168*q1**5*q2 + 2*q1**4*q2**5 + 2*q1**4*q2**4*q3 - 18*q1**4*q2**4 + 2*q1**4*q2**3*q3**2 - 18*q1**4*q2**3*q3 + 72*q1**4*q2**3 + 2*q1**4*q2**2*q3**3 - 18*q1**4*q2**2*q3**2 + 72*q1**4*q2**2*q3 - 168*q1**4*q2**2 + 2*q1**4*q2*q3**4 - 18*q1**4*q2*q3**3 + 72*q1**4*q2*q3**2 - 168*q1**4*q2*q3 + 252*q1**4*q2 + 2*q1**3*q2**6 + 2*q1**3*q2**5*q3 - 18*q1**3*q2**5 + 2*q1**3*q2**4*q3**2 - 18*q1**3*q2**4*q3 + 72*q1**3*q2**4 + 2*q1**3*q2**3*q3**3 - 18*q1**3*q2**3*q3**2 + 72*q1**3*q2**3*q3 - 168*q1**3*q2**3 + 2*q1**3*q2**2*q3**4 - 18*q1**3*q2**2*q3**3 + 72*q1**3*q2**2*q3**2 - 168*q1**3*q2**2*q3 + 252*q1**3*q2**2 + 2*q1**3*q2*q3**5 - 18*q1**3*q2*q3**4 + 72*q1**3*q2*q3**3 - 168*q1**3*q2*q3**2 + 252*q1**3*q2*q3 - 252*q1**3*q2 + 2*q1**2*q2**7 + 2*q1**2*q2**6*q3 - 18*q1**2*q2**6 + 2*q1**2*q2**5*q3**2 - 18*q1**2*q2**5*q3 + 72*q1**2*q2**5 + 2*q1**2*q2**4*q3**3 - 18*q1**2*q2**4*q3**2 + 72*q1**2*q2**4*q3 - 168*q1**2*q2**4 + 2*q1**2*q2**3*q3**4 - 18*q1**2*q2**3*q3**3 + 72*q1**2*q2**3*q3**2 - 168*q1**2*q2**3*q3 + 252*q1**2*q2**3 + 2*q1**2*q2**2*q3**5 - 18*q1**2*q2**2*q3**4 + 72*q1**2*q2**2*q3**3 - 168*q1**2*q2**2*q3**2 + 252*q1**2*q2**2*q3 - 252*q1**2*q2**2 + 2*q1**2*q2*q3**6 - 18*q1**2*q2*q3**5 + 72*q1**2*q2*q3**4 - 168*q1**2*q2*q3**3 + 252*q1**2*q2*q3**2 - 252*q1**2*q2*q3 + 168*q1**2*q2 + 2*q1*q2**8 + 2*q1*q2**7*q3 - 18*q1*q2**7 + 2*q1*q2**6*q3**2 - 18*q1*q2**6*q3 + 72*q1*q2**6 + 2*q1*q2**5*q3**3 - 18*q1*q2**5*q3**2 + 72*q1*q2**5*q3 - 168*q1*q2**5 + 2*q1*q2**4*q3**4 - 18*q1*q2**4*q3**3 + 72*q1*q2**4*q3**2 - 168*q1*q2**4*q3 + 252*q1*q2**4 + 2*q1*q2**3*q3**5 - 18*q1*q2**3*q3**4 + 72*q1*q2**3*q3**3 - 168*q1*q2**3*q3**2 + 252*q1*q2**3*q3 - 252*q1*q2**3 + 2*q1*q2**2*q3**6 - 18*q1*q2**2*q3**5 + 72*q1*q2**2*q3**4 - 168*q1*q2**2*q3**3 + 252*q1*q2**2*q3**2 - 252*q1*q2**2*q3 + 168*q1*q2**2 + 2*q1*q2*q3**7 - 18*q1*q2*q3**6 + 72*q1*q2*q3**5 - 168*q1*q2*q3**4 + 252*q1*q2*q3**3 - 252*q1*q2*q3**2 + 168*q1*q2*q3 - 72*q1*q2)'
        f_multiparam[10][4] = '-5*p*(p - 1)**6*(2*p**3*q1**7*q2*q3 + 2*p**3*q1**6*q2**2*q3 + 2*p**3*q1**6*q2*q3**2 + 2*p**3*q1**6*q2*q3*q4 - 18*p**3*q1**6*q2*q3 + 2*p**3*q1**5*q2**3*q3 + 2*p**3*q1**5*q2**2*q3**2 + 2*p**3*q1**5*q2**2*q3*q4 - 18*p**3*q1**5*q2**2*q3 + 2*p**3*q1**5*q2*q3**3 + 2*p**3*q1**5*q2*q3**2*q4 - 18*p**3*q1**5*q2*q3**2 + 2*p**3*q1**5*q2*q3*q4**2 - 18*p**3*q1**5*q2*q3*q4 + 72*p**3*q1**5*q2*q3 + 2*p**3*q1**4*q2**4*q3 + 2*p**3*q1**4*q2**3*q3**2 + 2*p**3*q1**4*q2**3*q3*q4 - 18*p**3*q1**4*q2**3*q3 + 2*p**3*q1**4*q2**2*q3**3 + 2*p**3*q1**4*q2**2*q3**2*q4 - 18*p**3*q1**4*q2**2*q3**2 + 2*p**3*q1**4*q2**2*q3*q4**2 - 18*p**3*q1**4*q2**2*q3*q4 + 72*p**3*q1**4*q2**2*q3 + 2*p**3*q1**4*q2*q3**4 + 2*p**3*q1**4*q2*q3**3*q4 - 18*p**3*q1**4*q2*q3**3 + 2*p**3*q1**4*q2*q3**2*q4**2 - 18*p**3*q1**4*q2*q3**2*q4 + 72*p**3*q1**4*q2*q3**2 + 2*p**3*q1**4*q2*q3*q4**3 - 18*p**3*q1**4*q2*q3*q4**2 + 72*p**3*q1**4*q2*q3*q4 - 168*p**3*q1**4*q2*q3 + 2*p**3*q1**3*q2**5*q3 + 2*p**3*q1**3*q2**4*q3**2 + 2*p**3*q1**3*q2**4*q3*q4 - 18*p**3*q1**3*q2**4*q3 + 2*p**3*q1**3*q2**3*q3**3 + 2*p**3*q1**3*q2**3*q3**2*q4 - 18*p**3*q1**3*q2**3*q3**2 + 2*p**3*q1**3*q2**3*q3*q4**2 - 18*p**3*q1**3*q2**3*q3*q4 + 72*p**3*q1**3*q2**3*q3 + 2*p**3*q1**3*q2**2*q3**4 + 2*p**3*q1**3*q2**2*q3**3*q4 - 18*p**3*q1**3*q2**2*q3**3 + 2*p**3*q1**3*q2**2*q3**2*q4**2 - 18*p**3*q1**3*q2**2*q3**2*q4 + 72*p**3*q1**3*q2**2*q3**2 + 2*p**3*q1**3*q2**2*q3*q4**3 - 18*p**3*q1**3*q2**2*q3*q4**2 + 72*p**3*q1**3*q2**2*q3*q4 - 168*p**3*q1**3*q2**2*q3 + 2*p**3*q1**3*q2*q3**5 + 2*p**3*q1**3*q2*q3**4*q4 - 18*p**3*q1**3*q2*q3**4 + 2*p**3*q1**3*q2*q3**3*q4**2 - 18*p**3*q1**3*q2*q3**3*q4 + 72*p**3*q1**3*q2*q3**3 + 2*p**3*q1**3*q2*q3**2*q4**3 - 18*p**3*q1**3*q2*q3**2*q4**2 + 72*p**3*q1**3*q2*q3**2*q4 - 168*p**3*q1**3*q2*q3**2 + 2*p**3*q1**3*q2*q3*q4**4 - 18*p**3*q1**3*q2*q3*q4**3 + 72*p**3*q1**3*q2*q3*q4**2 - 168*p**3*q1**3*q2*q3*q4 + 252*p**3*q1**3*q2*q3 + 2*p**3*q1**2*q2**6*q3 + 2*p**3*q1**2*q2**5*q3**2 + 2*p**3*q1**2*q2**5*q3*q4 - 18*p**3*q1**2*q2**5*q3 + 2*p**3*q1**2*q2**4*q3**3 + 2*p**3*q1**2*q2**4*q3**2*q4 - 18*p**3*q1**2*q2**4*q3**2 + 2*p**3*q1**2*q2**4*q3*q4**2 - 18*p**3*q1**2*q2**4*q3*q4 + 72*p**3*q1**2*q2**4*q3 + 2*p**3*q1**2*q2**3*q3**4 + 2*p**3*q1**2*q2**3*q3**3*q4 - 18*p**3*q1**2*q2**3*q3**3 + 2*p**3*q1**2*q2**3*q3**2*q4**2 - 18*p**3*q1**2*q2**3*q3**2*q4 + 72*p**3*q1**2*q2**3*q3**2 + 2*p**3*q1**2*q2**3*q3*q4**3 - 18*p**3*q1**2*q2**3*q3*q4**2 + 72*p**3*q1**2*q2**3*q3*q4 - 168*p**3*q1**2*q2**3*q3 + 2*p**3*q1**2*q2**2*q3**5 + 2*p**3*q1**2*q2**2*q3**4*q4 - 18*p**3*q1**2*q2**2*q3**4 + 2*p**3*q1**2*q2**2*q3**3*q4**2 - 18*p**3*q1**2*q2**2*q3**3*q4 + 72*p**3*q1**2*q2**2*q3**3 + 2*p**3*q1**2*q2**2*q3**2*q4**3 - 18*p**3*q1**2*q2**2*q3**2*q4**2 + 72*p**3*q1**2*q2**2*q3**2*q4 - 168*p**3*q1**2*q2**2*q3**2 + 2*p**3*q1**2*q2**2*q3*q4**4 - 18*p**3*q1**2*q2**2*q3*q4**3 + 72*p**3*q1**2*q2**2*q3*q4**2 - 168*p**3*q1**2*q2**2*q3*q4 + 252*p**3*q1**2*q2**2*q3 + 2*p**3*q1**2*q2*q3**6 + 2*p**3*q1**2*q2*q3**5*q4 - 18*p**3*q1**2*q2*q3**5 + 2*p**3*q1**2*q2*q3**4*q4**2 - 18*p**3*q1**2*q2*q3**4*q4 + 72*p**3*q1**2*q2*q3**4 + 2*p**3*q1**2*q2*q3**3*q4**3 - 18*p**3*q1**2*q2*q3**3*q4**2 + 72*p**3*q1**2*q2*q3**3*q4 - 168*p**3*q1**2*q2*q3**3 + 2*p**3*q1**2*q2*q3**2*q4**4 - 18*p**3*q1**2*q2*q3**2*q4**3 + 72*p**3*q1**2*q2*q3**2*q4**2 - 168*p**3*q1**2*q2*q3**2*q4 + 252*p**3*q1**2*q2*q3**2 + 2*p**3*q1**2*q2*q3*q4**5 - 18*p**3*q1**2*q2*q3*q4**4 + 72*p**3*q1**2*q2*q3*q4**3 - 168*p**3*q1**2*q2*q3*q4**2 + 252*p**3*q1**2*q2*q3*q4 - 252*p**3*q1**2*q2*q3 + 2*p**3*q1*q2**7*q3 + 2*p**3*q1*q2**6*q3**2 + 2*p**3*q1*q2**6*q3*q4 - 18*p**3*q1*q2**6*q3 + 2*p**3*q1*q2**5*q3**3 + 2*p**3*q1*q2**5*q3**2*q4 - 18*p**3*q1*q2**5*q3**2 + 2*p**3*q1*q2**5*q3*q4**2 - 18*p**3*q1*q2**5*q3*q4 + 72*p**3*q1*q2**5*q3 + 2*p**3*q1*q2**4*q3**4 + 2*p**3*q1*q2**4*q3**3*q4 - 18*p**3*q1*q2**4*q3**3 + 2*p**3*q1*q2**4*q3**2*q4**2 - 18*p**3*q1*q2**4*q3**2*q4 + 72*p**3*q1*q2**4*q3**2 + 2*p**3*q1*q2**4*q3*q4**3 - 18*p**3*q1*q2**4*q3*q4**2 + 72*p**3*q1*q2**4*q3*q4 - 168*p**3*q1*q2**4*q3 + 2*p**3*q1*q2**3*q3**5 + 2*p**3*q1*q2**3*q3**4*q4 - 18*p**3*q1*q2**3*q3**4 + 2*p**3*q1*q2**3*q3**3*q4**2 - 18*p**3*q1*q2**3*q3**3*q4 + 72*p**3*q1*q2**3*q3**3 + 2*p**3*q1*q2**3*q3**2*q4**3 - 18*p**3*q1*q2**3*q3**2*q4**2 + 72*p**3*q1*q2**3*q3**2*q4 - 168*p**3*q1*q2**3*q3**2 + 2*p**3*q1*q2**3*q3*q4**4 - 18*p**3*q1*q2**3*q3*q4**3 + 72*p**3*q1*q2**3*q3*q4**2 - 168*p**3*q1*q2**3*q3*q4 + 252*p**3*q1*q2**3*q3 + 2*p**3*q1*q2**2*q3**6 + 2*p**3*q1*q2**2*q3**5*q4 - 18*p**3*q1*q2**2*q3**5 + 2*p**3*q1*q2**2*q3**4*q4**2 - 18*p**3*q1*q2**2*q3**4*q4 + 72*p**3*q1*q2**2*q3**4 + 2*p**3*q1*q2**2*q3**3*q4**3 - 18*p**3*q1*q2**2*q3**3*q4**2 + 72*p**3*q1*q2**2*q3**3*q4 - 168*p**3*q1*q2**2*q3**3 + 2*p**3*q1*q2**2*q3**2*q4**4 - 18*p**3*q1*q2**2*q3**2*q4**3 + 72*p**3*q1*q2**2*q3**2*q4**2 - 168*p**3*q1*q2**2*q3**2*q4 + 252*p**3*q1*q2**2*q3**2 + 2*p**3*q1*q2**2*q3*q4**5 - 18*p**3*q1*q2**2*q3*q4**4 + 72*p**3*q1*q2**2*q3*q4**3 - 168*p**3*q1*q2**2*q3*q4**2 + 252*p**3*q1*q2**2*q3*q4 - 252*p**3*q1*q2**2*q3 + 2*p**3*q1*q2*q3**7 + 2*p**3*q1*q2*q3**6*q4 - 18*p**3*q1*q2*q3**6 + 2*p**3*q1*q2*q3**5*q4**2 - 18*p**3*q1*q2*q3**5*q4 + 72*p**3*q1*q2*q3**5 + 2*p**3*q1*q2*q3**4*q4**3 - 18*p**3*q1*q2*q3**4*q4**2 + 72*p**3*q1*q2*q3**4*q4 - 168*p**3*q1*q2*q3**4 + 2*p**3*q1*q2*q3**3*q4**4 - 18*p**3*q1*q2*q3**3*q4**3 + 72*p**3*q1*q2*q3**3*q4**2 - 168*p**3*q1*q2*q3**3*q4 + 252*p**3*q1*q2*q3**3 + 2*p**3*q1*q2*q3**2*q4**5 - 18*p**3*q1*q2*q3**2*q4**4 + 72*p**3*q1*q2*q3**2*q4**3 - 168*p**3*q1*q2*q3**2*q4**2 + 252*p**3*q1*q2*q3**2*q4 - 252*p**3*q1*q2*q3**2 + 2*p**3*q1*q2*q3*q4**6 - 18*p**3*q1*q2*q3*q4**5 + 72*p**3*q1*q2*q3*q4**4 - 168*p**3*q1*q2*q3*q4**3 + 252*p**3*q1*q2*q3*q4**2 - 252*p**3*q1*q2*q3*q4 + 168*p**3*q1*q2*q3 - 9*p**3*q2**7*q3 - 9*p**3*q2**6*q3**2 - 9*p**3*q2**6*q3*q4 + 72*p**3*q2**6*q3 - 9*p**3*q2**5*q3**3 - 9*p**3*q2**5*q3**2*q4 + 72*p**3*q2**5*q3**2 - 9*p**3*q2**5*q3*q4**2 + 72*p**3*q2**5*q3*q4 - 252*p**3*q2**5*q3 - 9*p**3*q2**4*q3**4 - 9*p**3*q2**4*q3**3*q4 + 72*p**3*q2**4*q3**3 - 9*p**3*q2**4*q3**2*q4**2 + 72*p**3*q2**4*q3**2*q4 - 252*p**3*q2**4*q3**2 - 9*p**3*q2**4*q3*q4**3 + 72*p**3*q2**4*q3*q4**2 - 252*p**3*q2**4*q3*q4 + 504*p**3*q2**4*q3 - 9*p**3*q2**3*q3**5 - 9*p**3*q2**3*q3**4*q4 + 72*p**3*q2**3*q3**4 - 9*p**3*q2**3*q3**3*q4**2 + 72*p**3*q2**3*q3**3*q4 - 252*p**3*q2**3*q3**3 - 9*p**3*q2**3*q3**2*q4**3 + 72*p**3*q2**3*q3**2*q4**2 - 252*p**3*q2**3*q3**2*q4 + 504*p**3*q2**3*q3**2 - 9*p**3*q2**3*q3*q4**4 + 72*p**3*q2**3*q3*q4**3 - 252*p**3*q2**3*q3*q4**2 + 504*p**3*q2**3*q3*q4 - 630*p**3*q2**3*q3 - 9*p**3*q2**2*q3**6 - 9*p**3*q2**2*q3**5*q4 + 72*p**3*q2**2*q3**5 - 9*p**3*q2**2*q3**4*q4**2 + 72*p**3*q2**2*q3**4*q4 - 252*p**3*q2**2*q3**4 - 9*p**3*q2**2*q3**3*q4**3 + 72*p**3*q2**2*q3**3*q4**2 - 252*p**3*q2**2*q3**3*q4 + 504*p**3*q2**2*q3**3 - 9*p**3*q2**2*q3**2*q4**4 + 72*p**3*q2**2*q3**2*q4**3 - 252*p**3*q2**2*q3**2*q4**2 + 504*p**3*q2**2*q3**2*q4 - 630*p**3*q2**2*q3**2 - 9*p**3*q2**2*q3*q4**5 + 72*p**3*q2**2*q3*q4**4 - 252*p**3*q2**2*q3*q4**3 + 504*p**3*q2**2*q3*q4**2 - 630*p**3*q2**2*q3*q4 + 504*p**3*q2**2*q3 - 9*p**3*q2*q3**7 - 9*p**3*q2*q3**6*q4 + 72*p**3*q2*q3**6 - 9*p**3*q2*q3**5*q4**2 + 72*p**3*q2*q3**5*q4 - 252*p**3*q2*q3**5 - 9*p**3*q2*q3**4*q4**3 + 72*p**3*q2*q3**4*q4**2 - 252*p**3*q2*q3**4*q4 + 504*p**3*q2*q3**4 - 9*p**3*q2*q3**3*q4**4 + 72*p**3*q2*q3**3*q4**3 - 252*p**3*q2*q3**3*q4**2 + 504*p**3*q2*q3**3*q4 - 630*p**3*q2*q3**3 - 9*p**3*q2*q3**2*q4**5 + 72*p**3*q2*q3**2*q4**4 - 252*p**3*q2*q3**2*q4**3 + 504*p**3*q2*q3**2*q4**2 - 630*p**3*q2*q3**2*q4 + 504*p**3*q2*q3**2 - 9*p**3*q2*q3*q4**6 + 72*p**3*q2*q3*q4**5 - 252*p**3*q2*q3*q4**4 + 504*p**3*q2*q3*q4**3 - 630*p**3*q2*q3*q4**2 + 504*p**3*q2*q3*q4 - 252*p**3*q2*q3 + 24*p**3*q3**7 + 24*p**3*q3**6*q4 - 168*p**3*q3**6 + 24*p**3*q3**5*q4**2 - 168*p**3*q3**5*q4 + 504*p**3*q3**5 + 24*p**3*q3**4*q4**3 - 168*p**3*q3**4*q4**2 + 504*p**3*q3**4*q4 - 840*p**3*q3**4 + 24*p**3*q3**3*q4**4 - 168*p**3*q3**3*q4**3 + 504*p**3*q3**3*q4**2 - 840*p**3*q3**3*q4 + 840*p**3*q3**3 + 24*p**3*q3**2*q4**5 - 168*p**3*q3**2*q4**4 + 504*p**3*q3**2*q4**3 - 840*p**3*q3**2*q4**2 + 840*p**3*q3**2*q4 - 504*p**3*q3**2 + 24*p**3*q3*q4**6 - 168*p**3*q3*q4**5 + 504*p**3*q3*q4**4 - 840*p**3*q3*q4**3 + 840*p**3*q3*q4**2 - 504*p**3*q3*q4 + 168*p**3*q3 - 42*p**3*q4**6 + 252*p**3*q4**5 - 630*p**3*q4**4 + 840*p**3*q4**3 - 630*p**3*q4**2 + 252*p**3*q4 - 42*p**3 - 6*p**2*q1**7*q2*q3 - 6*p**2*q1**6*q2**2*q3 - 6*p**2*q1**6*q2*q3**2 - 6*p**2*q1**6*q2*q3*q4 + 54*p**2*q1**6*q2*q3 - 6*p**2*q1**5*q2**3*q3 - 6*p**2*q1**5*q2**2*q3**2 - 6*p**2*q1**5*q2**2*q3*q4 + 54*p**2*q1**5*q2**2*q3 - 6*p**2*q1**5*q2*q3**3 - 6*p**2*q1**5*q2*q3**2*q4 + 54*p**2*q1**5*q2*q3**2 - 6*p**2*q1**5*q2*q3*q4**2 + 54*p**2*q1**5*q2*q3*q4 - 216*p**2*q1**5*q2*q3 - 6*p**2*q1**4*q2**4*q3 - 6*p**2*q1**4*q2**3*q3**2 - 6*p**2*q1**4*q2**3*q3*q4 + 54*p**2*q1**4*q2**3*q3 - 6*p**2*q1**4*q2**2*q3**3 - 6*p**2*q1**4*q2**2*q3**2*q4 + 54*p**2*q1**4*q2**2*q3**2 - 6*p**2*q1**4*q2**2*q3*q4**2 + 54*p**2*q1**4*q2**2*q3*q4 - 216*p**2*q1**4*q2**2*q3 - 6*p**2*q1**4*q2*q3**4 - 6*p**2*q1**4*q2*q3**3*q4 + 54*p**2*q1**4*q2*q3**3 - 6*p**2*q1**4*q2*q3**2*q4**2 + 54*p**2*q1**4*q2*q3**2*q4 - 216*p**2*q1**4*q2*q3**2 - 6*p**2*q1**4*q2*q3*q4**3 + 54*p**2*q1**4*q2*q3*q4**2 - 216*p**2*q1**4*q2*q3*q4 + 504*p**2*q1**4*q2*q3 - 6*p**2*q1**3*q2**5*q3 - 6*p**2*q1**3*q2**4*q3**2 - 6*p**2*q1**3*q2**4*q3*q4 + 54*p**2*q1**3*q2**4*q3 - 6*p**2*q1**3*q2**3*q3**3 - 6*p**2*q1**3*q2**3*q3**2*q4 + 54*p**2*q1**3*q2**3*q3**2 - 6*p**2*q1**3*q2**3*q3*q4**2 + 54*p**2*q1**3*q2**3*q3*q4 - 216*p**2*q1**3*q2**3*q3 - 6*p**2*q1**3*q2**2*q3**4 - 6*p**2*q1**3*q2**2*q3**3*q4 + 54*p**2*q1**3*q2**2*q3**3 - 6*p**2*q1**3*q2**2*q3**2*q4**2 + 54*p**2*q1**3*q2**2*q3**2*q4 - 216*p**2*q1**3*q2**2*q3**2 - 6*p**2*q1**3*q2**2*q3*q4**3 + 54*p**2*q1**3*q2**2*q3*q4**2 - 216*p**2*q1**3*q2**2*q3*q4 + 504*p**2*q1**3*q2**2*q3 - 6*p**2*q1**3*q2*q3**5 - 6*p**2*q1**3*q2*q3**4*q4 + 54*p**2*q1**3*q2*q3**4 - 6*p**2*q1**3*q2*q3**3*q4**2 + 54*p**2*q1**3*q2*q3**3*q4 - 216*p**2*q1**3*q2*q3**3 - 6*p**2*q1**3*q2*q3**2*q4**3 + 54*p**2*q1**3*q2*q3**2*q4**2 - 216*p**2*q1**3*q2*q3**2*q4 + 504*p**2*q1**3*q2*q3**2 - 6*p**2*q1**3*q2*q3*q4**4 + 54*p**2*q1**3*q2*q3*q4**3 - 216*p**2*q1**3*q2*q3*q4**2 + 504*p**2*q1**3*q2*q3*q4 - 756*p**2*q1**3*q2*q3 - 6*p**2*q1**2*q2**6*q3 - 6*p**2*q1**2*q2**5*q3**2 - 6*p**2*q1**2*q2**5*q3*q4 + 54*p**2*q1**2*q2**5*q3 - 6*p**2*q1**2*q2**4*q3**3 - 6*p**2*q1**2*q2**4*q3**2*q4 + 54*p**2*q1**2*q2**4*q3**2 - 6*p**2*q1**2*q2**4*q3*q4**2 + 54*p**2*q1**2*q2**4*q3*q4 - 216*p**2*q1**2*q2**4*q3 - 6*p**2*q1**2*q2**3*q3**4 - 6*p**2*q1**2*q2**3*q3**3*q4 + 54*p**2*q1**2*q2**3*q3**3 - 6*p**2*q1**2*q2**3*q3**2*q4**2 + 54*p**2*q1**2*q2**3*q3**2*q4 - 216*p**2*q1**2*q2**3*q3**2 - 6*p**2*q1**2*q2**3*q3*q4**3 + 54*p**2*q1**2*q2**3*q3*q4**2 - 216*p**2*q1**2*q2**3*q3*q4 + 504*p**2*q1**2*q2**3*q3 - 6*p**2*q1**2*q2**2*q3**5 - 6*p**2*q1**2*q2**2*q3**4*q4 + 54*p**2*q1**2*q2**2*q3**4 - 6*p**2*q1**2*q2**2*q3**3*q4**2 + 54*p**2*q1**2*q2**2*q3**3*q4 - 216*p**2*q1**2*q2**2*q3**3 - 6*p**2*q1**2*q2**2*q3**2*q4**3 + 54*p**2*q1**2*q2**2*q3**2*q4**2 - 216*p**2*q1**2*q2**2*q3**2*q4 + 504*p**2*q1**2*q2**2*q3**2 - 6*p**2*q1**2*q2**2*q3*q4**4 + 54*p**2*q1**2*q2**2*q3*q4**3 - 216*p**2*q1**2*q2**2*q3*q4**2 + 504*p**2*q1**2*q2**2*q3*q4 - 756*p**2*q1**2*q2**2*q3 - 6*p**2*q1**2*q2*q3**6 - 6*p**2*q1**2*q2*q3**5*q4 + 54*p**2*q1**2*q2*q3**5 - 6*p**2*q1**2*q2*q3**4*q4**2 + 54*p**2*q1**2*q2*q3**4*q4 - 216*p**2*q1**2*q2*q3**4 - 6*p**2*q1**2*q2*q3**3*q4**3 + 54*p**2*q1**2*q2*q3**3*q4**2 - 216*p**2*q1**2*q2*q3**3*q4 + 504*p**2*q1**2*q2*q3**3 - 6*p**2*q1**2*q2*q3**2*q4**4 + 54*p**2*q1**2*q2*q3**2*q4**3 - 216*p**2*q1**2*q2*q3**2*q4**2 + 504*p**2*q1**2*q2*q3**2*q4 - 756*p**2*q1**2*q2*q3**2 - 6*p**2*q1**2*q2*q3*q4**5 + 54*p**2*q1**2*q2*q3*q4**4 - 216*p**2*q1**2*q2*q3*q4**3 + 504*p**2*q1**2*q2*q3*q4**2 - 756*p**2*q1**2*q2*q3*q4 + 756*p**2*q1**2*q2*q3 - 6*p**2*q1*q2**7*q3 - 6*p**2*q1*q2**6*q3**2 - 6*p**2*q1*q2**6*q3*q4 + 54*p**2*q1*q2**6*q3 - 6*p**2*q1*q2**5*q3**3 - 6*p**2*q1*q2**5*q3**2*q4 + 54*p**2*q1*q2**5*q3**2 - 6*p**2*q1*q2**5*q3*q4**2 + 54*p**2*q1*q2**5*q3*q4 - 216*p**2*q1*q2**5*q3 - 6*p**2*q1*q2**4*q3**4 - 6*p**2*q1*q2**4*q3**3*q4 + 54*p**2*q1*q2**4*q3**3 - 6*p**2*q1*q2**4*q3**2*q4**2 + 54*p**2*q1*q2**4*q3**2*q4 - 216*p**2*q1*q2**4*q3**2 - 6*p**2*q1*q2**4*q3*q4**3 + 54*p**2*q1*q2**4*q3*q4**2 - 216*p**2*q1*q2**4*q3*q4 + 504*p**2*q1*q2**4*q3 - 6*p**2*q1*q2**3*q3**5 - 6*p**2*q1*q2**3*q3**4*q4 + 54*p**2*q1*q2**3*q3**4 - 6*p**2*q1*q2**3*q3**3*q4**2 + 54*p**2*q1*q2**3*q3**3*q4 - 216*p**2*q1*q2**3*q3**3 - 6*p**2*q1*q2**3*q3**2*q4**3 + 54*p**2*q1*q2**3*q3**2*q4**2 - 216*p**2*q1*q2**3*q3**2*q4 + 504*p**2*q1*q2**3*q3**2 - 6*p**2*q1*q2**3*q3*q4**4 + 54*p**2*q1*q2**3*q3*q4**3 - 216*p**2*q1*q2**3*q3*q4**2 + 504*p**2*q1*q2**3*q3*q4 - 756*p**2*q1*q2**3*q3 - 6*p**2*q1*q2**2*q3**6 - 6*p**2*q1*q2**2*q3**5*q4 + 54*p**2*q1*q2**2*q3**5 - 6*p**2*q1*q2**2*q3**4*q4**2 + 54*p**2*q1*q2**2*q3**4*q4 - 216*p**2*q1*q2**2*q3**4 - 6*p**2*q1*q2**2*q3**3*q4**3 + 54*p**2*q1*q2**2*q3**3*q4**2 - 216*p**2*q1*q2**2*q3**3*q4 + 504*p**2*q1*q2**2*q3**3 - 6*p**2*q1*q2**2*q3**2*q4**4 + 54*p**2*q1*q2**2*q3**2*q4**3 - 216*p**2*q1*q2**2*q3**2*q4**2 + 504*p**2*q1*q2**2*q3**2*q4 - 756*p**2*q1*q2**2*q3**2 - 6*p**2*q1*q2**2*q3*q4**5 + 54*p**2*q1*q2**2*q3*q4**4 - 216*p**2*q1*q2**2*q3*q4**3 + 504*p**2*q1*q2**2*q3*q4**2 - 756*p**2*q1*q2**2*q3*q4 + 756*p**2*q1*q2**2*q3 - 6*p**2*q1*q2*q3**7 - 6*p**2*q1*q2*q3**6*q4 + 54*p**2*q1*q2*q3**6 - 6*p**2*q1*q2*q3**5*q4**2 + 54*p**2*q1*q2*q3**5*q4 - 216*p**2*q1*q2*q3**5 - 6*p**2*q1*q2*q3**4*q4**3 + 54*p**2*q1*q2*q3**4*q4**2 - 216*p**2*q1*q2*q3**4*q4 + 504*p**2*q1*q2*q3**4 - 6*p**2*q1*q2*q3**3*q4**4 + 54*p**2*q1*q2*q3**3*q4**3 - 216*p**2*q1*q2*q3**3*q4**2 + 504*p**2*q1*q2*q3**3*q4 - 756*p**2*q1*q2*q3**3 - 6*p**2*q1*q2*q3**2*q4**5 + 54*p**2*q1*q2*q3**2*q4**4 - 216*p**2*q1*q2*q3**2*q4**3 + 504*p**2*q1*q2*q3**2*q4**2 - 756*p**2*q1*q2*q3**2*q4 + 756*p**2*q1*q2*q3**2 - 6*p**2*q1*q2*q3*q4**6 + 54*p**2*q1*q2*q3*q4**5 - 216*p**2*q1*q2*q3*q4**4 + 504*p**2*q1*q2*q3*q4**3 - 756*p**2*q1*q2*q3*q4**2 + 756*p**2*q1*q2*q3*q4 - 504*p**2*q1*q2*q3 + 18*p**2*q2**7*q3 + 18*p**2*q2**6*q3**2 + 18*p**2*q2**6*q3*q4 - 144*p**2*q2**6*q3 + 18*p**2*q2**5*q3**3 + 18*p**2*q2**5*q3**2*q4 - 144*p**2*q2**5*q3**2 + 18*p**2*q2**5*q3*q4**2 - 144*p**2*q2**5*q3*q4 + 504*p**2*q2**5*q3 + 18*p**2*q2**4*q3**4 + 18*p**2*q2**4*q3**3*q4 - 144*p**2*q2**4*q3**3 + 18*p**2*q2**4*q3**2*q4**2 - 144*p**2*q2**4*q3**2*q4 + 504*p**2*q2**4*q3**2 + 18*p**2*q2**4*q3*q4**3 - 144*p**2*q2**4*q3*q4**2 + 504*p**2*q2**4*q3*q4 - 1008*p**2*q2**4*q3 + 18*p**2*q2**3*q3**5 + 18*p**2*q2**3*q3**4*q4 - 144*p**2*q2**3*q3**4 + 18*p**2*q2**3*q3**3*q4**2 - 144*p**2*q2**3*q3**3*q4 + 504*p**2*q2**3*q3**3 + 18*p**2*q2**3*q3**2*q4**3 - 144*p**2*q2**3*q3**2*q4**2 + 504*p**2*q2**3*q3**2*q4 - 1008*p**2*q2**3*q3**2 + 18*p**2*q2**3*q3*q4**4 - 144*p**2*q2**3*q3*q4**3 + 504*p**2*q2**3*q3*q4**2 - 1008*p**2*q2**3*q3*q4 + 1260*p**2*q2**3*q3 + 18*p**2*q2**2*q3**6 + 18*p**2*q2**2*q3**5*q4 - 144*p**2*q2**2*q3**5 + 18*p**2*q2**2*q3**4*q4**2 - 144*p**2*q2**2*q3**4*q4 + 504*p**2*q2**2*q3**4 + 18*p**2*q2**2*q3**3*q4**3 - 144*p**2*q2**2*q3**3*q4**2 + 504*p**2*q2**2*q3**3*q4 - 1008*p**2*q2**2*q3**3 + 18*p**2*q2**2*q3**2*q4**4 - 144*p**2*q2**2*q3**2*q4**3 + 504*p**2*q2**2*q3**2*q4**2 - 1008*p**2*q2**2*q3**2*q4 + 1260*p**2*q2**2*q3**2 + 18*p**2*q2**2*q3*q4**5 - 144*p**2*q2**2*q3*q4**4 + 504*p**2*q2**2*q3*q4**3 - 1008*p**2*q2**2*q3*q4**2 + 1260*p**2*q2**2*q3*q4 - 1008*p**2*q2**2*q3 + 18*p**2*q2*q3**7 + 18*p**2*q2*q3**6*q4 - 144*p**2*q2*q3**6 + 18*p**2*q2*q3**5*q4**2 - 144*p**2*q2*q3**5*q4 + 504*p**2*q2*q3**5 + 18*p**2*q2*q3**4*q4**3 - 144*p**2*q2*q3**4*q4**2 + 504*p**2*q2*q3**4*q4 - 1008*p**2*q2*q3**4 + 18*p**2*q2*q3**3*q4**4 - 144*p**2*q2*q3**3*q4**3 + 504*p**2*q2*q3**3*q4**2 - 1008*p**2*q2*q3**3*q4 + 1260*p**2*q2*q3**3 + 18*p**2*q2*q3**2*q4**5 - 144*p**2*q2*q3**2*q4**4 + 504*p**2*q2*q3**2*q4**3 - 1008*p**2*q2*q3**2*q4**2 + 1260*p**2*q2*q3**2*q4 - 1008*p**2*q2*q3**2 + 18*p**2*q2*q3*q4**6 - 144*p**2*q2*q3*q4**5 + 504*p**2*q2*q3*q4**4 - 1008*p**2*q2*q3*q4**3 + 1260*p**2*q2*q3*q4**2 - 1008*p**2*q2*q3*q4 + 504*p**2*q2*q3 - 24*p**2*q3**7 - 24*p**2*q3**6*q4 + 168*p**2*q3**6 - 24*p**2*q3**5*q4**2 + 168*p**2*q3**5*q4 - 504*p**2*q3**5 - 24*p**2*q3**4*q4**3 + 168*p**2*q3**4*q4**2 - 504*p**2*q3**4*q4 + 840*p**2*q3**4 - 24*p**2*q3**3*q4**4 + 168*p**2*q3**3*q4**3 - 504*p**2*q3**3*q4**2 + 840*p**2*q3**3*q4 - 840*p**2*q3**3 - 24*p**2*q3**2*q4**5 + 168*p**2*q3**2*q4**4 - 504*p**2*q3**2*q4**3 + 840*p**2*q3**2*q4**2 - 840*p**2*q3**2*q4 + 504*p**2*q3**2 - 24*p**2*q3*q4**6 + 168*p**2*q3*q4**5 - 504*p**2*q3*q4**4 + 840*p**2*q3*q4**3 - 840*p**2*q3*q4**2 + 504*p**2*q3*q4 - 168*p**2*q3 + 6*p*q1**7*q2*q3 + 6*p*q1**6*q2**2*q3 + 6*p*q1**6*q2*q3**2 + 6*p*q1**6*q2*q3*q4 - 54*p*q1**6*q2*q3 + 6*p*q1**5*q2**3*q3 + 6*p*q1**5*q2**2*q3**2 + 6*p*q1**5*q2**2*q3*q4 - 54*p*q1**5*q2**2*q3 + 6*p*q1**5*q2*q3**3 + 6*p*q1**5*q2*q3**2*q4 - 54*p*q1**5*q2*q3**2 + 6*p*q1**5*q2*q3*q4**2 - 54*p*q1**5*q2*q3*q4 + 216*p*q1**5*q2*q3 + 6*p*q1**4*q2**4*q3 + 6*p*q1**4*q2**3*q3**2 + 6*p*q1**4*q2**3*q3*q4 - 54*p*q1**4*q2**3*q3 + 6*p*q1**4*q2**2*q3**3 + 6*p*q1**4*q2**2*q3**2*q4 - 54*p*q1**4*q2**2*q3**2 + 6*p*q1**4*q2**2*q3*q4**2 - 54*p*q1**4*q2**2*q3*q4 + 216*p*q1**4*q2**2*q3 + 6*p*q1**4*q2*q3**4 + 6*p*q1**4*q2*q3**3*q4 - 54*p*q1**4*q2*q3**3 + 6*p*q1**4*q2*q3**2*q4**2 - 54*p*q1**4*q2*q3**2*q4 + 216*p*q1**4*q2*q3**2 + 6*p*q1**4*q2*q3*q4**3 - 54*p*q1**4*q2*q3*q4**2 + 216*p*q1**4*q2*q3*q4 - 504*p*q1**4*q2*q3 + 6*p*q1**3*q2**5*q3 + 6*p*q1**3*q2**4*q3**2 + 6*p*q1**3*q2**4*q3*q4 - 54*p*q1**3*q2**4*q3 + 6*p*q1**3*q2**3*q3**3 + 6*p*q1**3*q2**3*q3**2*q4 - 54*p*q1**3*q2**3*q3**2 + 6*p*q1**3*q2**3*q3*q4**2 - 54*p*q1**3*q2**3*q3*q4 + 216*p*q1**3*q2**3*q3 + 6*p*q1**3*q2**2*q3**4 + 6*p*q1**3*q2**2*q3**3*q4 - 54*p*q1**3*q2**2*q3**3 + 6*p*q1**3*q2**2*q3**2*q4**2 - 54*p*q1**3*q2**2*q3**2*q4 + 216*p*q1**3*q2**2*q3**2 + 6*p*q1**3*q2**2*q3*q4**3 - 54*p*q1**3*q2**2*q3*q4**2 + 216*p*q1**3*q2**2*q3*q4 - 504*p*q1**3*q2**2*q3 + 6*p*q1**3*q2*q3**5 + 6*p*q1**3*q2*q3**4*q4 - 54*p*q1**3*q2*q3**4 + 6*p*q1**3*q2*q3**3*q4**2 - 54*p*q1**3*q2*q3**3*q4 + 216*p*q1**3*q2*q3**3 + 6*p*q1**3*q2*q3**2*q4**3 - 54*p*q1**3*q2*q3**2*q4**2 + 216*p*q1**3*q2*q3**2*q4 - 504*p*q1**3*q2*q3**2 + 6*p*q1**3*q2*q3*q4**4 - 54*p*q1**3*q2*q3*q4**3 + 216*p*q1**3*q2*q3*q4**2 - 504*p*q1**3*q2*q3*q4 + 756*p*q1**3*q2*q3 + 6*p*q1**2*q2**6*q3 + 6*p*q1**2*q2**5*q3**2 + 6*p*q1**2*q2**5*q3*q4 - 54*p*q1**2*q2**5*q3 + 6*p*q1**2*q2**4*q3**3 + 6*p*q1**2*q2**4*q3**2*q4 - 54*p*q1**2*q2**4*q3**2 + 6*p*q1**2*q2**4*q3*q4**2 - 54*p*q1**2*q2**4*q3*q4 + 216*p*q1**2*q2**4*q3 + 6*p*q1**2*q2**3*q3**4 + 6*p*q1**2*q2**3*q3**3*q4 - 54*p*q1**2*q2**3*q3**3 + 6*p*q1**2*q2**3*q3**2*q4**2 - 54*p*q1**2*q2**3*q3**2*q4 + 216*p*q1**2*q2**3*q3**2 + 6*p*q1**2*q2**3*q3*q4**3 - 54*p*q1**2*q2**3*q3*q4**2 + 216*p*q1**2*q2**3*q3*q4 - 504*p*q1**2*q2**3*q3 + 6*p*q1**2*q2**2*q3**5 + 6*p*q1**2*q2**2*q3**4*q4 - 54*p*q1**2*q2**2*q3**4 + 6*p*q1**2*q2**2*q3**3*q4**2 - 54*p*q1**2*q2**2*q3**3*q4 + 216*p*q1**2*q2**2*q3**3 + 6*p*q1**2*q2**2*q3**2*q4**3 - 54*p*q1**2*q2**2*q3**2*q4**2 + 216*p*q1**2*q2**2*q3**2*q4 - 504*p*q1**2*q2**2*q3**2 + 6*p*q1**2*q2**2*q3*q4**4 - 54*p*q1**2*q2**2*q3*q4**3 + 216*p*q1**2*q2**2*q3*q4**2 - 504*p*q1**2*q2**2*q3*q4 + 756*p*q1**2*q2**2*q3 + 6*p*q1**2*q2*q3**6 + 6*p*q1**2*q2*q3**5*q4 - 54*p*q1**2*q2*q3**5 + 6*p*q1**2*q2*q3**4*q4**2 - 54*p*q1**2*q2*q3**4*q4 + 216*p*q1**2*q2*q3**4 + 6*p*q1**2*q2*q3**3*q4**3 - 54*p*q1**2*q2*q3**3*q4**2 + 216*p*q1**2*q2*q3**3*q4 - 504*p*q1**2*q2*q3**3 + 6*p*q1**2*q2*q3**2*q4**4 - 54*p*q1**2*q2*q3**2*q4**3 + 216*p*q1**2*q2*q3**2*q4**2 - 504*p*q1**2*q2*q3**2*q4 + 756*p*q1**2*q2*q3**2 + 6*p*q1**2*q2*q3*q4**5 - 54*p*q1**2*q2*q3*q4**4 + 216*p*q1**2*q2*q3*q4**3 - 504*p*q1**2*q2*q3*q4**2 + 756*p*q1**2*q2*q3*q4 - 756*p*q1**2*q2*q3 + 6*p*q1*q2**7*q3 + 6*p*q1*q2**6*q3**2 + 6*p*q1*q2**6*q3*q4 - 54*p*q1*q2**6*q3 + 6*p*q1*q2**5*q3**3 + 6*p*q1*q2**5*q3**2*q4 - 54*p*q1*q2**5*q3**2 + 6*p*q1*q2**5*q3*q4**2 - 54*p*q1*q2**5*q3*q4 + 216*p*q1*q2**5*q3 + 6*p*q1*q2**4*q3**4 + 6*p*q1*q2**4*q3**3*q4 - 54*p*q1*q2**4*q3**3 + 6*p*q1*q2**4*q3**2*q4**2 - 54*p*q1*q2**4*q3**2*q4 + 216*p*q1*q2**4*q3**2 + 6*p*q1*q2**4*q3*q4**3 - 54*p*q1*q2**4*q3*q4**2 + 216*p*q1*q2**4*q3*q4 - 504*p*q1*q2**4*q3 + 6*p*q1*q2**3*q3**5 + 6*p*q1*q2**3*q3**4*q4 - 54*p*q1*q2**3*q3**4 + 6*p*q1*q2**3*q3**3*q4**2 - 54*p*q1*q2**3*q3**3*q4 + 216*p*q1*q2**3*q3**3 + 6*p*q1*q2**3*q3**2*q4**3 - 54*p*q1*q2**3*q3**2*q4**2 + 216*p*q1*q2**3*q3**2*q4 - 504*p*q1*q2**3*q3**2 + 6*p*q1*q2**3*q3*q4**4 - 54*p*q1*q2**3*q3*q4**3 + 216*p*q1*q2**3*q3*q4**2 - 504*p*q1*q2**3*q3*q4 + 756*p*q1*q2**3*q3 + 6*p*q1*q2**2*q3**6 + 6*p*q1*q2**2*q3**5*q4 - 54*p*q1*q2**2*q3**5 + 6*p*q1*q2**2*q3**4*q4**2 - 54*p*q1*q2**2*q3**4*q4 + 216*p*q1*q2**2*q3**4 + 6*p*q1*q2**2*q3**3*q4**3 - 54*p*q1*q2**2*q3**3*q4**2 + 216*p*q1*q2**2*q3**3*q4 - 504*p*q1*q2**2*q3**3 + 6*p*q1*q2**2*q3**2*q4**4 - 54*p*q1*q2**2*q3**2*q4**3 + 216*p*q1*q2**2*q3**2*q4**2 - 504*p*q1*q2**2*q3**2*q4 + 756*p*q1*q2**2*q3**2 + 6*p*q1*q2**2*q3*q4**5 - 54*p*q1*q2**2*q3*q4**4 + 216*p*q1*q2**2*q3*q4**3 - 504*p*q1*q2**2*q3*q4**2 + 756*p*q1*q2**2*q3*q4 - 756*p*q1*q2**2*q3 + 6*p*q1*q2*q3**7 + 6*p*q1*q2*q3**6*q4 - 54*p*q1*q2*q3**6 + 6*p*q1*q2*q3**5*q4**2 - 54*p*q1*q2*q3**5*q4 + 216*p*q1*q2*q3**5 + 6*p*q1*q2*q3**4*q4**3 - 54*p*q1*q2*q3**4*q4**2 + 216*p*q1*q2*q3**4*q4 - 504*p*q1*q2*q3**4 + 6*p*q1*q2*q3**3*q4**4 - 54*p*q1*q2*q3**3*q4**3 + 216*p*q1*q2*q3**3*q4**2 - 504*p*q1*q2*q3**3*q4 + 756*p*q1*q2*q3**3 + 6*p*q1*q2*q3**2*q4**5 - 54*p*q1*q2*q3**2*q4**4 + 216*p*q1*q2*q3**2*q4**3 - 504*p*q1*q2*q3**2*q4**2 + 756*p*q1*q2*q3**2*q4 - 756*p*q1*q2*q3**2 + 6*p*q1*q2*q3*q4**6 - 54*p*q1*q2*q3*q4**5 + 216*p*q1*q2*q3*q4**4 - 504*p*q1*q2*q3*q4**3 + 756*p*q1*q2*q3*q4**2 - 756*p*q1*q2*q3*q4 + 504*p*q1*q2*q3 - 9*p*q2**7*q3 - 9*p*q2**6*q3**2 - 9*p*q2**6*q3*q4 + 72*p*q2**6*q3 - 9*p*q2**5*q3**3 - 9*p*q2**5*q3**2*q4 + 72*p*q2**5*q3**2 - 9*p*q2**5*q3*q4**2 + 72*p*q2**5*q3*q4 - 252*p*q2**5*q3 - 9*p*q2**4*q3**4 - 9*p*q2**4*q3**3*q4 + 72*p*q2**4*q3**3 - 9*p*q2**4*q3**2*q4**2 + 72*p*q2**4*q3**2*q4 - 252*p*q2**4*q3**2 - 9*p*q2**4*q3*q4**3 + 72*p*q2**4*q3*q4**2 - 252*p*q2**4*q3*q4 + 504*p*q2**4*q3 - 9*p*q2**3*q3**5 - 9*p*q2**3*q3**4*q4 + 72*p*q2**3*q3**4 - 9*p*q2**3*q3**3*q4**2 + 72*p*q2**3*q3**3*q4 - 252*p*q2**3*q3**3 - 9*p*q2**3*q3**2*q4**3 + 72*p*q2**3*q3**2*q4**2 - 252*p*q2**3*q3**2*q4 + 504*p*q2**3*q3**2 - 9*p*q2**3*q3*q4**4 + 72*p*q2**3*q3*q4**3 - 252*p*q2**3*q3*q4**2 + 504*p*q2**3*q3*q4 - 630*p*q2**3*q3 - 9*p*q2**2*q3**6 - 9*p*q2**2*q3**5*q4 + 72*p*q2**2*q3**5 - 9*p*q2**2*q3**4*q4**2 + 72*p*q2**2*q3**4*q4 - 252*p*q2**2*q3**4 - 9*p*q2**2*q3**3*q4**3 + 72*p*q2**2*q3**3*q4**2 - 252*p*q2**2*q3**3*q4 + 504*p*q2**2*q3**3 - 9*p*q2**2*q3**2*q4**4 + 72*p*q2**2*q3**2*q4**3 - 252*p*q2**2*q3**2*q4**2 + 504*p*q2**2*q3**2*q4 - 630*p*q2**2*q3**2 - 9*p*q2**2*q3*q4**5 + 72*p*q2**2*q3*q4**4 - 252*p*q2**2*q3*q4**3 + 504*p*q2**2*q3*q4**2 - 630*p*q2**2*q3*q4 + 504*p*q2**2*q3 - 9*p*q2*q3**7 - 9*p*q2*q3**6*q4 + 72*p*q2*q3**6 - 9*p*q2*q3**5*q4**2 + 72*p*q2*q3**5*q4 - 252*p*q2*q3**5 - 9*p*q2*q3**4*q4**3 + 72*p*q2*q3**4*q4**2 - 252*p*q2*q3**4*q4 + 504*p*q2*q3**4 - 9*p*q2*q3**3*q4**4 + 72*p*q2*q3**3*q4**3 - 252*p*q2*q3**3*q4**2 + 504*p*q2*q3**3*q4 - 630*p*q2*q3**3 - 9*p*q2*q3**2*q4**5 + 72*p*q2*q3**2*q4**4 - 252*p*q2*q3**2*q4**3 + 504*p*q2*q3**2*q4**2 - 630*p*q2*q3**2*q4 + 504*p*q2*q3**2 - 9*p*q2*q3*q4**6 + 72*p*q2*q3*q4**5 - 252*p*q2*q3*q4**4 + 504*p*q2*q3*q4**3 - 630*p*q2*q3*q4**2 + 504*p*q2*q3*q4 - 252*p*q2*q3 - 2*q1**7*q2*q3 - 2*q1**6*q2**2*q3 - 2*q1**6*q2*q3**2 - 2*q1**6*q2*q3*q4 + 18*q1**6*q2*q3 - 2*q1**5*q2**3*q3 - 2*q1**5*q2**2*q3**2 - 2*q1**5*q2**2*q3*q4 + 18*q1**5*q2**2*q3 - 2*q1**5*q2*q3**3 - 2*q1**5*q2*q3**2*q4 + 18*q1**5*q2*q3**2 - 2*q1**5*q2*q3*q4**2 + 18*q1**5*q2*q3*q4 - 72*q1**5*q2*q3 - 2*q1**4*q2**4*q3 - 2*q1**4*q2**3*q3**2 - 2*q1**4*q2**3*q3*q4 + 18*q1**4*q2**3*q3 - 2*q1**4*q2**2*q3**3 - 2*q1**4*q2**2*q3**2*q4 + 18*q1**4*q2**2*q3**2 - 2*q1**4*q2**2*q3*q4**2 + 18*q1**4*q2**2*q3*q4 - 72*q1**4*q2**2*q3 - 2*q1**4*q2*q3**4 - 2*q1**4*q2*q3**3*q4 + 18*q1**4*q2*q3**3 - 2*q1**4*q2*q3**2*q4**2 + 18*q1**4*q2*q3**2*q4 - 72*q1**4*q2*q3**2 - 2*q1**4*q2*q3*q4**3 + 18*q1**4*q2*q3*q4**2 - 72*q1**4*q2*q3*q4 + 168*q1**4*q2*q3 - 2*q1**3*q2**5*q3 - 2*q1**3*q2**4*q3**2 - 2*q1**3*q2**4*q3*q4 + 18*q1**3*q2**4*q3 - 2*q1**3*q2**3*q3**3 - 2*q1**3*q2**3*q3**2*q4 + 18*q1**3*q2**3*q3**2 - 2*q1**3*q2**3*q3*q4**2 + 18*q1**3*q2**3*q3*q4 - 72*q1**3*q2**3*q3 - 2*q1**3*q2**2*q3**4 - 2*q1**3*q2**2*q3**3*q4 + 18*q1**3*q2**2*q3**3 - 2*q1**3*q2**2*q3**2*q4**2 + 18*q1**3*q2**2*q3**2*q4 - 72*q1**3*q2**2*q3**2 - 2*q1**3*q2**2*q3*q4**3 + 18*q1**3*q2**2*q3*q4**2 - 72*q1**3*q2**2*q3*q4 + 168*q1**3*q2**2*q3 - 2*q1**3*q2*q3**5 - 2*q1**3*q2*q3**4*q4 + 18*q1**3*q2*q3**4 - 2*q1**3*q2*q3**3*q4**2 + 18*q1**3*q2*q3**3*q4 - 72*q1**3*q2*q3**3 - 2*q1**3*q2*q3**2*q4**3 + 18*q1**3*q2*q3**2*q4**2 - 72*q1**3*q2*q3**2*q4 + 168*q1**3*q2*q3**2 - 2*q1**3*q2*q3*q4**4 + 18*q1**3*q2*q3*q4**3 - 72*q1**3*q2*q3*q4**2 + 168*q1**3*q2*q3*q4 - 252*q1**3*q2*q3 - 2*q1**2*q2**6*q3 - 2*q1**2*q2**5*q3**2 - 2*q1**2*q2**5*q3*q4 + 18*q1**2*q2**5*q3 - 2*q1**2*q2**4*q3**3 - 2*q1**2*q2**4*q3**2*q4 + 18*q1**2*q2**4*q3**2 - 2*q1**2*q2**4*q3*q4**2 + 18*q1**2*q2**4*q3*q4 - 72*q1**2*q2**4*q3 - 2*q1**2*q2**3*q3**4 - 2*q1**2*q2**3*q3**3*q4 + 18*q1**2*q2**3*q3**3 - 2*q1**2*q2**3*q3**2*q4**2 + 18*q1**2*q2**3*q3**2*q4 - 72*q1**2*q2**3*q3**2 - 2*q1**2*q2**3*q3*q4**3 + 18*q1**2*q2**3*q3*q4**2 - 72*q1**2*q2**3*q3*q4 + 168*q1**2*q2**3*q3 - 2*q1**2*q2**2*q3**5 - 2*q1**2*q2**2*q3**4*q4 + 18*q1**2*q2**2*q3**4 - 2*q1**2*q2**2*q3**3*q4**2 + 18*q1**2*q2**2*q3**3*q4 - 72*q1**2*q2**2*q3**3 - 2*q1**2*q2**2*q3**2*q4**3 + 18*q1**2*q2**2*q3**2*q4**2 - 72*q1**2*q2**2*q3**2*q4 + 168*q1**2*q2**2*q3**2 - 2*q1**2*q2**2*q3*q4**4 + 18*q1**2*q2**2*q3*q4**3 - 72*q1**2*q2**2*q3*q4**2 + 168*q1**2*q2**2*q3*q4 - 252*q1**2*q2**2*q3 - 2*q1**2*q2*q3**6 - 2*q1**2*q2*q3**5*q4 + 18*q1**2*q2*q3**5 - 2*q1**2*q2*q3**4*q4**2 + 18*q1**2*q2*q3**4*q4 - 72*q1**2*q2*q3**4 - 2*q1**2*q2*q3**3*q4**3 + 18*q1**2*q2*q3**3*q4**2 - 72*q1**2*q2*q3**3*q4 + 168*q1**2*q2*q3**3 - 2*q1**2*q2*q3**2*q4**4 + 18*q1**2*q2*q3**2*q4**3 - 72*q1**2*q2*q3**2*q4**2 + 168*q1**2*q2*q3**2*q4 - 252*q1**2*q2*q3**2 - 2*q1**2*q2*q3*q4**5 + 18*q1**2*q2*q3*q4**4 - 72*q1**2*q2*q3*q4**3 + 168*q1**2*q2*q3*q4**2 - 252*q1**2*q2*q3*q4 + 252*q1**2*q2*q3 - 2*q1*q2**7*q3 - 2*q1*q2**6*q3**2 - 2*q1*q2**6*q3*q4 + 18*q1*q2**6*q3 - 2*q1*q2**5*q3**3 - 2*q1*q2**5*q3**2*q4 + 18*q1*q2**5*q3**2 - 2*q1*q2**5*q3*q4**2 + 18*q1*q2**5*q3*q4 - 72*q1*q2**5*q3 - 2*q1*q2**4*q3**4 - 2*q1*q2**4*q3**3*q4 + 18*q1*q2**4*q3**3 - 2*q1*q2**4*q3**2*q4**2 + 18*q1*q2**4*q3**2*q4 - 72*q1*q2**4*q3**2 - 2*q1*q2**4*q3*q4**3 + 18*q1*q2**4*q3*q4**2 - 72*q1*q2**4*q3*q4 + 168*q1*q2**4*q3 - 2*q1*q2**3*q3**5 - 2*q1*q2**3*q3**4*q4 + 18*q1*q2**3*q3**4 - 2*q1*q2**3*q3**3*q4**2 + 18*q1*q2**3*q3**3*q4 - 72*q1*q2**3*q3**3 - 2*q1*q2**3*q3**2*q4**3 + 18*q1*q2**3*q3**2*q4**2 - 72*q1*q2**3*q3**2*q4 + 168*q1*q2**3*q3**2 - 2*q1*q2**3*q3*q4**4 + 18*q1*q2**3*q3*q4**3 - 72*q1*q2**3*q3*q4**2 + 168*q1*q2**3*q3*q4 - 252*q1*q2**3*q3 - 2*q1*q2**2*q3**6 - 2*q1*q2**2*q3**5*q4 + 18*q1*q2**2*q3**5 - 2*q1*q2**2*q3**4*q4**2 + 18*q1*q2**2*q3**4*q4 - 72*q1*q2**2*q3**4 - 2*q1*q2**2*q3**3*q4**3 + 18*q1*q2**2*q3**3*q4**2 - 72*q1*q2**2*q3**3*q4 + 168*q1*q2**2*q3**3 - 2*q1*q2**2*q3**2*q4**4 + 18*q1*q2**2*q3**2*q4**3 - 72*q1*q2**2*q3**2*q4**2 + 168*q1*q2**2*q3**2*q4 - 252*q1*q2**2*q3**2 - 2*q1*q2**2*q3*q4**5 + 18*q1*q2**2*q3*q4**4 - 72*q1*q2**2*q3*q4**3 + 168*q1*q2**2*q3*q4**2 - 252*q1*q2**2*q3*q4 + 252*q1*q2**2*q3 - 2*q1*q2*q3**7 - 2*q1*q2*q3**6*q4 + 18*q1*q2*q3**6 - 2*q1*q2*q3**5*q4**2 + 18*q1*q2*q3**5*q4 - 72*q1*q2*q3**5 - 2*q1*q2*q3**4*q4**3 + 18*q1*q2*q3**4*q4**2 - 72*q1*q2*q3**4*q4 + 168*q1*q2*q3**4 - 2*q1*q2*q3**3*q4**4 + 18*q1*q2*q3**3*q4**3 - 72*q1*q2*q3**3*q4**2 + 168*q1*q2*q3**3*q4 - 252*q1*q2*q3**3 - 2*q1*q2*q3**2*q4**5 + 18*q1*q2*q3**2*q4**4 - 72*q1*q2*q3**2*q4**3 + 168*q1*q2*q3**2*q4**2 - 252*q1*q2*q3**2*q4 + 252*q1*q2*q3**2 - 2*q1*q2*q3*q4**6 + 18*q1*q2*q3*q4**5 - 72*q1*q2*q3*q4**4 + 168*q1*q2*q3*q4**3 - 252*q1*q2*q3*q4**2 + 252*q1*q2*q3*q4 - 168*q1*q2*q3)'
        f_multiparam[10][5] = 'p*(p - 1)**5*(10*p**4*q1**6*q2*q3*q4 + 10*p**4*q1**5*q2**2*q3*q4 + 10*p**4*q1**5*q2*q3**2*q4 + 10*p**4*q1**5*q2*q3*q4**2 + 10*p**4*q1**5*q2*q3*q4*q5 - 90*p**4*q1**5*q2*q3*q4 + 10*p**4*q1**4*q2**3*q3*q4 + 10*p**4*q1**4*q2**2*q3**2*q4 + 10*p**4*q1**4*q2**2*q3*q4**2 + 10*p**4*q1**4*q2**2*q3*q4*q5 - 90*p**4*q1**4*q2**2*q3*q4 + 10*p**4*q1**4*q2*q3**3*q4 + 10*p**4*q1**4*q2*q3**2*q4**2 + 10*p**4*q1**4*q2*q3**2*q4*q5 - 90*p**4*q1**4*q2*q3**2*q4 + 10*p**4*q1**4*q2*q3*q4**3 + 10*p**4*q1**4*q2*q3*q4**2*q5 - 90*p**4*q1**4*q2*q3*q4**2 + 10*p**4*q1**4*q2*q3*q4*q5**2 - 90*p**4*q1**4*q2*q3*q4*q5 + 360*p**4*q1**4*q2*q3*q4 + 10*p**4*q1**3*q2**4*q3*q4 + 10*p**4*q1**3*q2**3*q3**2*q4 + 10*p**4*q1**3*q2**3*q3*q4**2 + 10*p**4*q1**3*q2**3*q3*q4*q5 - 90*p**4*q1**3*q2**3*q3*q4 + 10*p**4*q1**3*q2**2*q3**3*q4 + 10*p**4*q1**3*q2**2*q3**2*q4**2 + 10*p**4*q1**3*q2**2*q3**2*q4*q5 - 90*p**4*q1**3*q2**2*q3**2*q4 + 10*p**4*q1**3*q2**2*q3*q4**3 + 10*p**4*q1**3*q2**2*q3*q4**2*q5 - 90*p**4*q1**3*q2**2*q3*q4**2 + 10*p**4*q1**3*q2**2*q3*q4*q5**2 - 90*p**4*q1**3*q2**2*q3*q4*q5 + 360*p**4*q1**3*q2**2*q3*q4 + 10*p**4*q1**3*q2*q3**4*q4 + 10*p**4*q1**3*q2*q3**3*q4**2 + 10*p**4*q1**3*q2*q3**3*q4*q5 - 90*p**4*q1**3*q2*q3**3*q4 + 10*p**4*q1**3*q2*q3**2*q4**3 + 10*p**4*q1**3*q2*q3**2*q4**2*q5 - 90*p**4*q1**3*q2*q3**2*q4**2 + 10*p**4*q1**3*q2*q3**2*q4*q5**2 - 90*p**4*q1**3*q2*q3**2*q4*q5 + 360*p**4*q1**3*q2*q3**2*q4 + 10*p**4*q1**3*q2*q3*q4**4 + 10*p**4*q1**3*q2*q3*q4**3*q5 - 90*p**4*q1**3*q2*q3*q4**3 + 10*p**4*q1**3*q2*q3*q4**2*q5**2 - 90*p**4*q1**3*q2*q3*q4**2*q5 + 360*p**4*q1**3*q2*q3*q4**2 + 10*p**4*q1**3*q2*q3*q4*q5**3 - 90*p**4*q1**3*q2*q3*q4*q5**2 + 360*p**4*q1**3*q2*q3*q4*q5 - 840*p**4*q1**3*q2*q3*q4 + 10*p**4*q1**2*q2**5*q3*q4 + 10*p**4*q1**2*q2**4*q3**2*q4 + 10*p**4*q1**2*q2**4*q3*q4**2 + 10*p**4*q1**2*q2**4*q3*q4*q5 - 90*p**4*q1**2*q2**4*q3*q4 + 10*p**4*q1**2*q2**3*q3**3*q4 + 10*p**4*q1**2*q2**3*q3**2*q4**2 + 10*p**4*q1**2*q2**3*q3**2*q4*q5 - 90*p**4*q1**2*q2**3*q3**2*q4 + 10*p**4*q1**2*q2**3*q3*q4**3 + 10*p**4*q1**2*q2**3*q3*q4**2*q5 - 90*p**4*q1**2*q2**3*q3*q4**2 + 10*p**4*q1**2*q2**3*q3*q4*q5**2 - 90*p**4*q1**2*q2**3*q3*q4*q5 + 360*p**4*q1**2*q2**3*q3*q4 + 10*p**4*q1**2*q2**2*q3**4*q4 + 10*p**4*q1**2*q2**2*q3**3*q4**2 + 10*p**4*q1**2*q2**2*q3**3*q4*q5 - 90*p**4*q1**2*q2**2*q3**3*q4 + 10*p**4*q1**2*q2**2*q3**2*q4**3 + 10*p**4*q1**2*q2**2*q3**2*q4**2*q5 - 90*p**4*q1**2*q2**2*q3**2*q4**2 + 10*p**4*q1**2*q2**2*q3**2*q4*q5**2 - 90*p**4*q1**2*q2**2*q3**2*q4*q5 + 360*p**4*q1**2*q2**2*q3**2*q4 + 10*p**4*q1**2*q2**2*q3*q4**4 + 10*p**4*q1**2*q2**2*q3*q4**3*q5 - 90*p**4*q1**2*q2**2*q3*q4**3 + 10*p**4*q1**2*q2**2*q3*q4**2*q5**2 - 90*p**4*q1**2*q2**2*q3*q4**2*q5 + 360*p**4*q1**2*q2**2*q3*q4**2 + 10*p**4*q1**2*q2**2*q3*q4*q5**3 - 90*p**4*q1**2*q2**2*q3*q4*q5**2 + 360*p**4*q1**2*q2**2*q3*q4*q5 - 840*p**4*q1**2*q2**2*q3*q4 + 10*p**4*q1**2*q2*q3**5*q4 + 10*p**4*q1**2*q2*q3**4*q4**2 + 10*p**4*q1**2*q2*q3**4*q4*q5 - 90*p**4*q1**2*q2*q3**4*q4 + 10*p**4*q1**2*q2*q3**3*q4**3 + 10*p**4*q1**2*q2*q3**3*q4**2*q5 - 90*p**4*q1**2*q2*q3**3*q4**2 + 10*p**4*q1**2*q2*q3**3*q4*q5**2 - 90*p**4*q1**2*q2*q3**3*q4*q5 + 360*p**4*q1**2*q2*q3**3*q4 + 10*p**4*q1**2*q2*q3**2*q4**4 + 10*p**4*q1**2*q2*q3**2*q4**3*q5 - 90*p**4*q1**2*q2*q3**2*q4**3 + 10*p**4*q1**2*q2*q3**2*q4**2*q5**2 - 90*p**4*q1**2*q2*q3**2*q4**2*q5 + 360*p**4*q1**2*q2*q3**2*q4**2 + 10*p**4*q1**2*q2*q3**2*q4*q5**3 - 90*p**4*q1**2*q2*q3**2*q4*q5**2 + 360*p**4*q1**2*q2*q3**2*q4*q5 - 840*p**4*q1**2*q2*q3**2*q4 + 10*p**4*q1**2*q2*q3*q4**5 + 10*p**4*q1**2*q2*q3*q4**4*q5 - 90*p**4*q1**2*q2*q3*q4**4 + 10*p**4*q1**2*q2*q3*q4**3*q5**2 - 90*p**4*q1**2*q2*q3*q4**3*q5 + 360*p**4*q1**2*q2*q3*q4**3 + 10*p**4*q1**2*q2*q3*q4**2*q5**3 - 90*p**4*q1**2*q2*q3*q4**2*q5**2 + 360*p**4*q1**2*q2*q3*q4**2*q5 - 840*p**4*q1**2*q2*q3*q4**2 + 10*p**4*q1**2*q2*q3*q4*q5**4 - 90*p**4*q1**2*q2*q3*q4*q5**3 + 360*p**4*q1**2*q2*q3*q4*q5**2 - 840*p**4*q1**2*q2*q3*q4*q5 + 1260*p**4*q1**2*q2*q3*q4 + 10*p**4*q1*q2**6*q3*q4 + 10*p**4*q1*q2**5*q3**2*q4 + 10*p**4*q1*q2**5*q3*q4**2 + 10*p**4*q1*q2**5*q3*q4*q5 - 90*p**4*q1*q2**5*q3*q4 + 10*p**4*q1*q2**4*q3**3*q4 + 10*p**4*q1*q2**4*q3**2*q4**2 + 10*p**4*q1*q2**4*q3**2*q4*q5 - 90*p**4*q1*q2**4*q3**2*q4 + 10*p**4*q1*q2**4*q3*q4**3 + 10*p**4*q1*q2**4*q3*q4**2*q5 - 90*p**4*q1*q2**4*q3*q4**2 + 10*p**4*q1*q2**4*q3*q4*q5**2 - 90*p**4*q1*q2**4*q3*q4*q5 + 360*p**4*q1*q2**4*q3*q4 + 10*p**4*q1*q2**3*q3**4*q4 + 10*p**4*q1*q2**3*q3**3*q4**2 + 10*p**4*q1*q2**3*q3**3*q4*q5 - 90*p**4*q1*q2**3*q3**3*q4 + 10*p**4*q1*q2**3*q3**2*q4**3 + 10*p**4*q1*q2**3*q3**2*q4**2*q5 - 90*p**4*q1*q2**3*q3**2*q4**2 + 10*p**4*q1*q2**3*q3**2*q4*q5**2 - 90*p**4*q1*q2**3*q3**2*q4*q5 + 360*p**4*q1*q2**3*q3**2*q4 + 10*p**4*q1*q2**3*q3*q4**4 + 10*p**4*q1*q2**3*q3*q4**3*q5 - 90*p**4*q1*q2**3*q3*q4**3 + 10*p**4*q1*q2**3*q3*q4**2*q5**2 - 90*p**4*q1*q2**3*q3*q4**2*q5 + 360*p**4*q1*q2**3*q3*q4**2 + 10*p**4*q1*q2**3*q3*q4*q5**3 - 90*p**4*q1*q2**3*q3*q4*q5**2 + 360*p**4*q1*q2**3*q3*q4*q5 - 840*p**4*q1*q2**3*q3*q4 + 10*p**4*q1*q2**2*q3**5*q4 + 10*p**4*q1*q2**2*q3**4*q4**2 + 10*p**4*q1*q2**2*q3**4*q4*q5 - 90*p**4*q1*q2**2*q3**4*q4 + 10*p**4*q1*q2**2*q3**3*q4**3 + 10*p**4*q1*q2**2*q3**3*q4**2*q5 - 90*p**4*q1*q2**2*q3**3*q4**2 + 10*p**4*q1*q2**2*q3**3*q4*q5**2 - 90*p**4*q1*q2**2*q3**3*q4*q5 + 360*p**4*q1*q2**2*q3**3*q4 + 10*p**4*q1*q2**2*q3**2*q4**4 + 10*p**4*q1*q2**2*q3**2*q4**3*q5 - 90*p**4*q1*q2**2*q3**2*q4**3 + 10*p**4*q1*q2**2*q3**2*q4**2*q5**2 - 90*p**4*q1*q2**2*q3**2*q4**2*q5 + 360*p**4*q1*q2**2*q3**2*q4**2 + 10*p**4*q1*q2**2*q3**2*q4*q5**3 - 90*p**4*q1*q2**2*q3**2*q4*q5**2 + 360*p**4*q1*q2**2*q3**2*q4*q5 - 840*p**4*q1*q2**2*q3**2*q4 + 10*p**4*q1*q2**2*q3*q4**5 + 10*p**4*q1*q2**2*q3*q4**4*q5 - 90*p**4*q1*q2**2*q3*q4**4 + 10*p**4*q1*q2**2*q3*q4**3*q5**2 - 90*p**4*q1*q2**2*q3*q4**3*q5 + 360*p**4*q1*q2**2*q3*q4**3 + 10*p**4*q1*q2**2*q3*q4**2*q5**3 - 90*p**4*q1*q2**2*q3*q4**2*q5**2 + 360*p**4*q1*q2**2*q3*q4**2*q5 - 840*p**4*q1*q2**2*q3*q4**2 + 10*p**4*q1*q2**2*q3*q4*q5**4 - 90*p**4*q1*q2**2*q3*q4*q5**3 + 360*p**4*q1*q2**2*q3*q4*q5**2 - 840*p**4*q1*q2**2*q3*q4*q5 + 1260*p**4*q1*q2**2*q3*q4 + 10*p**4*q1*q2*q3**6*q4 + 10*p**4*q1*q2*q3**5*q4**2 + 10*p**4*q1*q2*q3**5*q4*q5 - 90*p**4*q1*q2*q3**5*q4 + 10*p**4*q1*q2*q3**4*q4**3 + 10*p**4*q1*q2*q3**4*q4**2*q5 - 90*p**4*q1*q2*q3**4*q4**2 + 10*p**4*q1*q2*q3**4*q4*q5**2 - 90*p**4*q1*q2*q3**4*q4*q5 + 360*p**4*q1*q2*q3**4*q4 + 10*p**4*q1*q2*q3**3*q4**4 + 10*p**4*q1*q2*q3**3*q4**3*q5 - 90*p**4*q1*q2*q3**3*q4**3 + 10*p**4*q1*q2*q3**3*q4**2*q5**2 - 90*p**4*q1*q2*q3**3*q4**2*q5 + 360*p**4*q1*q2*q3**3*q4**2 + 10*p**4*q1*q2*q3**3*q4*q5**3 - 90*p**4*q1*q2*q3**3*q4*q5**2 + 360*p**4*q1*q2*q3**3*q4*q5 - 840*p**4*q1*q2*q3**3*q4 + 10*p**4*q1*q2*q3**2*q4**5 + 10*p**4*q1*q2*q3**2*q4**4*q5 - 90*p**4*q1*q2*q3**2*q4**4 + 10*p**4*q1*q2*q3**2*q4**3*q5**2 - 90*p**4*q1*q2*q3**2*q4**3*q5 + 360*p**4*q1*q2*q3**2*q4**3 + 10*p**4*q1*q2*q3**2*q4**2*q5**3 - 90*p**4*q1*q2*q3**2*q4**2*q5**2 + 360*p**4*q1*q2*q3**2*q4**2*q5 - 840*p**4*q1*q2*q3**2*q4**2 + 10*p**4*q1*q2*q3**2*q4*q5**4 - 90*p**4*q1*q2*q3**2*q4*q5**3 + 360*p**4*q1*q2*q3**2*q4*q5**2 - 840*p**4*q1*q2*q3**2*q4*q5 + 1260*p**4*q1*q2*q3**2*q4 + 10*p**4*q1*q2*q3*q4**6 + 10*p**4*q1*q2*q3*q4**5*q5 - 90*p**4*q1*q2*q3*q4**5 + 10*p**4*q1*q2*q3*q4**4*q5**2 - 90*p**4*q1*q2*q3*q4**4*q5 + 360*p**4*q1*q2*q3*q4**4 + 10*p**4*q1*q2*q3*q4**3*q5**3 - 90*p**4*q1*q2*q3*q4**3*q5**2 + 360*p**4*q1*q2*q3*q4**3*q5 - 840*p**4*q1*q2*q3*q4**3 + 10*p**4*q1*q2*q3*q4**2*q5**4 - 90*p**4*q1*q2*q3*q4**2*q5**3 + 360*p**4*q1*q2*q3*q4**2*q5**2 - 840*p**4*q1*q2*q3*q4**2*q5 + 1260*p**4*q1*q2*q3*q4**2 + 10*p**4*q1*q2*q3*q4*q5**5 - 90*p**4*q1*q2*q3*q4*q5**4 + 360*p**4*q1*q2*q3*q4*q5**3 - 840*p**4*q1*q2*q3*q4*q5**2 + 1260*p**4*q1*q2*q3*q4*q5 - 1260*p**4*q1*q2*q3*q4 - 45*p**4*q2**6*q3*q4 - 45*p**4*q2**5*q3**2*q4 - 45*p**4*q2**5*q3*q4**2 - 45*p**4*q2**5*q3*q4*q5 + 360*p**4*q2**5*q3*q4 - 45*p**4*q2**4*q3**3*q4 - 45*p**4*q2**4*q3**2*q4**2 - 45*p**4*q2**4*q3**2*q4*q5 + 360*p**4*q2**4*q3**2*q4 - 45*p**4*q2**4*q3*q4**3 - 45*p**4*q2**4*q3*q4**2*q5 + 360*p**4*q2**4*q3*q4**2 - 45*p**4*q2**4*q3*q4*q5**2 + 360*p**4*q2**4*q3*q4*q5 - 1260*p**4*q2**4*q3*q4 - 45*p**4*q2**3*q3**4*q4 - 45*p**4*q2**3*q3**3*q4**2 - 45*p**4*q2**3*q3**3*q4*q5 + 360*p**4*q2**3*q3**3*q4 - 45*p**4*q2**3*q3**2*q4**3 - 45*p**4*q2**3*q3**2*q4**2*q5 + 360*p**4*q2**3*q3**2*q4**2 - 45*p**4*q2**3*q3**2*q4*q5**2 + 360*p**4*q2**3*q3**2*q4*q5 - 1260*p**4*q2**3*q3**2*q4 - 45*p**4*q2**3*q3*q4**4 - 45*p**4*q2**3*q3*q4**3*q5 + 360*p**4*q2**3*q3*q4**3 - 45*p**4*q2**3*q3*q4**2*q5**2 + 360*p**4*q2**3*q3*q4**2*q5 - 1260*p**4*q2**3*q3*q4**2 - 45*p**4*q2**3*q3*q4*q5**3 + 360*p**4*q2**3*q3*q4*q5**2 - 1260*p**4*q2**3*q3*q4*q5 + 2520*p**4*q2**3*q3*q4 - 45*p**4*q2**2*q3**5*q4 - 45*p**4*q2**2*q3**4*q4**2 - 45*p**4*q2**2*q3**4*q4*q5 + 360*p**4*q2**2*q3**4*q4 - 45*p**4*q2**2*q3**3*q4**3 - 45*p**4*q2**2*q3**3*q4**2*q5 + 360*p**4*q2**2*q3**3*q4**2 - 45*p**4*q2**2*q3**3*q4*q5**2 + 360*p**4*q2**2*q3**3*q4*q5 - 1260*p**4*q2**2*q3**3*q4 - 45*p**4*q2**2*q3**2*q4**4 - 45*p**4*q2**2*q3**2*q4**3*q5 + 360*p**4*q2**2*q3**2*q4**3 - 45*p**4*q2**2*q3**2*q4**2*q5**2 + 360*p**4*q2**2*q3**2*q4**2*q5 - 1260*p**4*q2**2*q3**2*q4**2 - 45*p**4*q2**2*q3**2*q4*q5**3 + 360*p**4*q2**2*q3**2*q4*q5**2 - 1260*p**4*q2**2*q3**2*q4*q5 + 2520*p**4*q2**2*q3**2*q4 - 45*p**4*q2**2*q3*q4**5 - 45*p**4*q2**2*q3*q4**4*q5 + 360*p**4*q2**2*q3*q4**4 - 45*p**4*q2**2*q3*q4**3*q5**2 + 360*p**4*q2**2*q3*q4**3*q5 - 1260*p**4*q2**2*q3*q4**3 - 45*p**4*q2**2*q3*q4**2*q5**3 + 360*p**4*q2**2*q3*q4**2*q5**2 - 1260*p**4*q2**2*q3*q4**2*q5 + 2520*p**4*q2**2*q3*q4**2 - 45*p**4*q2**2*q3*q4*q5**4 + 360*p**4*q2**2*q3*q4*q5**3 - 1260*p**4*q2**2*q3*q4*q5**2 + 2520*p**4*q2**2*q3*q4*q5 - 3150*p**4*q2**2*q3*q4 - 45*p**4*q2*q3**6*q4 - 45*p**4*q2*q3**5*q4**2 - 45*p**4*q2*q3**5*q4*q5 + 360*p**4*q2*q3**5*q4 - 45*p**4*q2*q3**4*q4**3 - 45*p**4*q2*q3**4*q4**2*q5 + 360*p**4*q2*q3**4*q4**2 - 45*p**4*q2*q3**4*q4*q5**2 + 360*p**4*q2*q3**4*q4*q5 - 1260*p**4*q2*q3**4*q4 - 45*p**4*q2*q3**3*q4**4 - 45*p**4*q2*q3**3*q4**3*q5 + 360*p**4*q2*q3**3*q4**3 - 45*p**4*q2*q3**3*q4**2*q5**2 + 360*p**4*q2*q3**3*q4**2*q5 - 1260*p**4*q2*q3**3*q4**2 - 45*p**4*q2*q3**3*q4*q5**3 + 360*p**4*q2*q3**3*q4*q5**2 - 1260*p**4*q2*q3**3*q4*q5 + 2520*p**4*q2*q3**3*q4 - 45*p**4*q2*q3**2*q4**5 - 45*p**4*q2*q3**2*q4**4*q5 + 360*p**4*q2*q3**2*q4**4 - 45*p**4*q2*q3**2*q4**3*q5**2 + 360*p**4*q2*q3**2*q4**3*q5 - 1260*p**4*q2*q3**2*q4**3 - 45*p**4*q2*q3**2*q4**2*q5**3 + 360*p**4*q2*q3**2*q4**2*q5**2 - 1260*p**4*q2*q3**2*q4**2*q5 + 2520*p**4*q2*q3**2*q4**2 - 45*p**4*q2*q3**2*q4*q5**4 + 360*p**4*q2*q3**2*q4*q5**3 - 1260*p**4*q2*q3**2*q4*q5**2 + 2520*p**4*q2*q3**2*q4*q5 - 3150*p**4*q2*q3**2*q4 - 45*p**4*q2*q3*q4**6 - 45*p**4*q2*q3*q4**5*q5 + 360*p**4*q2*q3*q4**5 - 45*p**4*q2*q3*q4**4*q5**2 + 360*p**4*q2*q3*q4**4*q5 - 1260*p**4*q2*q3*q4**4 - 45*p**4*q2*q3*q4**3*q5**3 + 360*p**4*q2*q3*q4**3*q5**2 - 1260*p**4*q2*q3*q4**3*q5 + 2520*p**4*q2*q3*q4**3 - 45*p**4*q2*q3*q4**2*q5**4 + 360*p**4*q2*q3*q4**2*q5**3 - 1260*p**4*q2*q3*q4**2*q5**2 + 2520*p**4*q2*q3*q4**2*q5 - 3150*p**4*q2*q3*q4**2 - 45*p**4*q2*q3*q4*q5**5 + 360*p**4*q2*q3*q4*q5**4 - 1260*p**4*q2*q3*q4*q5**3 + 2520*p**4*q2*q3*q4*q5**2 - 3150*p**4*q2*q3*q4*q5 + 2520*p**4*q2*q3*q4 + 120*p**4*q3**6*q4 + 120*p**4*q3**5*q4**2 + 120*p**4*q3**5*q4*q5 - 840*p**4*q3**5*q4 + 120*p**4*q3**4*q4**3 + 120*p**4*q3**4*q4**2*q5 - 840*p**4*q3**4*q4**2 + 120*p**4*q3**4*q4*q5**2 - 840*p**4*q3**4*q4*q5 + 2520*p**4*q3**4*q4 + 120*p**4*q3**3*q4**4 + 120*p**4*q3**3*q4**3*q5 - 840*p**4*q3**3*q4**3 + 120*p**4*q3**3*q4**2*q5**2 - 840*p**4*q3**3*q4**2*q5 + 2520*p**4*q3**3*q4**2 + 120*p**4*q3**3*q4*q5**3 - 840*p**4*q3**3*q4*q5**2 + 2520*p**4*q3**3*q4*q5 - 4200*p**4*q3**3*q4 + 120*p**4*q3**2*q4**5 + 120*p**4*q3**2*q4**4*q5 - 840*p**4*q3**2*q4**4 + 120*p**4*q3**2*q4**3*q5**2 - 840*p**4*q3**2*q4**3*q5 + 2520*p**4*q3**2*q4**3 + 120*p**4*q3**2*q4**2*q5**3 - 840*p**4*q3**2*q4**2*q5**2 + 2520*p**4*q3**2*q4**2*q5 - 4200*p**4*q3**2*q4**2 + 120*p**4*q3**2*q4*q5**4 - 840*p**4*q3**2*q4*q5**3 + 2520*p**4*q3**2*q4*q5**2 - 4200*p**4*q3**2*q4*q5 + 4200*p**4*q3**2*q4 + 120*p**4*q3*q4**6 + 120*p**4*q3*q4**5*q5 - 840*p**4*q3*q4**5 + 120*p**4*q3*q4**4*q5**2 - 840*p**4*q3*q4**4*q5 + 2520*p**4*q3*q4**4 + 120*p**4*q3*q4**3*q5**3 - 840*p**4*q3*q4**3*q5**2 + 2520*p**4*q3*q4**3*q5 - 4200*p**4*q3*q4**3 + 120*p**4*q3*q4**2*q5**4 - 840*p**4*q3*q4**2*q5**3 + 2520*p**4*q3*q4**2*q5**2 - 4200*p**4*q3*q4**2*q5 + 4200*p**4*q3*q4**2 + 120*p**4*q3*q4*q5**5 - 840*p**4*q3*q4*q5**4 + 2520*p**4*q3*q4*q5**3 - 4200*p**4*q3*q4*q5**2 + 4200*p**4*q3*q4*q5 - 2520*p**4*q3*q4 - 210*p**4*q4**6 - 210*p**4*q4**5*q5 + 1260*p**4*q4**5 - 210*p**4*q4**4*q5**2 + 1260*p**4*q4**4*q5 - 3150*p**4*q4**4 - 210*p**4*q4**3*q5**3 + 1260*p**4*q4**3*q5**2 - 3150*p**4*q4**3*q5 + 4200*p**4*q4**3 - 210*p**4*q4**2*q5**4 + 1260*p**4*q4**2*q5**3 - 3150*p**4*q4**2*q5**2 + 4200*p**4*q4**2*q5 - 3150*p**4*q4**2 - 210*p**4*q4*q5**5 + 1260*p**4*q4*q5**4 - 3150*p**4*q4*q5**3 + 4200*p**4*q4*q5**2 - 3150*p**4*q4*q5 + 1260*p**4*q4 + 252*p**4*q5**5 - 1260*p**4*q5**4 + 2520*p**4*q5**3 - 2520*p**4*q5**2 + 1260*p**4*q5 - 252*p**4 - 40*p**3*q1**6*q2*q3*q4 - 40*p**3*q1**5*q2**2*q3*q4 - 40*p**3*q1**5*q2*q3**2*q4 - 40*p**3*q1**5*q2*q3*q4**2 - 40*p**3*q1**5*q2*q3*q4*q5 + 360*p**3*q1**5*q2*q3*q4 - 40*p**3*q1**4*q2**3*q3*q4 - 40*p**3*q1**4*q2**2*q3**2*q4 - 40*p**3*q1**4*q2**2*q3*q4**2 - 40*p**3*q1**4*q2**2*q3*q4*q5 + 360*p**3*q1**4*q2**2*q3*q4 - 40*p**3*q1**4*q2*q3**3*q4 - 40*p**3*q1**4*q2*q3**2*q4**2 - 40*p**3*q1**4*q2*q3**2*q4*q5 + 360*p**3*q1**4*q2*q3**2*q4 - 40*p**3*q1**4*q2*q3*q4**3 - 40*p**3*q1**4*q2*q3*q4**2*q5 + 360*p**3*q1**4*q2*q3*q4**2 - 40*p**3*q1**4*q2*q3*q4*q5**2 + 360*p**3*q1**4*q2*q3*q4*q5 - 1440*p**3*q1**4*q2*q3*q4 - 40*p**3*q1**3*q2**4*q3*q4 - 40*p**3*q1**3*q2**3*q3**2*q4 - 40*p**3*q1**3*q2**3*q3*q4**2 - 40*p**3*q1**3*q2**3*q3*q4*q5 + 360*p**3*q1**3*q2**3*q3*q4 - 40*p**3*q1**3*q2**2*q3**3*q4 - 40*p**3*q1**3*q2**2*q3**2*q4**2 - 40*p**3*q1**3*q2**2*q3**2*q4*q5 + 360*p**3*q1**3*q2**2*q3**2*q4 - 40*p**3*q1**3*q2**2*q3*q4**3 - 40*p**3*q1**3*q2**2*q3*q4**2*q5 + 360*p**3*q1**3*q2**2*q3*q4**2 - 40*p**3*q1**3*q2**2*q3*q4*q5**2 + 360*p**3*q1**3*q2**2*q3*q4*q5 - 1440*p**3*q1**3*q2**2*q3*q4 - 40*p**3*q1**3*q2*q3**4*q4 - 40*p**3*q1**3*q2*q3**3*q4**2 - 40*p**3*q1**3*q2*q3**3*q4*q5 + 360*p**3*q1**3*q2*q3**3*q4 - 40*p**3*q1**3*q2*q3**2*q4**3 - 40*p**3*q1**3*q2*q3**2*q4**2*q5 + 360*p**3*q1**3*q2*q3**2*q4**2 - 40*p**3*q1**3*q2*q3**2*q4*q5**2 + 360*p**3*q1**3*q2*q3**2*q4*q5 - 1440*p**3*q1**3*q2*q3**2*q4 - 40*p**3*q1**3*q2*q3*q4**4 - 40*p**3*q1**3*q2*q3*q4**3*q5 + 360*p**3*q1**3*q2*q3*q4**3 - 40*p**3*q1**3*q2*q3*q4**2*q5**2 + 360*p**3*q1**3*q2*q3*q4**2*q5 - 1440*p**3*q1**3*q2*q3*q4**2 - 40*p**3*q1**3*q2*q3*q4*q5**3 + 360*p**3*q1**3*q2*q3*q4*q5**2 - 1440*p**3*q1**3*q2*q3*q4*q5 + 3360*p**3*q1**3*q2*q3*q4 - 40*p**3*q1**2*q2**5*q3*q4 - 40*p**3*q1**2*q2**4*q3**2*q4 - 40*p**3*q1**2*q2**4*q3*q4**2 - 40*p**3*q1**2*q2**4*q3*q4*q5 + 360*p**3*q1**2*q2**4*q3*q4 - 40*p**3*q1**2*q2**3*q3**3*q4 - 40*p**3*q1**2*q2**3*q3**2*q4**2 - 40*p**3*q1**2*q2**3*q3**2*q4*q5 + 360*p**3*q1**2*q2**3*q3**2*q4 - 40*p**3*q1**2*q2**3*q3*q4**3 - 40*p**3*q1**2*q2**3*q3*q4**2*q5 + 360*p**3*q1**2*q2**3*q3*q4**2 - 40*p**3*q1**2*q2**3*q3*q4*q5**2 + 360*p**3*q1**2*q2**3*q3*q4*q5 - 1440*p**3*q1**2*q2**3*q3*q4 - 40*p**3*q1**2*q2**2*q3**4*q4 - 40*p**3*q1**2*q2**2*q3**3*q4**2 - 40*p**3*q1**2*q2**2*q3**3*q4*q5 + 360*p**3*q1**2*q2**2*q3**3*q4 - 40*p**3*q1**2*q2**2*q3**2*q4**3 - 40*p**3*q1**2*q2**2*q3**2*q4**2*q5 + 360*p**3*q1**2*q2**2*q3**2*q4**2 - 40*p**3*q1**2*q2**2*q3**2*q4*q5**2 + 360*p**3*q1**2*q2**2*q3**2*q4*q5 - 1440*p**3*q1**2*q2**2*q3**2*q4 - 40*p**3*q1**2*q2**2*q3*q4**4 - 40*p**3*q1**2*q2**2*q3*q4**3*q5 + 360*p**3*q1**2*q2**2*q3*q4**3 - 40*p**3*q1**2*q2**2*q3*q4**2*q5**2 + 360*p**3*q1**2*q2**2*q3*q4**2*q5 - 1440*p**3*q1**2*q2**2*q3*q4**2 - 40*p**3*q1**2*q2**2*q3*q4*q5**3 + 360*p**3*q1**2*q2**2*q3*q4*q5**2 - 1440*p**3*q1**2*q2**2*q3*q4*q5 + 3360*p**3*q1**2*q2**2*q3*q4 - 40*p**3*q1**2*q2*q3**5*q4 - 40*p**3*q1**2*q2*q3**4*q4**2 - 40*p**3*q1**2*q2*q3**4*q4*q5 + 360*p**3*q1**2*q2*q3**4*q4 - 40*p**3*q1**2*q2*q3**3*q4**3 - 40*p**3*q1**2*q2*q3**3*q4**2*q5 + 360*p**3*q1**2*q2*q3**3*q4**2 - 40*p**3*q1**2*q2*q3**3*q4*q5**2 + 360*p**3*q1**2*q2*q3**3*q4*q5 - 1440*p**3*q1**2*q2*q3**3*q4 - 40*p**3*q1**2*q2*q3**2*q4**4 - 40*p**3*q1**2*q2*q3**2*q4**3*q5 + 360*p**3*q1**2*q2*q3**2*q4**3 - 40*p**3*q1**2*q2*q3**2*q4**2*q5**2 + 360*p**3*q1**2*q2*q3**2*q4**2*q5 - 1440*p**3*q1**2*q2*q3**2*q4**2 - 40*p**3*q1**2*q2*q3**2*q4*q5**3 + 360*p**3*q1**2*q2*q3**2*q4*q5**2 - 1440*p**3*q1**2*q2*q3**2*q4*q5 + 3360*p**3*q1**2*q2*q3**2*q4 - 40*p**3*q1**2*q2*q3*q4**5 - 40*p**3*q1**2*q2*q3*q4**4*q5 + 360*p**3*q1**2*q2*q3*q4**4 - 40*p**3*q1**2*q2*q3*q4**3*q5**2 + 360*p**3*q1**2*q2*q3*q4**3*q5 - 1440*p**3*q1**2*q2*q3*q4**3 - 40*p**3*q1**2*q2*q3*q4**2*q5**3 + 360*p**3*q1**2*q2*q3*q4**2*q5**2 - 1440*p**3*q1**2*q2*q3*q4**2*q5 + 3360*p**3*q1**2*q2*q3*q4**2 - 40*p**3*q1**2*q2*q3*q4*q5**4 + 360*p**3*q1**2*q2*q3*q4*q5**3 - 1440*p**3*q1**2*q2*q3*q4*q5**2 + 3360*p**3*q1**2*q2*q3*q4*q5 - 5040*p**3*q1**2*q2*q3*q4 - 40*p**3*q1*q2**6*q3*q4 - 40*p**3*q1*q2**5*q3**2*q4 - 40*p**3*q1*q2**5*q3*q4**2 - 40*p**3*q1*q2**5*q3*q4*q5 + 360*p**3*q1*q2**5*q3*q4 - 40*p**3*q1*q2**4*q3**3*q4 - 40*p**3*q1*q2**4*q3**2*q4**2 - 40*p**3*q1*q2**4*q3**2*q4*q5 + 360*p**3*q1*q2**4*q3**2*q4 - 40*p**3*q1*q2**4*q3*q4**3 - 40*p**3*q1*q2**4*q3*q4**2*q5 + 360*p**3*q1*q2**4*q3*q4**2 - 40*p**3*q1*q2**4*q3*q4*q5**2 + 360*p**3*q1*q2**4*q3*q4*q5 - 1440*p**3*q1*q2**4*q3*q4 - 40*p**3*q1*q2**3*q3**4*q4 - 40*p**3*q1*q2**3*q3**3*q4**2 - 40*p**3*q1*q2**3*q3**3*q4*q5 + 360*p**3*q1*q2**3*q3**3*q4 - 40*p**3*q1*q2**3*q3**2*q4**3 - 40*p**3*q1*q2**3*q3**2*q4**2*q5 + 360*p**3*q1*q2**3*q3**2*q4**2 - 40*p**3*q1*q2**3*q3**2*q4*q5**2 + 360*p**3*q1*q2**3*q3**2*q4*q5 - 1440*p**3*q1*q2**3*q3**2*q4 - 40*p**3*q1*q2**3*q3*q4**4 - 40*p**3*q1*q2**3*q3*q4**3*q5 + 360*p**3*q1*q2**3*q3*q4**3 - 40*p**3*q1*q2**3*q3*q4**2*q5**2 + 360*p**3*q1*q2**3*q3*q4**2*q5 - 1440*p**3*q1*q2**3*q3*q4**2 - 40*p**3*q1*q2**3*q3*q4*q5**3 + 360*p**3*q1*q2**3*q3*q4*q5**2 - 1440*p**3*q1*q2**3*q3*q4*q5 + 3360*p**3*q1*q2**3*q3*q4 - 40*p**3*q1*q2**2*q3**5*q4 - 40*p**3*q1*q2**2*q3**4*q4**2 - 40*p**3*q1*q2**2*q3**4*q4*q5 + 360*p**3*q1*q2**2*q3**4*q4 - 40*p**3*q1*q2**2*q3**3*q4**3 - 40*p**3*q1*q2**2*q3**3*q4**2*q5 + 360*p**3*q1*q2**2*q3**3*q4**2 - 40*p**3*q1*q2**2*q3**3*q4*q5**2 + 360*p**3*q1*q2**2*q3**3*q4*q5 - 1440*p**3*q1*q2**2*q3**3*q4 - 40*p**3*q1*q2**2*q3**2*q4**4 - 40*p**3*q1*q2**2*q3**2*q4**3*q5 + 360*p**3*q1*q2**2*q3**2*q4**3 - 40*p**3*q1*q2**2*q3**2*q4**2*q5**2 + 360*p**3*q1*q2**2*q3**2*q4**2*q5 - 1440*p**3*q1*q2**2*q3**2*q4**2 - 40*p**3*q1*q2**2*q3**2*q4*q5**3 + 360*p**3*q1*q2**2*q3**2*q4*q5**2 - 1440*p**3*q1*q2**2*q3**2*q4*q5 + 3360*p**3*q1*q2**2*q3**2*q4 - 40*p**3*q1*q2**2*q3*q4**5 - 40*p**3*q1*q2**2*q3*q4**4*q5 + 360*p**3*q1*q2**2*q3*q4**4 - 40*p**3*q1*q2**2*q3*q4**3*q5**2 + 360*p**3*q1*q2**2*q3*q4**3*q5 - 1440*p**3*q1*q2**2*q3*q4**3 - 40*p**3*q1*q2**2*q3*q4**2*q5**3 + 360*p**3*q1*q2**2*q3*q4**2*q5**2 - 1440*p**3*q1*q2**2*q3*q4**2*q5 + 3360*p**3*q1*q2**2*q3*q4**2 - 40*p**3*q1*q2**2*q3*q4*q5**4 + 360*p**3*q1*q2**2*q3*q4*q5**3 - 1440*p**3*q1*q2**2*q3*q4*q5**2 + 3360*p**3*q1*q2**2*q3*q4*q5 - 5040*p**3*q1*q2**2*q3*q4 - 40*p**3*q1*q2*q3**6*q4 - 40*p**3*q1*q2*q3**5*q4**2 - 40*p**3*q1*q2*q3**5*q4*q5 + 360*p**3*q1*q2*q3**5*q4 - 40*p**3*q1*q2*q3**4*q4**3 - 40*p**3*q1*q2*q3**4*q4**2*q5 + 360*p**3*q1*q2*q3**4*q4**2 - 40*p**3*q1*q2*q3**4*q4*q5**2 + 360*p**3*q1*q2*q3**4*q4*q5 - 1440*p**3*q1*q2*q3**4*q4 - 40*p**3*q1*q2*q3**3*q4**4 - 40*p**3*q1*q2*q3**3*q4**3*q5 + 360*p**3*q1*q2*q3**3*q4**3 - 40*p**3*q1*q2*q3**3*q4**2*q5**2 + 360*p**3*q1*q2*q3**3*q4**2*q5 - 1440*p**3*q1*q2*q3**3*q4**2 - 40*p**3*q1*q2*q3**3*q4*q5**3 + 360*p**3*q1*q2*q3**3*q4*q5**2 - 1440*p**3*q1*q2*q3**3*q4*q5 + 3360*p**3*q1*q2*q3**3*q4 - 40*p**3*q1*q2*q3**2*q4**5 - 40*p**3*q1*q2*q3**2*q4**4*q5 + 360*p**3*q1*q2*q3**2*q4**4 - 40*p**3*q1*q2*q3**2*q4**3*q5**2 + 360*p**3*q1*q2*q3**2*q4**3*q5 - 1440*p**3*q1*q2*q3**2*q4**3 - 40*p**3*q1*q2*q3**2*q4**2*q5**3 + 360*p**3*q1*q2*q3**2*q4**2*q5**2 - 1440*p**3*q1*q2*q3**2*q4**2*q5 + 3360*p**3*q1*q2*q3**2*q4**2 - 40*p**3*q1*q2*q3**2*q4*q5**4 + 360*p**3*q1*q2*q3**2*q4*q5**3 - 1440*p**3*q1*q2*q3**2*q4*q5**2 + 3360*p**3*q1*q2*q3**2*q4*q5 - 5040*p**3*q1*q2*q3**2*q4 - 40*p**3*q1*q2*q3*q4**6 - 40*p**3*q1*q2*q3*q4**5*q5 + 360*p**3*q1*q2*q3*q4**5 - 40*p**3*q1*q2*q3*q4**4*q5**2 + 360*p**3*q1*q2*q3*q4**4*q5 - 1440*p**3*q1*q2*q3*q4**4 - 40*p**3*q1*q2*q3*q4**3*q5**3 + 360*p**3*q1*q2*q3*q4**3*q5**2 - 1440*p**3*q1*q2*q3*q4**3*q5 + 3360*p**3*q1*q2*q3*q4**3 - 40*p**3*q1*q2*q3*q4**2*q5**4 + 360*p**3*q1*q2*q3*q4**2*q5**3 - 1440*p**3*q1*q2*q3*q4**2*q5**2 + 3360*p**3*q1*q2*q3*q4**2*q5 - 5040*p**3*q1*q2*q3*q4**2 - 40*p**3*q1*q2*q3*q4*q5**5 + 360*p**3*q1*q2*q3*q4*q5**4 - 1440*p**3*q1*q2*q3*q4*q5**3 + 3360*p**3*q1*q2*q3*q4*q5**2 - 5040*p**3*q1*q2*q3*q4*q5 + 5040*p**3*q1*q2*q3*q4 + 135*p**3*q2**6*q3*q4 + 135*p**3*q2**5*q3**2*q4 + 135*p**3*q2**5*q3*q4**2 + 135*p**3*q2**5*q3*q4*q5 - 1080*p**3*q2**5*q3*q4 + 135*p**3*q2**4*q3**3*q4 + 135*p**3*q2**4*q3**2*q4**2 + 135*p**3*q2**4*q3**2*q4*q5 - 1080*p**3*q2**4*q3**2*q4 + 135*p**3*q2**4*q3*q4**3 + 135*p**3*q2**4*q3*q4**2*q5 - 1080*p**3*q2**4*q3*q4**2 + 135*p**3*q2**4*q3*q4*q5**2 - 1080*p**3*q2**4*q3*q4*q5 + 3780*p**3*q2**4*q3*q4 + 135*p**3*q2**3*q3**4*q4 + 135*p**3*q2**3*q3**3*q4**2 + 135*p**3*q2**3*q3**3*q4*q5 - 1080*p**3*q2**3*q3**3*q4 + 135*p**3*q2**3*q3**2*q4**3 + 135*p**3*q2**3*q3**2*q4**2*q5 - 1080*p**3*q2**3*q3**2*q4**2 + 135*p**3*q2**3*q3**2*q4*q5**2 - 1080*p**3*q2**3*q3**2*q4*q5 + 3780*p**3*q2**3*q3**2*q4 + 135*p**3*q2**3*q3*q4**4 + 135*p**3*q2**3*q3*q4**3*q5 - 1080*p**3*q2**3*q3*q4**3 + 135*p**3*q2**3*q3*q4**2*q5**2 - 1080*p**3*q2**3*q3*q4**2*q5 + 3780*p**3*q2**3*q3*q4**2 + 135*p**3*q2**3*q3*q4*q5**3 - 1080*p**3*q2**3*q3*q4*q5**2 + 3780*p**3*q2**3*q3*q4*q5 - 7560*p**3*q2**3*q3*q4 + 135*p**3*q2**2*q3**5*q4 + 135*p**3*q2**2*q3**4*q4**2 + 135*p**3*q2**2*q3**4*q4*q5 - 1080*p**3*q2**2*q3**4*q4 + 135*p**3*q2**2*q3**3*q4**3 + 135*p**3*q2**2*q3**3*q4**2*q5 - 1080*p**3*q2**2*q3**3*q4**2 + 135*p**3*q2**2*q3**3*q4*q5**2 - 1080*p**3*q2**2*q3**3*q4*q5 + 3780*p**3*q2**2*q3**3*q4 + 135*p**3*q2**2*q3**2*q4**4 + 135*p**3*q2**2*q3**2*q4**3*q5 - 1080*p**3*q2**2*q3**2*q4**3 + 135*p**3*q2**2*q3**2*q4**2*q5**2 - 1080*p**3*q2**2*q3**2*q4**2*q5 + 3780*p**3*q2**2*q3**2*q4**2 + 135*p**3*q2**2*q3**2*q4*q5**3 - 1080*p**3*q2**2*q3**2*q4*q5**2 + 3780*p**3*q2**2*q3**2*q4*q5 - 7560*p**3*q2**2*q3**2*q4 + 135*p**3*q2**2*q3*q4**5 + 135*p**3*q2**2*q3*q4**4*q5 - 1080*p**3*q2**2*q3*q4**4 + 135*p**3*q2**2*q3*q4**3*q5**2 - 1080*p**3*q2**2*q3*q4**3*q5 + 3780*p**3*q2**2*q3*q4**3 + 135*p**3*q2**2*q3*q4**2*q5**3 - 1080*p**3*q2**2*q3*q4**2*q5**2 + 3780*p**3*q2**2*q3*q4**2*q5 - 7560*p**3*q2**2*q3*q4**2 + 135*p**3*q2**2*q3*q4*q5**4 - 1080*p**3*q2**2*q3*q4*q5**3 + 3780*p**3*q2**2*q3*q4*q5**2 - 7560*p**3*q2**2*q3*q4*q5 + 9450*p**3*q2**2*q3*q4 + 135*p**3*q2*q3**6*q4 + 135*p**3*q2*q3**5*q4**2 + 135*p**3*q2*q3**5*q4*q5 - 1080*p**3*q2*q3**5*q4 + 135*p**3*q2*q3**4*q4**3 + 135*p**3*q2*q3**4*q4**2*q5 - 1080*p**3*q2*q3**4*q4**2 + 135*p**3*q2*q3**4*q4*q5**2 - 1080*p**3*q2*q3**4*q4*q5 + 3780*p**3*q2*q3**4*q4 + 135*p**3*q2*q3**3*q4**4 + 135*p**3*q2*q3**3*q4**3*q5 - 1080*p**3*q2*q3**3*q4**3 + 135*p**3*q2*q3**3*q4**2*q5**2 - 1080*p**3*q2*q3**3*q4**2*q5 + 3780*p**3*q2*q3**3*q4**2 + 135*p**3*q2*q3**3*q4*q5**3 - 1080*p**3*q2*q3**3*q4*q5**2 + 3780*p**3*q2*q3**3*q4*q5 - 7560*p**3*q2*q3**3*q4 + 135*p**3*q2*q3**2*q4**5 + 135*p**3*q2*q3**2*q4**4*q5 - 1080*p**3*q2*q3**2*q4**4 + 135*p**3*q2*q3**2*q4**3*q5**2 - 1080*p**3*q2*q3**2*q4**3*q5 + 3780*p**3*q2*q3**2*q4**3 + 135*p**3*q2*q3**2*q4**2*q5**3 - 1080*p**3*q2*q3**2*q4**2*q5**2 + 3780*p**3*q2*q3**2*q4**2*q5 - 7560*p**3*q2*q3**2*q4**2 + 135*p**3*q2*q3**2*q4*q5**4 - 1080*p**3*q2*q3**2*q4*q5**3 + 3780*p**3*q2*q3**2*q4*q5**2 - 7560*p**3*q2*q3**2*q4*q5 + 9450*p**3*q2*q3**2*q4 + 135*p**3*q2*q3*q4**6 + 135*p**3*q2*q3*q4**5*q5 - 1080*p**3*q2*q3*q4**5 + 135*p**3*q2*q3*q4**4*q5**2 - 1080*p**3*q2*q3*q4**4*q5 + 3780*p**3*q2*q3*q4**4 + 135*p**3*q2*q3*q4**3*q5**3 - 1080*p**3*q2*q3*q4**3*q5**2 + 3780*p**3*q2*q3*q4**3*q5 - 7560*p**3*q2*q3*q4**3 + 135*p**3*q2*q3*q4**2*q5**4 - 1080*p**3*q2*q3*q4**2*q5**3 + 3780*p**3*q2*q3*q4**2*q5**2 - 7560*p**3*q2*q3*q4**2*q5 + 9450*p**3*q2*q3*q4**2 + 135*p**3*q2*q3*q4*q5**5 - 1080*p**3*q2*q3*q4*q5**4 + 3780*p**3*q2*q3*q4*q5**3 - 7560*p**3*q2*q3*q4*q5**2 + 9450*p**3*q2*q3*q4*q5 - 7560*p**3*q2*q3*q4 - 240*p**3*q3**6*q4 - 240*p**3*q3**5*q4**2 - 240*p**3*q3**5*q4*q5 + 1680*p**3*q3**5*q4 - 240*p**3*q3**4*q4**3 - 240*p**3*q3**4*q4**2*q5 + 1680*p**3*q3**4*q4**2 - 240*p**3*q3**4*q4*q5**2 + 1680*p**3*q3**4*q4*q5 - 5040*p**3*q3**4*q4 - 240*p**3*q3**3*q4**4 - 240*p**3*q3**3*q4**3*q5 + 1680*p**3*q3**3*q4**3 - 240*p**3*q3**3*q4**2*q5**2 + 1680*p**3*q3**3*q4**2*q5 - 5040*p**3*q3**3*q4**2 - 240*p**3*q3**3*q4*q5**3 + 1680*p**3*q3**3*q4*q5**2 - 5040*p**3*q3**3*q4*q5 + 8400*p**3*q3**3*q4 - 240*p**3*q3**2*q4**5 - 240*p**3*q3**2*q4**4*q5 + 1680*p**3*q3**2*q4**4 - 240*p**3*q3**2*q4**3*q5**2 + 1680*p**3*q3**2*q4**3*q5 - 5040*p**3*q3**2*q4**3 - 240*p**3*q3**2*q4**2*q5**3 + 1680*p**3*q3**2*q4**2*q5**2 - 5040*p**3*q3**2*q4**2*q5 + 8400*p**3*q3**2*q4**2 - 240*p**3*q3**2*q4*q5**4 + 1680*p**3*q3**2*q4*q5**3 - 5040*p**3*q3**2*q4*q5**2 + 8400*p**3*q3**2*q4*q5 - 8400*p**3*q3**2*q4 - 240*p**3*q3*q4**6 - 240*p**3*q3*q4**5*q5 + 1680*p**3*q3*q4**5 - 240*p**3*q3*q4**4*q5**2 + 1680*p**3*q3*q4**4*q5 - 5040*p**3*q3*q4**4 - 240*p**3*q3*q4**3*q5**3 + 1680*p**3*q3*q4**3*q5**2 - 5040*p**3*q3*q4**3*q5 + 8400*p**3*q3*q4**3 - 240*p**3*q3*q4**2*q5**4 + 1680*p**3*q3*q4**2*q5**3 - 5040*p**3*q3*q4**2*q5**2 + 8400*p**3*q3*q4**2*q5 - 8400*p**3*q3*q4**2 - 240*p**3*q3*q4*q5**5 + 1680*p**3*q3*q4*q5**4 - 5040*p**3*q3*q4*q5**3 + 8400*p**3*q3*q4*q5**2 - 8400*p**3*q3*q4*q5 + 5040*p**3*q3*q4 + 210*p**3*q4**6 + 210*p**3*q4**5*q5 - 1260*p**3*q4**5 + 210*p**3*q4**4*q5**2 - 1260*p**3*q4**4*q5 + 3150*p**3*q4**4 + 210*p**3*q4**3*q5**3 - 1260*p**3*q4**3*q5**2 + 3150*p**3*q4**3*q5 - 4200*p**3*q4**3 + 210*p**3*q4**2*q5**4 - 1260*p**3*q4**2*q5**3 + 3150*p**3*q4**2*q5**2 - 4200*p**3*q4**2*q5 + 3150*p**3*q4**2 + 210*p**3*q4*q5**5 - 1260*p**3*q4*q5**4 + 3150*p**3*q4*q5**3 - 4200*p**3*q4*q5**2 + 3150*p**3*q4*q5 - 1260*p**3*q4 + 60*p**2*q1**6*q2*q3*q4 + 60*p**2*q1**5*q2**2*q3*q4 + 60*p**2*q1**5*q2*q3**2*q4 + 60*p**2*q1**5*q2*q3*q4**2 + 60*p**2*q1**5*q2*q3*q4*q5 - 540*p**2*q1**5*q2*q3*q4 + 60*p**2*q1**4*q2**3*q3*q4 + 60*p**2*q1**4*q2**2*q3**2*q4 + 60*p**2*q1**4*q2**2*q3*q4**2 + 60*p**2*q1**4*q2**2*q3*q4*q5 - 540*p**2*q1**4*q2**2*q3*q4 + 60*p**2*q1**4*q2*q3**3*q4 + 60*p**2*q1**4*q2*q3**2*q4**2 + 60*p**2*q1**4*q2*q3**2*q4*q5 - 540*p**2*q1**4*q2*q3**2*q4 + 60*p**2*q1**4*q2*q3*q4**3 + 60*p**2*q1**4*q2*q3*q4**2*q5 - 540*p**2*q1**4*q2*q3*q4**2 + 60*p**2*q1**4*q2*q3*q4*q5**2 - 540*p**2*q1**4*q2*q3*q4*q5 + 2160*p**2*q1**4*q2*q3*q4 + 60*p**2*q1**3*q2**4*q3*q4 + 60*p**2*q1**3*q2**3*q3**2*q4 + 60*p**2*q1**3*q2**3*q3*q4**2 + 60*p**2*q1**3*q2**3*q3*q4*q5 - 540*p**2*q1**3*q2**3*q3*q4 + 60*p**2*q1**3*q2**2*q3**3*q4 + 60*p**2*q1**3*q2**2*q3**2*q4**2 + 60*p**2*q1**3*q2**2*q3**2*q4*q5 - 540*p**2*q1**3*q2**2*q3**2*q4 + 60*p**2*q1**3*q2**2*q3*q4**3 + 60*p**2*q1**3*q2**2*q3*q4**2*q5 - 540*p**2*q1**3*q2**2*q3*q4**2 + 60*p**2*q1**3*q2**2*q3*q4*q5**2 - 540*p**2*q1**3*q2**2*q3*q4*q5 + 2160*p**2*q1**3*q2**2*q3*q4 + 60*p**2*q1**3*q2*q3**4*q4 + 60*p**2*q1**3*q2*q3**3*q4**2 + 60*p**2*q1**3*q2*q3**3*q4*q5 - 540*p**2*q1**3*q2*q3**3*q4 + 60*p**2*q1**3*q2*q3**2*q4**3 + 60*p**2*q1**3*q2*q3**2*q4**2*q5 - 540*p**2*q1**3*q2*q3**2*q4**2 + 60*p**2*q1**3*q2*q3**2*q4*q5**2 - 540*p**2*q1**3*q2*q3**2*q4*q5 + 2160*p**2*q1**3*q2*q3**2*q4 + 60*p**2*q1**3*q2*q3*q4**4 + 60*p**2*q1**3*q2*q3*q4**3*q5 - 540*p**2*q1**3*q2*q3*q4**3 + 60*p**2*q1**3*q2*q3*q4**2*q5**2 - 540*p**2*q1**3*q2*q3*q4**2*q5 + 2160*p**2*q1**3*q2*q3*q4**2 + 60*p**2*q1**3*q2*q3*q4*q5**3 - 540*p**2*q1**3*q2*q3*q4*q5**2 + 2160*p**2*q1**3*q2*q3*q4*q5 - 5040*p**2*q1**3*q2*q3*q4 + 60*p**2*q1**2*q2**5*q3*q4 + 60*p**2*q1**2*q2**4*q3**2*q4 + 60*p**2*q1**2*q2**4*q3*q4**2 + 60*p**2*q1**2*q2**4*q3*q4*q5 - 540*p**2*q1**2*q2**4*q3*q4 + 60*p**2*q1**2*q2**3*q3**3*q4 + 60*p**2*q1**2*q2**3*q3**2*q4**2 + 60*p**2*q1**2*q2**3*q3**2*q4*q5 - 540*p**2*q1**2*q2**3*q3**2*q4 + 60*p**2*q1**2*q2**3*q3*q4**3 + 60*p**2*q1**2*q2**3*q3*q4**2*q5 - 540*p**2*q1**2*q2**3*q3*q4**2 + 60*p**2*q1**2*q2**3*q3*q4*q5**2 - 540*p**2*q1**2*q2**3*q3*q4*q5 + 2160*p**2*q1**2*q2**3*q3*q4 + 60*p**2*q1**2*q2**2*q3**4*q4 + 60*p**2*q1**2*q2**2*q3**3*q4**2 + 60*p**2*q1**2*q2**2*q3**3*q4*q5 - 540*p**2*q1**2*q2**2*q3**3*q4 + 60*p**2*q1**2*q2**2*q3**2*q4**3 + 60*p**2*q1**2*q2**2*q3**2*q4**2*q5 - 540*p**2*q1**2*q2**2*q3**2*q4**2 + 60*p**2*q1**2*q2**2*q3**2*q4*q5**2 - 540*p**2*q1**2*q2**2*q3**2*q4*q5 + 2160*p**2*q1**2*q2**2*q3**2*q4 + 60*p**2*q1**2*q2**2*q3*q4**4 + 60*p**2*q1**2*q2**2*q3*q4**3*q5 - 540*p**2*q1**2*q2**2*q3*q4**3 + 60*p**2*q1**2*q2**2*q3*q4**2*q5**2 - 540*p**2*q1**2*q2**2*q3*q4**2*q5 + 2160*p**2*q1**2*q2**2*q3*q4**2 + 60*p**2*q1**2*q2**2*q3*q4*q5**3 - 540*p**2*q1**2*q2**2*q3*q4*q5**2 + 2160*p**2*q1**2*q2**2*q3*q4*q5 - 5040*p**2*q1**2*q2**2*q3*q4 + 60*p**2*q1**2*q2*q3**5*q4 + 60*p**2*q1**2*q2*q3**4*q4**2 + 60*p**2*q1**2*q2*q3**4*q4*q5 - 540*p**2*q1**2*q2*q3**4*q4 + 60*p**2*q1**2*q2*q3**3*q4**3 + 60*p**2*q1**2*q2*q3**3*q4**2*q5 - 540*p**2*q1**2*q2*q3**3*q4**2 + 60*p**2*q1**2*q2*q3**3*q4*q5**2 - 540*p**2*q1**2*q2*q3**3*q4*q5 + 2160*p**2*q1**2*q2*q3**3*q4 + 60*p**2*q1**2*q2*q3**2*q4**4 + 60*p**2*q1**2*q2*q3**2*q4**3*q5 - 540*p**2*q1**2*q2*q3**2*q4**3 + 60*p**2*q1**2*q2*q3**2*q4**2*q5**2 - 540*p**2*q1**2*q2*q3**2*q4**2*q5 + 2160*p**2*q1**2*q2*q3**2*q4**2 + 60*p**2*q1**2*q2*q3**2*q4*q5**3 - 540*p**2*q1**2*q2*q3**2*q4*q5**2 + 2160*p**2*q1**2*q2*q3**2*q4*q5 - 5040*p**2*q1**2*q2*q3**2*q4 + 60*p**2*q1**2*q2*q3*q4**5 + 60*p**2*q1**2*q2*q3*q4**4*q5 - 540*p**2*q1**2*q2*q3*q4**4 + 60*p**2*q1**2*q2*q3*q4**3*q5**2 - 540*p**2*q1**2*q2*q3*q4**3*q5 + 2160*p**2*q1**2*q2*q3*q4**3 + 60*p**2*q1**2*q2*q3*q4**2*q5**3 - 540*p**2*q1**2*q2*q3*q4**2*q5**2 + 2160*p**2*q1**2*q2*q3*q4**2*q5 - 5040*p**2*q1**2*q2*q3*q4**2 + 60*p**2*q1**2*q2*q3*q4*q5**4 - 540*p**2*q1**2*q2*q3*q4*q5**3 + 2160*p**2*q1**2*q2*q3*q4*q5**2 - 5040*p**2*q1**2*q2*q3*q4*q5 + 7560*p**2*q1**2*q2*q3*q4 + 60*p**2*q1*q2**6*q3*q4 + 60*p**2*q1*q2**5*q3**2*q4 + 60*p**2*q1*q2**5*q3*q4**2 + 60*p**2*q1*q2**5*q3*q4*q5 - 540*p**2*q1*q2**5*q3*q4 + 60*p**2*q1*q2**4*q3**3*q4 + 60*p**2*q1*q2**4*q3**2*q4**2 + 60*p**2*q1*q2**4*q3**2*q4*q5 - 540*p**2*q1*q2**4*q3**2*q4 + 60*p**2*q1*q2**4*q3*q4**3 + 60*p**2*q1*q2**4*q3*q4**2*q5 - 540*p**2*q1*q2**4*q3*q4**2 + 60*p**2*q1*q2**4*q3*q4*q5**2 - 540*p**2*q1*q2**4*q3*q4*q5 + 2160*p**2*q1*q2**4*q3*q4 + 60*p**2*q1*q2**3*q3**4*q4 + 60*p**2*q1*q2**3*q3**3*q4**2 + 60*p**2*q1*q2**3*q3**3*q4*q5 - 540*p**2*q1*q2**3*q3**3*q4 + 60*p**2*q1*q2**3*q3**2*q4**3 + 60*p**2*q1*q2**3*q3**2*q4**2*q5 - 540*p**2*q1*q2**3*q3**2*q4**2 + 60*p**2*q1*q2**3*q3**2*q4*q5**2 - 540*p**2*q1*q2**3*q3**2*q4*q5 + 2160*p**2*q1*q2**3*q3**2*q4 + 60*p**2*q1*q2**3*q3*q4**4 + 60*p**2*q1*q2**3*q3*q4**3*q5 - 540*p**2*q1*q2**3*q3*q4**3 + 60*p**2*q1*q2**3*q3*q4**2*q5**2 - 540*p**2*q1*q2**3*q3*q4**2*q5 + 2160*p**2*q1*q2**3*q3*q4**2 + 60*p**2*q1*q2**3*q3*q4*q5**3 - 540*p**2*q1*q2**3*q3*q4*q5**2 + 2160*p**2*q1*q2**3*q3*q4*q5 - 5040*p**2*q1*q2**3*q3*q4 + 60*p**2*q1*q2**2*q3**5*q4 + 60*p**2*q1*q2**2*q3**4*q4**2 + 60*p**2*q1*q2**2*q3**4*q4*q5 - 540*p**2*q1*q2**2*q3**4*q4 + 60*p**2*q1*q2**2*q3**3*q4**3 + 60*p**2*q1*q2**2*q3**3*q4**2*q5 - 540*p**2*q1*q2**2*q3**3*q4**2 + 60*p**2*q1*q2**2*q3**3*q4*q5**2 - 540*p**2*q1*q2**2*q3**3*q4*q5 + 2160*p**2*q1*q2**2*q3**3*q4 + 60*p**2*q1*q2**2*q3**2*q4**4 + 60*p**2*q1*q2**2*q3**2*q4**3*q5 - 540*p**2*q1*q2**2*q3**2*q4**3 + 60*p**2*q1*q2**2*q3**2*q4**2*q5**2 - 540*p**2*q1*q2**2*q3**2*q4**2*q5 + 2160*p**2*q1*q2**2*q3**2*q4**2 + 60*p**2*q1*q2**2*q3**2*q4*q5**3 - 540*p**2*q1*q2**2*q3**2*q4*q5**2 + 2160*p**2*q1*q2**2*q3**2*q4*q5 - 5040*p**2*q1*q2**2*q3**2*q4 + 60*p**2*q1*q2**2*q3*q4**5 + 60*p**2*q1*q2**2*q3*q4**4*q5 - 540*p**2*q1*q2**2*q3*q4**4 + 60*p**2*q1*q2**2*q3*q4**3*q5**2 - 540*p**2*q1*q2**2*q3*q4**3*q5 + 2160*p**2*q1*q2**2*q3*q4**3 + 60*p**2*q1*q2**2*q3*q4**2*q5**3 - 540*p**2*q1*q2**2*q3*q4**2*q5**2 + 2160*p**2*q1*q2**2*q3*q4**2*q5 - 5040*p**2*q1*q2**2*q3*q4**2 + 60*p**2*q1*q2**2*q3*q4*q5**4 - 540*p**2*q1*q2**2*q3*q4*q5**3 + 2160*p**2*q1*q2**2*q3*q4*q5**2 - 5040*p**2*q1*q2**2*q3*q4*q5 + 7560*p**2*q1*q2**2*q3*q4 + 60*p**2*q1*q2*q3**6*q4 + 60*p**2*q1*q2*q3**5*q4**2 + 60*p**2*q1*q2*q3**5*q4*q5 - 540*p**2*q1*q2*q3**5*q4 + 60*p**2*q1*q2*q3**4*q4**3 + 60*p**2*q1*q2*q3**4*q4**2*q5 - 540*p**2*q1*q2*q3**4*q4**2 + 60*p**2*q1*q2*q3**4*q4*q5**2 - 540*p**2*q1*q2*q3**4*q4*q5 + 2160*p**2*q1*q2*q3**4*q4 + 60*p**2*q1*q2*q3**3*q4**4 + 60*p**2*q1*q2*q3**3*q4**3*q5 - 540*p**2*q1*q2*q3**3*q4**3 + 60*p**2*q1*q2*q3**3*q4**2*q5**2 - 540*p**2*q1*q2*q3**3*q4**2*q5 + 2160*p**2*q1*q2*q3**3*q4**2 + 60*p**2*q1*q2*q3**3*q4*q5**3 - 540*p**2*q1*q2*q3**3*q4*q5**2 + 2160*p**2*q1*q2*q3**3*q4*q5 - 5040*p**2*q1*q2*q3**3*q4 + 60*p**2*q1*q2*q3**2*q4**5 + 60*p**2*q1*q2*q3**2*q4**4*q5 - 540*p**2*q1*q2*q3**2*q4**4 + 60*p**2*q1*q2*q3**2*q4**3*q5**2 - 540*p**2*q1*q2*q3**2*q4**3*q5 + 2160*p**2*q1*q2*q3**2*q4**3 + 60*p**2*q1*q2*q3**2*q4**2*q5**3 - 540*p**2*q1*q2*q3**2*q4**2*q5**2 + 2160*p**2*q1*q2*q3**2*q4**2*q5 - 5040*p**2*q1*q2*q3**2*q4**2 + 60*p**2*q1*q2*q3**2*q4*q5**4 - 540*p**2*q1*q2*q3**2*q4*q5**3 + 2160*p**2*q1*q2*q3**2*q4*q5**2 - 5040*p**2*q1*q2*q3**2*q4*q5 + 7560*p**2*q1*q2*q3**2*q4 + 60*p**2*q1*q2*q3*q4**6 + 60*p**2*q1*q2*q3*q4**5*q5 - 540*p**2*q1*q2*q3*q4**5 + 60*p**2*q1*q2*q3*q4**4*q5**2 - 540*p**2*q1*q2*q3*q4**4*q5 + 2160*p**2*q1*q2*q3*q4**4 + 60*p**2*q1*q2*q3*q4**3*q5**3 - 540*p**2*q1*q2*q3*q4**3*q5**2 + 2160*p**2*q1*q2*q3*q4**3*q5 - 5040*p**2*q1*q2*q3*q4**3 + 60*p**2*q1*q2*q3*q4**2*q5**4 - 540*p**2*q1*q2*q3*q4**2*q5**3 + 2160*p**2*q1*q2*q3*q4**2*q5**2 - 5040*p**2*q1*q2*q3*q4**2*q5 + 7560*p**2*q1*q2*q3*q4**2 + 60*p**2*q1*q2*q3*q4*q5**5 - 540*p**2*q1*q2*q3*q4*q5**4 + 2160*p**2*q1*q2*q3*q4*q5**3 - 5040*p**2*q1*q2*q3*q4*q5**2 + 7560*p**2*q1*q2*q3*q4*q5 - 7560*p**2*q1*q2*q3*q4 - 135*p**2*q2**6*q3*q4 - 135*p**2*q2**5*q3**2*q4 - 135*p**2*q2**5*q3*q4**2 - 135*p**2*q2**5*q3*q4*q5 + 1080*p**2*q2**5*q3*q4 - 135*p**2*q2**4*q3**3*q4 - 135*p**2*q2**4*q3**2*q4**2 - 135*p**2*q2**4*q3**2*q4*q5 + 1080*p**2*q2**4*q3**2*q4 - 135*p**2*q2**4*q3*q4**3 - 135*p**2*q2**4*q3*q4**2*q5 + 1080*p**2*q2**4*q3*q4**2 - 135*p**2*q2**4*q3*q4*q5**2 + 1080*p**2*q2**4*q3*q4*q5 - 3780*p**2*q2**4*q3*q4 - 135*p**2*q2**3*q3**4*q4 - 135*p**2*q2**3*q3**3*q4**2 - 135*p**2*q2**3*q3**3*q4*q5 + 1080*p**2*q2**3*q3**3*q4 - 135*p**2*q2**3*q3**2*q4**3 - 135*p**2*q2**3*q3**2*q4**2*q5 + 1080*p**2*q2**3*q3**2*q4**2 - 135*p**2*q2**3*q3**2*q4*q5**2 + 1080*p**2*q2**3*q3**2*q4*q5 - 3780*p**2*q2**3*q3**2*q4 - 135*p**2*q2**3*q3*q4**4 - 135*p**2*q2**3*q3*q4**3*q5 + 1080*p**2*q2**3*q3*q4**3 - 135*p**2*q2**3*q3*q4**2*q5**2 + 1080*p**2*q2**3*q3*q4**2*q5 - 3780*p**2*q2**3*q3*q4**2 - 135*p**2*q2**3*q3*q4*q5**3 + 1080*p**2*q2**3*q3*q4*q5**2 - 3780*p**2*q2**3*q3*q4*q5 + 7560*p**2*q2**3*q3*q4 - 135*p**2*q2**2*q3**5*q4 - 135*p**2*q2**2*q3**4*q4**2 - 135*p**2*q2**2*q3**4*q4*q5 + 1080*p**2*q2**2*q3**4*q4 - 135*p**2*q2**2*q3**3*q4**3 - 135*p**2*q2**2*q3**3*q4**2*q5 + 1080*p**2*q2**2*q3**3*q4**2 - 135*p**2*q2**2*q3**3*q4*q5**2 + 1080*p**2*q2**2*q3**3*q4*q5 - 3780*p**2*q2**2*q3**3*q4 - 135*p**2*q2**2*q3**2*q4**4 - 135*p**2*q2**2*q3**2*q4**3*q5 + 1080*p**2*q2**2*q3**2*q4**3 - 135*p**2*q2**2*q3**2*q4**2*q5**2 + 1080*p**2*q2**2*q3**2*q4**2*q5 - 3780*p**2*q2**2*q3**2*q4**2 - 135*p**2*q2**2*q3**2*q4*q5**3 + 1080*p**2*q2**2*q3**2*q4*q5**2 - 3780*p**2*q2**2*q3**2*q4*q5 + 7560*p**2*q2**2*q3**2*q4 - 135*p**2*q2**2*q3*q4**5 - 135*p**2*q2**2*q3*q4**4*q5 + 1080*p**2*q2**2*q3*q4**4 - 135*p**2*q2**2*q3*q4**3*q5**2 + 1080*p**2*q2**2*q3*q4**3*q5 - 3780*p**2*q2**2*q3*q4**3 - 135*p**2*q2**2*q3*q4**2*q5**3 + 1080*p**2*q2**2*q3*q4**2*q5**2 - 3780*p**2*q2**2*q3*q4**2*q5 + 7560*p**2*q2**2*q3*q4**2 - 135*p**2*q2**2*q3*q4*q5**4 + 1080*p**2*q2**2*q3*q4*q5**3 - 3780*p**2*q2**2*q3*q4*q5**2 + 7560*p**2*q2**2*q3*q4*q5 - 9450*p**2*q2**2*q3*q4 - 135*p**2*q2*q3**6*q4 - 135*p**2*q2*q3**5*q4**2 - 135*p**2*q2*q3**5*q4*q5 + 1080*p**2*q2*q3**5*q4 - 135*p**2*q2*q3**4*q4**3 - 135*p**2*q2*q3**4*q4**2*q5 + 1080*p**2*q2*q3**4*q4**2 - 135*p**2*q2*q3**4*q4*q5**2 + 1080*p**2*q2*q3**4*q4*q5 - 3780*p**2*q2*q3**4*q4 - 135*p**2*q2*q3**3*q4**4 - 135*p**2*q2*q3**3*q4**3*q5 + 1080*p**2*q2*q3**3*q4**3 - 135*p**2*q2*q3**3*q4**2*q5**2 + 1080*p**2*q2*q3**3*q4**2*q5 - 3780*p**2*q2*q3**3*q4**2 - 135*p**2*q2*q3**3*q4*q5**3 + 1080*p**2*q2*q3**3*q4*q5**2 - 3780*p**2*q2*q3**3*q4*q5 + 7560*p**2*q2*q3**3*q4 - 135*p**2*q2*q3**2*q4**5 - 135*p**2*q2*q3**2*q4**4*q5 + 1080*p**2*q2*q3**2*q4**4 - 135*p**2*q2*q3**2*q4**3*q5**2 + 1080*p**2*q2*q3**2*q4**3*q5 - 3780*p**2*q2*q3**2*q4**3 - 135*p**2*q2*q3**2*q4**2*q5**3 + 1080*p**2*q2*q3**2*q4**2*q5**2 - 3780*p**2*q2*q3**2*q4**2*q5 + 7560*p**2*q2*q3**2*q4**2 - 135*p**2*q2*q3**2*q4*q5**4 + 1080*p**2*q2*q3**2*q4*q5**3 - 3780*p**2*q2*q3**2*q4*q5**2 + 7560*p**2*q2*q3**2*q4*q5 - 9450*p**2*q2*q3**2*q4 - 135*p**2*q2*q3*q4**6 - 135*p**2*q2*q3*q4**5*q5 + 1080*p**2*q2*q3*q4**5 - 135*p**2*q2*q3*q4**4*q5**2 + 1080*p**2*q2*q3*q4**4*q5 - 3780*p**2*q2*q3*q4**4 - 135*p**2*q2*q3*q4**3*q5**3 + 1080*p**2*q2*q3*q4**3*q5**2 - 3780*p**2*q2*q3*q4**3*q5 + 7560*p**2*q2*q3*q4**3 - 135*p**2*q2*q3*q4**2*q5**4 + 1080*p**2*q2*q3*q4**2*q5**3 - 3780*p**2*q2*q3*q4**2*q5**2 + 7560*p**2*q2*q3*q4**2*q5 - 9450*p**2*q2*q3*q4**2 - 135*p**2*q2*q3*q4*q5**5 + 1080*p**2*q2*q3*q4*q5**4 - 3780*p**2*q2*q3*q4*q5**3 + 7560*p**2*q2*q3*q4*q5**2 - 9450*p**2*q2*q3*q4*q5 + 7560*p**2*q2*q3*q4 + 120*p**2*q3**6*q4 + 120*p**2*q3**5*q4**2 + 120*p**2*q3**5*q4*q5 - 840*p**2*q3**5*q4 + 120*p**2*q3**4*q4**3 + 120*p**2*q3**4*q4**2*q5 - 840*p**2*q3**4*q4**2 + 120*p**2*q3**4*q4*q5**2 - 840*p**2*q3**4*q4*q5 + 2520*p**2*q3**4*q4 + 120*p**2*q3**3*q4**4 + 120*p**2*q3**3*q4**3*q5 - 840*p**2*q3**3*q4**3 + 120*p**2*q3**3*q4**2*q5**2 - 840*p**2*q3**3*q4**2*q5 + 2520*p**2*q3**3*q4**2 + 120*p**2*q3**3*q4*q5**3 - 840*p**2*q3**3*q4*q5**2 + 2520*p**2*q3**3*q4*q5 - 4200*p**2*q3**3*q4 + 120*p**2*q3**2*q4**5 + 120*p**2*q3**2*q4**4*q5 - 840*p**2*q3**2*q4**4 + 120*p**2*q3**2*q4**3*q5**2 - 840*p**2*q3**2*q4**3*q5 + 2520*p**2*q3**2*q4**3 + 120*p**2*q3**2*q4**2*q5**3 - 840*p**2*q3**2*q4**2*q5**2 + 2520*p**2*q3**2*q4**2*q5 - 4200*p**2*q3**2*q4**2 + 120*p**2*q3**2*q4*q5**4 - 840*p**2*q3**2*q4*q5**3 + 2520*p**2*q3**2*q4*q5**2 - 4200*p**2*q3**2*q4*q5 + 4200*p**2*q3**2*q4 + 120*p**2*q3*q4**6 + 120*p**2*q3*q4**5*q5 - 840*p**2*q3*q4**5 + 120*p**2*q3*q4**4*q5**2 - 840*p**2*q3*q4**4*q5 + 2520*p**2*q3*q4**4 + 120*p**2*q3*q4**3*q5**3 - 840*p**2*q3*q4**3*q5**2 + 2520*p**2*q3*q4**3*q5 - 4200*p**2*q3*q4**3 + 120*p**2*q3*q4**2*q5**4 - 840*p**2*q3*q4**2*q5**3 + 2520*p**2*q3*q4**2*q5**2 - 4200*p**2*q3*q4**2*q5 + 4200*p**2*q3*q4**2 + 120*p**2*q3*q4*q5**5 - 840*p**2*q3*q4*q5**4 + 2520*p**2*q3*q4*q5**3 - 4200*p**2*q3*q4*q5**2 + 4200*p**2*q3*q4*q5 - 2520*p**2*q3*q4 - 40*p*q1**6*q2*q3*q4 - 40*p*q1**5*q2**2*q3*q4 - 40*p*q1**5*q2*q3**2*q4 - 40*p*q1**5*q2*q3*q4**2 - 40*p*q1**5*q2*q3*q4*q5 + 360*p*q1**5*q2*q3*q4 - 40*p*q1**4*q2**3*q3*q4 - 40*p*q1**4*q2**2*q3**2*q4 - 40*p*q1**4*q2**2*q3*q4**2 - 40*p*q1**4*q2**2*q3*q4*q5 + 360*p*q1**4*q2**2*q3*q4 - 40*p*q1**4*q2*q3**3*q4 - 40*p*q1**4*q2*q3**2*q4**2 - 40*p*q1**4*q2*q3**2*q4*q5 + 360*p*q1**4*q2*q3**2*q4 - 40*p*q1**4*q2*q3*q4**3 - 40*p*q1**4*q2*q3*q4**2*q5 + 360*p*q1**4*q2*q3*q4**2 - 40*p*q1**4*q2*q3*q4*q5**2 + 360*p*q1**4*q2*q3*q4*q5 - 1440*p*q1**4*q2*q3*q4 - 40*p*q1**3*q2**4*q3*q4 - 40*p*q1**3*q2**3*q3**2*q4 - 40*p*q1**3*q2**3*q3*q4**2 - 40*p*q1**3*q2**3*q3*q4*q5 + 360*p*q1**3*q2**3*q3*q4 - 40*p*q1**3*q2**2*q3**3*q4 - 40*p*q1**3*q2**2*q3**2*q4**2 - 40*p*q1**3*q2**2*q3**2*q4*q5 + 360*p*q1**3*q2**2*q3**2*q4 - 40*p*q1**3*q2**2*q3*q4**3 - 40*p*q1**3*q2**2*q3*q4**2*q5 + 360*p*q1**3*q2**2*q3*q4**2 - 40*p*q1**3*q2**2*q3*q4*q5**2 + 360*p*q1**3*q2**2*q3*q4*q5 - 1440*p*q1**3*q2**2*q3*q4 - 40*p*q1**3*q2*q3**4*q4 - 40*p*q1**3*q2*q3**3*q4**2 - 40*p*q1**3*q2*q3**3*q4*q5 + 360*p*q1**3*q2*q3**3*q4 - 40*p*q1**3*q2*q3**2*q4**3 - 40*p*q1**3*q2*q3**2*q4**2*q5 + 360*p*q1**3*q2*q3**2*q4**2 - 40*p*q1**3*q2*q3**2*q4*q5**2 + 360*p*q1**3*q2*q3**2*q4*q5 - 1440*p*q1**3*q2*q3**2*q4 - 40*p*q1**3*q2*q3*q4**4 - 40*p*q1**3*q2*q3*q4**3*q5 + 360*p*q1**3*q2*q3*q4**3 - 40*p*q1**3*q2*q3*q4**2*q5**2 + 360*p*q1**3*q2*q3*q4**2*q5 - 1440*p*q1**3*q2*q3*q4**2 - 40*p*q1**3*q2*q3*q4*q5**3 + 360*p*q1**3*q2*q3*q4*q5**2 - 1440*p*q1**3*q2*q3*q4*q5 + 3360*p*q1**3*q2*q3*q4 - 40*p*q1**2*q2**5*q3*q4 - 40*p*q1**2*q2**4*q3**2*q4 - 40*p*q1**2*q2**4*q3*q4**2 - 40*p*q1**2*q2**4*q3*q4*q5 + 360*p*q1**2*q2**4*q3*q4 - 40*p*q1**2*q2**3*q3**3*q4 - 40*p*q1**2*q2**3*q3**2*q4**2 - 40*p*q1**2*q2**3*q3**2*q4*q5 + 360*p*q1**2*q2**3*q3**2*q4 - 40*p*q1**2*q2**3*q3*q4**3 - 40*p*q1**2*q2**3*q3*q4**2*q5 + 360*p*q1**2*q2**3*q3*q4**2 - 40*p*q1**2*q2**3*q3*q4*q5**2 + 360*p*q1**2*q2**3*q3*q4*q5 - 1440*p*q1**2*q2**3*q3*q4 - 40*p*q1**2*q2**2*q3**4*q4 - 40*p*q1**2*q2**2*q3**3*q4**2 - 40*p*q1**2*q2**2*q3**3*q4*q5 + 360*p*q1**2*q2**2*q3**3*q4 - 40*p*q1**2*q2**2*q3**2*q4**3 - 40*p*q1**2*q2**2*q3**2*q4**2*q5 + 360*p*q1**2*q2**2*q3**2*q4**2 - 40*p*q1**2*q2**2*q3**2*q4*q5**2 + 360*p*q1**2*q2**2*q3**2*q4*q5 - 1440*p*q1**2*q2**2*q3**2*q4 - 40*p*q1**2*q2**2*q3*q4**4 - 40*p*q1**2*q2**2*q3*q4**3*q5 + 360*p*q1**2*q2**2*q3*q4**3 - 40*p*q1**2*q2**2*q3*q4**2*q5**2 + 360*p*q1**2*q2**2*q3*q4**2*q5 - 1440*p*q1**2*q2**2*q3*q4**2 - 40*p*q1**2*q2**2*q3*q4*q5**3 + 360*p*q1**2*q2**2*q3*q4*q5**2 - 1440*p*q1**2*q2**2*q3*q4*q5 + 3360*p*q1**2*q2**2*q3*q4 - 40*p*q1**2*q2*q3**5*q4 - 40*p*q1**2*q2*q3**4*q4**2 - 40*p*q1**2*q2*q3**4*q4*q5 + 360*p*q1**2*q2*q3**4*q4 - 40*p*q1**2*q2*q3**3*q4**3 - 40*p*q1**2*q2*q3**3*q4**2*q5 + 360*p*q1**2*q2*q3**3*q4**2 - 40*p*q1**2*q2*q3**3*q4*q5**2 + 360*p*q1**2*q2*q3**3*q4*q5 - 1440*p*q1**2*q2*q3**3*q4 - 40*p*q1**2*q2*q3**2*q4**4 - 40*p*q1**2*q2*q3**2*q4**3*q5 + 360*p*q1**2*q2*q3**2*q4**3 - 40*p*q1**2*q2*q3**2*q4**2*q5**2 + 360*p*q1**2*q2*q3**2*q4**2*q5 - 1440*p*q1**2*q2*q3**2*q4**2 - 40*p*q1**2*q2*q3**2*q4*q5**3 + 360*p*q1**2*q2*q3**2*q4*q5**2 - 1440*p*q1**2*q2*q3**2*q4*q5 + 3360*p*q1**2*q2*q3**2*q4 - 40*p*q1**2*q2*q3*q4**5 - 40*p*q1**2*q2*q3*q4**4*q5 + 360*p*q1**2*q2*q3*q4**4 - 40*p*q1**2*q2*q3*q4**3*q5**2 + 360*p*q1**2*q2*q3*q4**3*q5 - 1440*p*q1**2*q2*q3*q4**3 - 40*p*q1**2*q2*q3*q4**2*q5**3 + 360*p*q1**2*q2*q3*q4**2*q5**2 - 1440*p*q1**2*q2*q3*q4**2*q5 + 3360*p*q1**2*q2*q3*q4**2 - 40*p*q1**2*q2*q3*q4*q5**4 + 360*p*q1**2*q2*q3*q4*q5**3 - 1440*p*q1**2*q2*q3*q4*q5**2 + 3360*p*q1**2*q2*q3*q4*q5 - 5040*p*q1**2*q2*q3*q4 - 40*p*q1*q2**6*q3*q4 - 40*p*q1*q2**5*q3**2*q4 - 40*p*q1*q2**5*q3*q4**2 - 40*p*q1*q2**5*q3*q4*q5 + 360*p*q1*q2**5*q3*q4 - 40*p*q1*q2**4*q3**3*q4 - 40*p*q1*q2**4*q3**2*q4**2 - 40*p*q1*q2**4*q3**2*q4*q5 + 360*p*q1*q2**4*q3**2*q4 - 40*p*q1*q2**4*q3*q4**3 - 40*p*q1*q2**4*q3*q4**2*q5 + 360*p*q1*q2**4*q3*q4**2 - 40*p*q1*q2**4*q3*q4*q5**2 + 360*p*q1*q2**4*q3*q4*q5 - 1440*p*q1*q2**4*q3*q4 - 40*p*q1*q2**3*q3**4*q4 - 40*p*q1*q2**3*q3**3*q4**2 - 40*p*q1*q2**3*q3**3*q4*q5 + 360*p*q1*q2**3*q3**3*q4 - 40*p*q1*q2**3*q3**2*q4**3 - 40*p*q1*q2**3*q3**2*q4**2*q5 + 360*p*q1*q2**3*q3**2*q4**2 - 40*p*q1*q2**3*q3**2*q4*q5**2 + 360*p*q1*q2**3*q3**2*q4*q5 - 1440*p*q1*q2**3*q3**2*q4 - 40*p*q1*q2**3*q3*q4**4 - 40*p*q1*q2**3*q3*q4**3*q5 + 360*p*q1*q2**3*q3*q4**3 - 40*p*q1*q2**3*q3*q4**2*q5**2 + 360*p*q1*q2**3*q3*q4**2*q5 - 1440*p*q1*q2**3*q3*q4**2 - 40*p*q1*q2**3*q3*q4*q5**3 + 360*p*q1*q2**3*q3*q4*q5**2 - 1440*p*q1*q2**3*q3*q4*q5 + 3360*p*q1*q2**3*q3*q4 - 40*p*q1*q2**2*q3**5*q4 - 40*p*q1*q2**2*q3**4*q4**2 - 40*p*q1*q2**2*q3**4*q4*q5 + 360*p*q1*q2**2*q3**4*q4 - 40*p*q1*q2**2*q3**3*q4**3 - 40*p*q1*q2**2*q3**3*q4**2*q5 + 360*p*q1*q2**2*q3**3*q4**2 - 40*p*q1*q2**2*q3**3*q4*q5**2 + 360*p*q1*q2**2*q3**3*q4*q5 - 1440*p*q1*q2**2*q3**3*q4 - 40*p*q1*q2**2*q3**2*q4**4 - 40*p*q1*q2**2*q3**2*q4**3*q5 + 360*p*q1*q2**2*q3**2*q4**3 - 40*p*q1*q2**2*q3**2*q4**2*q5**2 + 360*p*q1*q2**2*q3**2*q4**2*q5 - 1440*p*q1*q2**2*q3**2*q4**2 - 40*p*q1*q2**2*q3**2*q4*q5**3 + 360*p*q1*q2**2*q3**2*q4*q5**2 - 1440*p*q1*q2**2*q3**2*q4*q5 + 3360*p*q1*q2**2*q3**2*q4 - 40*p*q1*q2**2*q3*q4**5 - 40*p*q1*q2**2*q3*q4**4*q5 + 360*p*q1*q2**2*q3*q4**4 - 40*p*q1*q2**2*q3*q4**3*q5**2 + 360*p*q1*q2**2*q3*q4**3*q5 - 1440*p*q1*q2**2*q3*q4**3 - 40*p*q1*q2**2*q3*q4**2*q5**3 + 360*p*q1*q2**2*q3*q4**2*q5**2 - 1440*p*q1*q2**2*q3*q4**2*q5 + 3360*p*q1*q2**2*q3*q4**2 - 40*p*q1*q2**2*q3*q4*q5**4 + 360*p*q1*q2**2*q3*q4*q5**3 - 1440*p*q1*q2**2*q3*q4*q5**2 + 3360*p*q1*q2**2*q3*q4*q5 - 5040*p*q1*q2**2*q3*q4 - 40*p*q1*q2*q3**6*q4 - 40*p*q1*q2*q3**5*q4**2 - 40*p*q1*q2*q3**5*q4*q5 + 360*p*q1*q2*q3**5*q4 - 40*p*q1*q2*q3**4*q4**3 - 40*p*q1*q2*q3**4*q4**2*q5 + 360*p*q1*q2*q3**4*q4**2 - 40*p*q1*q2*q3**4*q4*q5**2 + 360*p*q1*q2*q3**4*q4*q5 - 1440*p*q1*q2*q3**4*q4 - 40*p*q1*q2*q3**3*q4**4 - 40*p*q1*q2*q3**3*q4**3*q5 + 360*p*q1*q2*q3**3*q4**3 - 40*p*q1*q2*q3**3*q4**2*q5**2 + 360*p*q1*q2*q3**3*q4**2*q5 - 1440*p*q1*q2*q3**3*q4**2 - 40*p*q1*q2*q3**3*q4*q5**3 + 360*p*q1*q2*q3**3*q4*q5**2 - 1440*p*q1*q2*q3**3*q4*q5 + 3360*p*q1*q2*q3**3*q4 - 40*p*q1*q2*q3**2*q4**5 - 40*p*q1*q2*q3**2*q4**4*q5 + 360*p*q1*q2*q3**2*q4**4 - 40*p*q1*q2*q3**2*q4**3*q5**2 + 360*p*q1*q2*q3**2*q4**3*q5 - 1440*p*q1*q2*q3**2*q4**3 - 40*p*q1*q2*q3**2*q4**2*q5**3 + 360*p*q1*q2*q3**2*q4**2*q5**2 - 1440*p*q1*q2*q3**2*q4**2*q5 + 3360*p*q1*q2*q3**2*q4**2 - 40*p*q1*q2*q3**2*q4*q5**4 + 360*p*q1*q2*q3**2*q4*q5**3 - 1440*p*q1*q2*q3**2*q4*q5**2 + 3360*p*q1*q2*q3**2*q4*q5 - 5040*p*q1*q2*q3**2*q4 - 40*p*q1*q2*q3*q4**6 - 40*p*q1*q2*q3*q4**5*q5 + 360*p*q1*q2*q3*q4**5 - 40*p*q1*q2*q3*q4**4*q5**2 + 360*p*q1*q2*q3*q4**4*q5 - 1440*p*q1*q2*q3*q4**4 - 40*p*q1*q2*q3*q4**3*q5**3 + 360*p*q1*q2*q3*q4**3*q5**2 - 1440*p*q1*q2*q3*q4**3*q5 + 3360*p*q1*q2*q3*q4**3 - 40*p*q1*q2*q3*q4**2*q5**4 + 360*p*q1*q2*q3*q4**2*q5**3 - 1440*p*q1*q2*q3*q4**2*q5**2 + 3360*p*q1*q2*q3*q4**2*q5 - 5040*p*q1*q2*q3*q4**2 - 40*p*q1*q2*q3*q4*q5**5 + 360*p*q1*q2*q3*q4*q5**4 - 1440*p*q1*q2*q3*q4*q5**3 + 3360*p*q1*q2*q3*q4*q5**2 - 5040*p*q1*q2*q3*q4*q5 + 5040*p*q1*q2*q3*q4 + 45*p*q2**6*q3*q4 + 45*p*q2**5*q3**2*q4 + 45*p*q2**5*q3*q4**2 + 45*p*q2**5*q3*q4*q5 - 360*p*q2**5*q3*q4 + 45*p*q2**4*q3**3*q4 + 45*p*q2**4*q3**2*q4**2 + 45*p*q2**4*q3**2*q4*q5 - 360*p*q2**4*q3**2*q4 + 45*p*q2**4*q3*q4**3 + 45*p*q2**4*q3*q4**2*q5 - 360*p*q2**4*q3*q4**2 + 45*p*q2**4*q3*q4*q5**2 - 360*p*q2**4*q3*q4*q5 + 1260*p*q2**4*q3*q4 + 45*p*q2**3*q3**4*q4 + 45*p*q2**3*q3**3*q4**2 + 45*p*q2**3*q3**3*q4*q5 - 360*p*q2**3*q3**3*q4 + 45*p*q2**3*q3**2*q4**3 + 45*p*q2**3*q3**2*q4**2*q5 - 360*p*q2**3*q3**2*q4**2 + 45*p*q2**3*q3**2*q4*q5**2 - 360*p*q2**3*q3**2*q4*q5 + 1260*p*q2**3*q3**2*q4 + 45*p*q2**3*q3*q4**4 + 45*p*q2**3*q3*q4**3*q5 - 360*p*q2**3*q3*q4**3 + 45*p*q2**3*q3*q4**2*q5**2 - 360*p*q2**3*q3*q4**2*q5 + 1260*p*q2**3*q3*q4**2 + 45*p*q2**3*q3*q4*q5**3 - 360*p*q2**3*q3*q4*q5**2 + 1260*p*q2**3*q3*q4*q5 - 2520*p*q2**3*q3*q4 + 45*p*q2**2*q3**5*q4 + 45*p*q2**2*q3**4*q4**2 + 45*p*q2**2*q3**4*q4*q5 - 360*p*q2**2*q3**4*q4 + 45*p*q2**2*q3**3*q4**3 + 45*p*q2**2*q3**3*q4**2*q5 - 360*p*q2**2*q3**3*q4**2 + 45*p*q2**2*q3**3*q4*q5**2 - 360*p*q2**2*q3**3*q4*q5 + 1260*p*q2**2*q3**3*q4 + 45*p*q2**2*q3**2*q4**4 + 45*p*q2**2*q3**2*q4**3*q5 - 360*p*q2**2*q3**2*q4**3 + 45*p*q2**2*q3**2*q4**2*q5**2 - 360*p*q2**2*q3**2*q4**2*q5 + 1260*p*q2**2*q3**2*q4**2 + 45*p*q2**2*q3**2*q4*q5**3 - 360*p*q2**2*q3**2*q4*q5**2 + 1260*p*q2**2*q3**2*q4*q5 - 2520*p*q2**2*q3**2*q4 + 45*p*q2**2*q3*q4**5 + 45*p*q2**2*q3*q4**4*q5 - 360*p*q2**2*q3*q4**4 + 45*p*q2**2*q3*q4**3*q5**2 - 360*p*q2**2*q3*q4**3*q5 + 1260*p*q2**2*q3*q4**3 + 45*p*q2**2*q3*q4**2*q5**3 - 360*p*q2**2*q3*q4**2*q5**2 + 1260*p*q2**2*q3*q4**2*q5 - 2520*p*q2**2*q3*q4**2 + 45*p*q2**2*q3*q4*q5**4 - 360*p*q2**2*q3*q4*q5**3 + 1260*p*q2**2*q3*q4*q5**2 - 2520*p*q2**2*q3*q4*q5 + 3150*p*q2**2*q3*q4 + 45*p*q2*q3**6*q4 + 45*p*q2*q3**5*q4**2 + 45*p*q2*q3**5*q4*q5 - 360*p*q2*q3**5*q4 + 45*p*q2*q3**4*q4**3 + 45*p*q2*q3**4*q4**2*q5 - 360*p*q2*q3**4*q4**2 + 45*p*q2*q3**4*q4*q5**2 - 360*p*q2*q3**4*q4*q5 + 1260*p*q2*q3**4*q4 + 45*p*q2*q3**3*q4**4 + 45*p*q2*q3**3*q4**3*q5 - 360*p*q2*q3**3*q4**3 + 45*p*q2*q3**3*q4**2*q5**2 - 360*p*q2*q3**3*q4**2*q5 + 1260*p*q2*q3**3*q4**2 + 45*p*q2*q3**3*q4*q5**3 - 360*p*q2*q3**3*q4*q5**2 + 1260*p*q2*q3**3*q4*q5 - 2520*p*q2*q3**3*q4 + 45*p*q2*q3**2*q4**5 + 45*p*q2*q3**2*q4**4*q5 - 360*p*q2*q3**2*q4**4 + 45*p*q2*q3**2*q4**3*q5**2 - 360*p*q2*q3**2*q4**3*q5 + 1260*p*q2*q3**2*q4**3 + 45*p*q2*q3**2*q4**2*q5**3 - 360*p*q2*q3**2*q4**2*q5**2 + 1260*p*q2*q3**2*q4**2*q5 - 2520*p*q2*q3**2*q4**2 + 45*p*q2*q3**2*q4*q5**4 - 360*p*q2*q3**2*q4*q5**3 + 1260*p*q2*q3**2*q4*q5**2 - 2520*p*q2*q3**2*q4*q5 + 3150*p*q2*q3**2*q4 + 45*p*q2*q3*q4**6 + 45*p*q2*q3*q4**5*q5 - 360*p*q2*q3*q4**5 + 45*p*q2*q3*q4**4*q5**2 - 360*p*q2*q3*q4**4*q5 + 1260*p*q2*q3*q4**4 + 45*p*q2*q3*q4**3*q5**3 - 360*p*q2*q3*q4**3*q5**2 + 1260*p*q2*q3*q4**3*q5 - 2520*p*q2*q3*q4**3 + 45*p*q2*q3*q4**2*q5**4 - 360*p*q2*q3*q4**2*q5**3 + 1260*p*q2*q3*q4**2*q5**2 - 2520*p*q2*q3*q4**2*q5 + 3150*p*q2*q3*q4**2 + 45*p*q2*q3*q4*q5**5 - 360*p*q2*q3*q4*q5**4 + 1260*p*q2*q3*q4*q5**3 - 2520*p*q2*q3*q4*q5**2 + 3150*p*q2*q3*q4*q5 - 2520*p*q2*q3*q4 + 10*q1**6*q2*q3*q4 + 10*q1**5*q2**2*q3*q4 + 10*q1**5*q2*q3**2*q4 + 10*q1**5*q2*q3*q4**2 + 10*q1**5*q2*q3*q4*q5 - 90*q1**5*q2*q3*q4 + 10*q1**4*q2**3*q3*q4 + 10*q1**4*q2**2*q3**2*q4 + 10*q1**4*q2**2*q3*q4**2 + 10*q1**4*q2**2*q3*q4*q5 - 90*q1**4*q2**2*q3*q4 + 10*q1**4*q2*q3**3*q4 + 10*q1**4*q2*q3**2*q4**2 + 10*q1**4*q2*q3**2*q4*q5 - 90*q1**4*q2*q3**2*q4 + 10*q1**4*q2*q3*q4**3 + 10*q1**4*q2*q3*q4**2*q5 - 90*q1**4*q2*q3*q4**2 + 10*q1**4*q2*q3*q4*q5**2 - 90*q1**4*q2*q3*q4*q5 + 360*q1**4*q2*q3*q4 + 10*q1**3*q2**4*q3*q4 + 10*q1**3*q2**3*q3**2*q4 + 10*q1**3*q2**3*q3*q4**2 + 10*q1**3*q2**3*q3*q4*q5 - 90*q1**3*q2**3*q3*q4 + 10*q1**3*q2**2*q3**3*q4 + 10*q1**3*q2**2*q3**2*q4**2 + 10*q1**3*q2**2*q3**2*q4*q5 - 90*q1**3*q2**2*q3**2*q4 + 10*q1**3*q2**2*q3*q4**3 + 10*q1**3*q2**2*q3*q4**2*q5 - 90*q1**3*q2**2*q3*q4**2 + 10*q1**3*q2**2*q3*q4*q5**2 - 90*q1**3*q2**2*q3*q4*q5 + 360*q1**3*q2**2*q3*q4 + 10*q1**3*q2*q3**4*q4 + 10*q1**3*q2*q3**3*q4**2 + 10*q1**3*q2*q3**3*q4*q5 - 90*q1**3*q2*q3**3*q4 + 10*q1**3*q2*q3**2*q4**3 + 10*q1**3*q2*q3**2*q4**2*q5 - 90*q1**3*q2*q3**2*q4**2 + 10*q1**3*q2*q3**2*q4*q5**2 - 90*q1**3*q2*q3**2*q4*q5 + 360*q1**3*q2*q3**2*q4 + 10*q1**3*q2*q3*q4**4 + 10*q1**3*q2*q3*q4**3*q5 - 90*q1**3*q2*q3*q4**3 + 10*q1**3*q2*q3*q4**2*q5**2 - 90*q1**3*q2*q3*q4**2*q5 + 360*q1**3*q2*q3*q4**2 + 10*q1**3*q2*q3*q4*q5**3 - 90*q1**3*q2*q3*q4*q5**2 + 360*q1**3*q2*q3*q4*q5 - 840*q1**3*q2*q3*q4 + 10*q1**2*q2**5*q3*q4 + 10*q1**2*q2**4*q3**2*q4 + 10*q1**2*q2**4*q3*q4**2 + 10*q1**2*q2**4*q3*q4*q5 - 90*q1**2*q2**4*q3*q4 + 10*q1**2*q2**3*q3**3*q4 + 10*q1**2*q2**3*q3**2*q4**2 + 10*q1**2*q2**3*q3**2*q4*q5 - 90*q1**2*q2**3*q3**2*q4 + 10*q1**2*q2**3*q3*q4**3 + 10*q1**2*q2**3*q3*q4**2*q5 - 90*q1**2*q2**3*q3*q4**2 + 10*q1**2*q2**3*q3*q4*q5**2 - 90*q1**2*q2**3*q3*q4*q5 + 360*q1**2*q2**3*q3*q4 + 10*q1**2*q2**2*q3**4*q4 + 10*q1**2*q2**2*q3**3*q4**2 + 10*q1**2*q2**2*q3**3*q4*q5 - 90*q1**2*q2**2*q3**3*q4 + 10*q1**2*q2**2*q3**2*q4**3 + 10*q1**2*q2**2*q3**2*q4**2*q5 - 90*q1**2*q2**2*q3**2*q4**2 + 10*q1**2*q2**2*q3**2*q4*q5**2 - 90*q1**2*q2**2*q3**2*q4*q5 + 360*q1**2*q2**2*q3**2*q4 + 10*q1**2*q2**2*q3*q4**4 + 10*q1**2*q2**2*q3*q4**3*q5 - 90*q1**2*q2**2*q3*q4**3 + 10*q1**2*q2**2*q3*q4**2*q5**2 - 90*q1**2*q2**2*q3*q4**2*q5 + 360*q1**2*q2**2*q3*q4**2 + 10*q1**2*q2**2*q3*q4*q5**3 - 90*q1**2*q2**2*q3*q4*q5**2 + 360*q1**2*q2**2*q3*q4*q5 - 840*q1**2*q2**2*q3*q4 + 10*q1**2*q2*q3**5*q4 + 10*q1**2*q2*q3**4*q4**2 + 10*q1**2*q2*q3**4*q4*q5 - 90*q1**2*q2*q3**4*q4 + 10*q1**2*q2*q3**3*q4**3 + 10*q1**2*q2*q3**3*q4**2*q5 - 90*q1**2*q2*q3**3*q4**2 + 10*q1**2*q2*q3**3*q4*q5**2 - 90*q1**2*q2*q3**3*q4*q5 + 360*q1**2*q2*q3**3*q4 + 10*q1**2*q2*q3**2*q4**4 + 10*q1**2*q2*q3**2*q4**3*q5 - 90*q1**2*q2*q3**2*q4**3 + 10*q1**2*q2*q3**2*q4**2*q5**2 - 90*q1**2*q2*q3**2*q4**2*q5 + 360*q1**2*q2*q3**2*q4**2 + 10*q1**2*q2*q3**2*q4*q5**3 - 90*q1**2*q2*q3**2*q4*q5**2 + 360*q1**2*q2*q3**2*q4*q5 - 840*q1**2*q2*q3**2*q4 + 10*q1**2*q2*q3*q4**5 + 10*q1**2*q2*q3*q4**4*q5 - 90*q1**2*q2*q3*q4**4 + 10*q1**2*q2*q3*q4**3*q5**2 - 90*q1**2*q2*q3*q4**3*q5 + 360*q1**2*q2*q3*q4**3 + 10*q1**2*q2*q3*q4**2*q5**3 - 90*q1**2*q2*q3*q4**2*q5**2 + 360*q1**2*q2*q3*q4**2*q5 - 840*q1**2*q2*q3*q4**2 + 10*q1**2*q2*q3*q4*q5**4 - 90*q1**2*q2*q3*q4*q5**3 + 360*q1**2*q2*q3*q4*q5**2 - 840*q1**2*q2*q3*q4*q5 + 1260*q1**2*q2*q3*q4 + 10*q1*q2**6*q3*q4 + 10*q1*q2**5*q3**2*q4 + 10*q1*q2**5*q3*q4**2 + 10*q1*q2**5*q3*q4*q5 - 90*q1*q2**5*q3*q4 + 10*q1*q2**4*q3**3*q4 + 10*q1*q2**4*q3**2*q4**2 + 10*q1*q2**4*q3**2*q4*q5 - 90*q1*q2**4*q3**2*q4 + 10*q1*q2**4*q3*q4**3 + 10*q1*q2**4*q3*q4**2*q5 - 90*q1*q2**4*q3*q4**2 + 10*q1*q2**4*q3*q4*q5**2 - 90*q1*q2**4*q3*q4*q5 + 360*q1*q2**4*q3*q4 + 10*q1*q2**3*q3**4*q4 + 10*q1*q2**3*q3**3*q4**2 + 10*q1*q2**3*q3**3*q4*q5 - 90*q1*q2**3*q3**3*q4 + 10*q1*q2**3*q3**2*q4**3 + 10*q1*q2**3*q3**2*q4**2*q5 - 90*q1*q2**3*q3**2*q4**2 + 10*q1*q2**3*q3**2*q4*q5**2 - 90*q1*q2**3*q3**2*q4*q5 + 360*q1*q2**3*q3**2*q4 + 10*q1*q2**3*q3*q4**4 + 10*q1*q2**3*q3*q4**3*q5 - 90*q1*q2**3*q3*q4**3 + 10*q1*q2**3*q3*q4**2*q5**2 - 90*q1*q2**3*q3*q4**2*q5 + 360*q1*q2**3*q3*q4**2 + 10*q1*q2**3*q3*q4*q5**3 - 90*q1*q2**3*q3*q4*q5**2 + 360*q1*q2**3*q3*q4*q5 - 840*q1*q2**3*q3*q4 + 10*q1*q2**2*q3**5*q4 + 10*q1*q2**2*q3**4*q4**2 + 10*q1*q2**2*q3**4*q4*q5 - 90*q1*q2**2*q3**4*q4 + 10*q1*q2**2*q3**3*q4**3 + 10*q1*q2**2*q3**3*q4**2*q5 - 90*q1*q2**2*q3**3*q4**2 + 10*q1*q2**2*q3**3*q4*q5**2 - 90*q1*q2**2*q3**3*q4*q5 + 360*q1*q2**2*q3**3*q4 + 10*q1*q2**2*q3**2*q4**4 + 10*q1*q2**2*q3**2*q4**3*q5 - 90*q1*q2**2*q3**2*q4**3 + 10*q1*q2**2*q3**2*q4**2*q5**2 - 90*q1*q2**2*q3**2*q4**2*q5 + 360*q1*q2**2*q3**2*q4**2 + 10*q1*q2**2*q3**2*q4*q5**3 - 90*q1*q2**2*q3**2*q4*q5**2 + 360*q1*q2**2*q3**2*q4*q5 - 840*q1*q2**2*q3**2*q4 + 10*q1*q2**2*q3*q4**5 + 10*q1*q2**2*q3*q4**4*q5 - 90*q1*q2**2*q3*q4**4 + 10*q1*q2**2*q3*q4**3*q5**2 - 90*q1*q2**2*q3*q4**3*q5 + 360*q1*q2**2*q3*q4**3 + 10*q1*q2**2*q3*q4**2*q5**3 - 90*q1*q2**2*q3*q4**2*q5**2 + 360*q1*q2**2*q3*q4**2*q5 - 840*q1*q2**2*q3*q4**2 + 10*q1*q2**2*q3*q4*q5**4 - 90*q1*q2**2*q3*q4*q5**3 + 360*q1*q2**2*q3*q4*q5**2 - 840*q1*q2**2*q3*q4*q5 + 1260*q1*q2**2*q3*q4 + 10*q1*q2*q3**6*q4 + 10*q1*q2*q3**5*q4**2 + 10*q1*q2*q3**5*q4*q5 - 90*q1*q2*q3**5*q4 + 10*q1*q2*q3**4*q4**3 + 10*q1*q2*q3**4*q4**2*q5 - 90*q1*q2*q3**4*q4**2 + 10*q1*q2*q3**4*q4*q5**2 - 90*q1*q2*q3**4*q4*q5 + 360*q1*q2*q3**4*q4 + 10*q1*q2*q3**3*q4**4 + 10*q1*q2*q3**3*q4**3*q5 - 90*q1*q2*q3**3*q4**3 + 10*q1*q2*q3**3*q4**2*q5**2 - 90*q1*q2*q3**3*q4**2*q5 + 360*q1*q2*q3**3*q4**2 + 10*q1*q2*q3**3*q4*q5**3 - 90*q1*q2*q3**3*q4*q5**2 + 360*q1*q2*q3**3*q4*q5 - 840*q1*q2*q3**3*q4 + 10*q1*q2*q3**2*q4**5 + 10*q1*q2*q3**2*q4**4*q5 - 90*q1*q2*q3**2*q4**4 + 10*q1*q2*q3**2*q4**3*q5**2 - 90*q1*q2*q3**2*q4**3*q5 + 360*q1*q2*q3**2*q4**3 + 10*q1*q2*q3**2*q4**2*q5**3 - 90*q1*q2*q3**2*q4**2*q5**2 + 360*q1*q2*q3**2*q4**2*q5 - 840*q1*q2*q3**2*q4**2 + 10*q1*q2*q3**2*q4*q5**4 - 90*q1*q2*q3**2*q4*q5**3 + 360*q1*q2*q3**2*q4*q5**2 - 840*q1*q2*q3**2*q4*q5 + 1260*q1*q2*q3**2*q4 + 10*q1*q2*q3*q4**6 + 10*q1*q2*q3*q4**5*q5 - 90*q1*q2*q3*q4**5 + 10*q1*q2*q3*q4**4*q5**2 - 90*q1*q2*q3*q4**4*q5 + 360*q1*q2*q3*q4**4 + 10*q1*q2*q3*q4**3*q5**3 - 90*q1*q2*q3*q4**3*q5**2 + 360*q1*q2*q3*q4**3*q5 - 840*q1*q2*q3*q4**3 + 10*q1*q2*q3*q4**2*q5**4 - 90*q1*q2*q3*q4**2*q5**3 + 360*q1*q2*q3*q4**2*q5**2 - 840*q1*q2*q3*q4**2*q5 + 1260*q1*q2*q3*q4**2 + 10*q1*q2*q3*q4*q5**5 - 90*q1*q2*q3*q4*q5**4 + 360*q1*q2*q3*q4*q5**3 - 840*q1*q2*q3*q4*q5**2 + 1260*q1*q2*q3*q4*q5 - 1260*q1*q2*q3*q4)'
        f_multiparam[10][6] = '-p*(p - 1)**4*(10*p**5*q1**5*q2*q3*q4*q5 + 10*p**5*q1**4*q2**2*q3*q4*q5 + 10*p**5*q1**4*q2*q3**2*q4*q5 + 10*p**5*q1**4*q2*q3*q4**2*q5 + 10*p**5*q1**4*q2*q3*q4*q5**2 + 10*p**5*q1**4*q2*q3*q4*q5*q6 - 90*p**5*q1**4*q2*q3*q4*q5 + 10*p**5*q1**3*q2**3*q3*q4*q5 + 10*p**5*q1**3*q2**2*q3**2*q4*q5 + 10*p**5*q1**3*q2**2*q3*q4**2*q5 + 10*p**5*q1**3*q2**2*q3*q4*q5**2 + 10*p**5*q1**3*q2**2*q3*q4*q5*q6 - 90*p**5*q1**3*q2**2*q3*q4*q5 + 10*p**5*q1**3*q2*q3**3*q4*q5 + 10*p**5*q1**3*q2*q3**2*q4**2*q5 + 10*p**5*q1**3*q2*q3**2*q4*q5**2 + 10*p**5*q1**3*q2*q3**2*q4*q5*q6 - 90*p**5*q1**3*q2*q3**2*q4*q5 + 10*p**5*q1**3*q2*q3*q4**3*q5 + 10*p**5*q1**3*q2*q3*q4**2*q5**2 + 10*p**5*q1**3*q2*q3*q4**2*q5*q6 - 90*p**5*q1**3*q2*q3*q4**2*q5 + 10*p**5*q1**3*q2*q3*q4*q5**3 + 10*p**5*q1**3*q2*q3*q4*q5**2*q6 - 90*p**5*q1**3*q2*q3*q4*q5**2 + 10*p**5*q1**3*q2*q3*q4*q5*q6**2 - 90*p**5*q1**3*q2*q3*q4*q5*q6 + 360*p**5*q1**3*q2*q3*q4*q5 + 10*p**5*q1**2*q2**4*q3*q4*q5 + 10*p**5*q1**2*q2**3*q3**2*q4*q5 + 10*p**5*q1**2*q2**3*q3*q4**2*q5 + 10*p**5*q1**2*q2**3*q3*q4*q5**2 + 10*p**5*q1**2*q2**3*q3*q4*q5*q6 - 90*p**5*q1**2*q2**3*q3*q4*q5 + 10*p**5*q1**2*q2**2*q3**3*q4*q5 + 10*p**5*q1**2*q2**2*q3**2*q4**2*q5 + 10*p**5*q1**2*q2**2*q3**2*q4*q5**2 + 10*p**5*q1**2*q2**2*q3**2*q4*q5*q6 - 90*p**5*q1**2*q2**2*q3**2*q4*q5 + 10*p**5*q1**2*q2**2*q3*q4**3*q5 + 10*p**5*q1**2*q2**2*q3*q4**2*q5**2 + 10*p**5*q1**2*q2**2*q3*q4**2*q5*q6 - 90*p**5*q1**2*q2**2*q3*q4**2*q5 + 10*p**5*q1**2*q2**2*q3*q4*q5**3 + 10*p**5*q1**2*q2**2*q3*q4*q5**2*q6 - 90*p**5*q1**2*q2**2*q3*q4*q5**2 + 10*p**5*q1**2*q2**2*q3*q4*q5*q6**2 - 90*p**5*q1**2*q2**2*q3*q4*q5*q6 + 360*p**5*q1**2*q2**2*q3*q4*q5 + 10*p**5*q1**2*q2*q3**4*q4*q5 + 10*p**5*q1**2*q2*q3**3*q4**2*q5 + 10*p**5*q1**2*q2*q3**3*q4*q5**2 + 10*p**5*q1**2*q2*q3**3*q4*q5*q6 - 90*p**5*q1**2*q2*q3**3*q4*q5 + 10*p**5*q1**2*q2*q3**2*q4**3*q5 + 10*p**5*q1**2*q2*q3**2*q4**2*q5**2 + 10*p**5*q1**2*q2*q3**2*q4**2*q5*q6 - 90*p**5*q1**2*q2*q3**2*q4**2*q5 + 10*p**5*q1**2*q2*q3**2*q4*q5**3 + 10*p**5*q1**2*q2*q3**2*q4*q5**2*q6 - 90*p**5*q1**2*q2*q3**2*q4*q5**2 + 10*p**5*q1**2*q2*q3**2*q4*q5*q6**2 - 90*p**5*q1**2*q2*q3**2*q4*q5*q6 + 360*p**5*q1**2*q2*q3**2*q4*q5 + 10*p**5*q1**2*q2*q3*q4**4*q5 + 10*p**5*q1**2*q2*q3*q4**3*q5**2 + 10*p**5*q1**2*q2*q3*q4**3*q5*q6 - 90*p**5*q1**2*q2*q3*q4**3*q5 + 10*p**5*q1**2*q2*q3*q4**2*q5**3 + 10*p**5*q1**2*q2*q3*q4**2*q5**2*q6 - 90*p**5*q1**2*q2*q3*q4**2*q5**2 + 10*p**5*q1**2*q2*q3*q4**2*q5*q6**2 - 90*p**5*q1**2*q2*q3*q4**2*q5*q6 + 360*p**5*q1**2*q2*q3*q4**2*q5 + 10*p**5*q1**2*q2*q3*q4*q5**4 + 10*p**5*q1**2*q2*q3*q4*q5**3*q6 - 90*p**5*q1**2*q2*q3*q4*q5**3 + 10*p**5*q1**2*q2*q3*q4*q5**2*q6**2 - 90*p**5*q1**2*q2*q3*q4*q5**2*q6 + 360*p**5*q1**2*q2*q3*q4*q5**2 + 10*p**5*q1**2*q2*q3*q4*q5*q6**3 - 90*p**5*q1**2*q2*q3*q4*q5*q6**2 + 360*p**5*q1**2*q2*q3*q4*q5*q6 - 840*p**5*q1**2*q2*q3*q4*q5 + 10*p**5*q1*q2**5*q3*q4*q5 + 10*p**5*q1*q2**4*q3**2*q4*q5 + 10*p**5*q1*q2**4*q3*q4**2*q5 + 10*p**5*q1*q2**4*q3*q4*q5**2 + 10*p**5*q1*q2**4*q3*q4*q5*q6 - 90*p**5*q1*q2**4*q3*q4*q5 + 10*p**5*q1*q2**3*q3**3*q4*q5 + 10*p**5*q1*q2**3*q3**2*q4**2*q5 + 10*p**5*q1*q2**3*q3**2*q4*q5**2 + 10*p**5*q1*q2**3*q3**2*q4*q5*q6 - 90*p**5*q1*q2**3*q3**2*q4*q5 + 10*p**5*q1*q2**3*q3*q4**3*q5 + 10*p**5*q1*q2**3*q3*q4**2*q5**2 + 10*p**5*q1*q2**3*q3*q4**2*q5*q6 - 90*p**5*q1*q2**3*q3*q4**2*q5 + 10*p**5*q1*q2**3*q3*q4*q5**3 + 10*p**5*q1*q2**3*q3*q4*q5**2*q6 - 90*p**5*q1*q2**3*q3*q4*q5**2 + 10*p**5*q1*q2**3*q3*q4*q5*q6**2 - 90*p**5*q1*q2**3*q3*q4*q5*q6 + 360*p**5*q1*q2**3*q3*q4*q5 + 10*p**5*q1*q2**2*q3**4*q4*q5 + 10*p**5*q1*q2**2*q3**3*q4**2*q5 + 10*p**5*q1*q2**2*q3**3*q4*q5**2 + 10*p**5*q1*q2**2*q3**3*q4*q5*q6 - 90*p**5*q1*q2**2*q3**3*q4*q5 + 10*p**5*q1*q2**2*q3**2*q4**3*q5 + 10*p**5*q1*q2**2*q3**2*q4**2*q5**2 + 10*p**5*q1*q2**2*q3**2*q4**2*q5*q6 - 90*p**5*q1*q2**2*q3**2*q4**2*q5 + 10*p**5*q1*q2**2*q3**2*q4*q5**3 + 10*p**5*q1*q2**2*q3**2*q4*q5**2*q6 - 90*p**5*q1*q2**2*q3**2*q4*q5**2 + 10*p**5*q1*q2**2*q3**2*q4*q5*q6**2 - 90*p**5*q1*q2**2*q3**2*q4*q5*q6 + 360*p**5*q1*q2**2*q3**2*q4*q5 + 10*p**5*q1*q2**2*q3*q4**4*q5 + 10*p**5*q1*q2**2*q3*q4**3*q5**2 + 10*p**5*q1*q2**2*q3*q4**3*q5*q6 - 90*p**5*q1*q2**2*q3*q4**3*q5 + 10*p**5*q1*q2**2*q3*q4**2*q5**3 + 10*p**5*q1*q2**2*q3*q4**2*q5**2*q6 - 90*p**5*q1*q2**2*q3*q4**2*q5**2 + 10*p**5*q1*q2**2*q3*q4**2*q5*q6**2 - 90*p**5*q1*q2**2*q3*q4**2*q5*q6 + 360*p**5*q1*q2**2*q3*q4**2*q5 + 10*p**5*q1*q2**2*q3*q4*q5**4 + 10*p**5*q1*q2**2*q3*q4*q5**3*q6 - 90*p**5*q1*q2**2*q3*q4*q5**3 + 10*p**5*q1*q2**2*q3*q4*q5**2*q6**2 - 90*p**5*q1*q2**2*q3*q4*q5**2*q6 + 360*p**5*q1*q2**2*q3*q4*q5**2 + 10*p**5*q1*q2**2*q3*q4*q5*q6**3 - 90*p**5*q1*q2**2*q3*q4*q5*q6**2 + 360*p**5*q1*q2**2*q3*q4*q5*q6 - 840*p**5*q1*q2**2*q3*q4*q5 + 10*p**5*q1*q2*q3**5*q4*q5 + 10*p**5*q1*q2*q3**4*q4**2*q5 + 10*p**5*q1*q2*q3**4*q4*q5**2 + 10*p**5*q1*q2*q3**4*q4*q5*q6 - 90*p**5*q1*q2*q3**4*q4*q5 + 10*p**5*q1*q2*q3**3*q4**3*q5 + 10*p**5*q1*q2*q3**3*q4**2*q5**2 + 10*p**5*q1*q2*q3**3*q4**2*q5*q6 - 90*p**5*q1*q2*q3**3*q4**2*q5 + 10*p**5*q1*q2*q3**3*q4*q5**3 + 10*p**5*q1*q2*q3**3*q4*q5**2*q6 - 90*p**5*q1*q2*q3**3*q4*q5**2 + 10*p**5*q1*q2*q3**3*q4*q5*q6**2 - 90*p**5*q1*q2*q3**3*q4*q5*q6 + 360*p**5*q1*q2*q3**3*q4*q5 + 10*p**5*q1*q2*q3**2*q4**4*q5 + 10*p**5*q1*q2*q3**2*q4**3*q5**2 + 10*p**5*q1*q2*q3**2*q4**3*q5*q6 - 90*p**5*q1*q2*q3**2*q4**3*q5 + 10*p**5*q1*q2*q3**2*q4**2*q5**3 + 10*p**5*q1*q2*q3**2*q4**2*q5**2*q6 - 90*p**5*q1*q2*q3**2*q4**2*q5**2 + 10*p**5*q1*q2*q3**2*q4**2*q5*q6**2 - 90*p**5*q1*q2*q3**2*q4**2*q5*q6 + 360*p**5*q1*q2*q3**2*q4**2*q5 + 10*p**5*q1*q2*q3**2*q4*q5**4 + 10*p**5*q1*q2*q3**2*q4*q5**3*q6 - 90*p**5*q1*q2*q3**2*q4*q5**3 + 10*p**5*q1*q2*q3**2*q4*q5**2*q6**2 - 90*p**5*q1*q2*q3**2*q4*q5**2*q6 + 360*p**5*q1*q2*q3**2*q4*q5**2 + 10*p**5*q1*q2*q3**2*q4*q5*q6**3 - 90*p**5*q1*q2*q3**2*q4*q5*q6**2 + 360*p**5*q1*q2*q3**2*q4*q5*q6 - 840*p**5*q1*q2*q3**2*q4*q5 + 10*p**5*q1*q2*q3*q4**5*q5 + 10*p**5*q1*q2*q3*q4**4*q5**2 + 10*p**5*q1*q2*q3*q4**4*q5*q6 - 90*p**5*q1*q2*q3*q4**4*q5 + 10*p**5*q1*q2*q3*q4**3*q5**3 + 10*p**5*q1*q2*q3*q4**3*q5**2*q6 - 90*p**5*q1*q2*q3*q4**3*q5**2 + 10*p**5*q1*q2*q3*q4**3*q5*q6**2 - 90*p**5*q1*q2*q3*q4**3*q5*q6 + 360*p**5*q1*q2*q3*q4**3*q5 + 10*p**5*q1*q2*q3*q4**2*q5**4 + 10*p**5*q1*q2*q3*q4**2*q5**3*q6 - 90*p**5*q1*q2*q3*q4**2*q5**3 + 10*p**5*q1*q2*q3*q4**2*q5**2*q6**2 - 90*p**5*q1*q2*q3*q4**2*q5**2*q6 + 360*p**5*q1*q2*q3*q4**2*q5**2 + 10*p**5*q1*q2*q3*q4**2*q5*q6**3 - 90*p**5*q1*q2*q3*q4**2*q5*q6**2 + 360*p**5*q1*q2*q3*q4**2*q5*q6 - 840*p**5*q1*q2*q3*q4**2*q5 + 10*p**5*q1*q2*q3*q4*q5**5 + 10*p**5*q1*q2*q3*q4*q5**4*q6 - 90*p**5*q1*q2*q3*q4*q5**4 + 10*p**5*q1*q2*q3*q4*q5**3*q6**2 - 90*p**5*q1*q2*q3*q4*q5**3*q6 + 360*p**5*q1*q2*q3*q4*q5**3 + 10*p**5*q1*q2*q3*q4*q5**2*q6**3 - 90*p**5*q1*q2*q3*q4*q5**2*q6**2 + 360*p**5*q1*q2*q3*q4*q5**2*q6 - 840*p**5*q1*q2*q3*q4*q5**2 + 10*p**5*q1*q2*q3*q4*q5*q6**4 - 90*p**5*q1*q2*q3*q4*q5*q6**3 + 360*p**5*q1*q2*q3*q4*q5*q6**2 - 840*p**5*q1*q2*q3*q4*q5*q6 + 1260*p**5*q1*q2*q3*q4*q5 - 45*p**5*q2**5*q3*q4*q5 - 45*p**5*q2**4*q3**2*q4*q5 - 45*p**5*q2**4*q3*q4**2*q5 - 45*p**5*q2**4*q3*q4*q5**2 - 45*p**5*q2**4*q3*q4*q5*q6 + 360*p**5*q2**4*q3*q4*q5 - 45*p**5*q2**3*q3**3*q4*q5 - 45*p**5*q2**3*q3**2*q4**2*q5 - 45*p**5*q2**3*q3**2*q4*q5**2 - 45*p**5*q2**3*q3**2*q4*q5*q6 + 360*p**5*q2**3*q3**2*q4*q5 - 45*p**5*q2**3*q3*q4**3*q5 - 45*p**5*q2**3*q3*q4**2*q5**2 - 45*p**5*q2**3*q3*q4**2*q5*q6 + 360*p**5*q2**3*q3*q4**2*q5 - 45*p**5*q2**3*q3*q4*q5**3 - 45*p**5*q2**3*q3*q4*q5**2*q6 + 360*p**5*q2**3*q3*q4*q5**2 - 45*p**5*q2**3*q3*q4*q5*q6**2 + 360*p**5*q2**3*q3*q4*q5*q6 - 1260*p**5*q2**3*q3*q4*q5 - 45*p**5*q2**2*q3**4*q4*q5 - 45*p**5*q2**2*q3**3*q4**2*q5 - 45*p**5*q2**2*q3**3*q4*q5**2 - 45*p**5*q2**2*q3**3*q4*q5*q6 + 360*p**5*q2**2*q3**3*q4*q5 - 45*p**5*q2**2*q3**2*q4**3*q5 - 45*p**5*q2**2*q3**2*q4**2*q5**2 - 45*p**5*q2**2*q3**2*q4**2*q5*q6 + 360*p**5*q2**2*q3**2*q4**2*q5 - 45*p**5*q2**2*q3**2*q4*q5**3 - 45*p**5*q2**2*q3**2*q4*q5**2*q6 + 360*p**5*q2**2*q3**2*q4*q5**2 - 45*p**5*q2**2*q3**2*q4*q5*q6**2 + 360*p**5*q2**2*q3**2*q4*q5*q6 - 1260*p**5*q2**2*q3**2*q4*q5 - 45*p**5*q2**2*q3*q4**4*q5 - 45*p**5*q2**2*q3*q4**3*q5**2 - 45*p**5*q2**2*q3*q4**3*q5*q6 + 360*p**5*q2**2*q3*q4**3*q5 - 45*p**5*q2**2*q3*q4**2*q5**3 - 45*p**5*q2**2*q3*q4**2*q5**2*q6 + 360*p**5*q2**2*q3*q4**2*q5**2 - 45*p**5*q2**2*q3*q4**2*q5*q6**2 + 360*p**5*q2**2*q3*q4**2*q5*q6 - 1260*p**5*q2**2*q3*q4**2*q5 - 45*p**5*q2**2*q3*q4*q5**4 - 45*p**5*q2**2*q3*q4*q5**3*q6 + 360*p**5*q2**2*q3*q4*q5**3 - 45*p**5*q2**2*q3*q4*q5**2*q6**2 + 360*p**5*q2**2*q3*q4*q5**2*q6 - 1260*p**5*q2**2*q3*q4*q5**2 - 45*p**5*q2**2*q3*q4*q5*q6**3 + 360*p**5*q2**2*q3*q4*q5*q6**2 - 1260*p**5*q2**2*q3*q4*q5*q6 + 2520*p**5*q2**2*q3*q4*q5 - 45*p**5*q2*q3**5*q4*q5 - 45*p**5*q2*q3**4*q4**2*q5 - 45*p**5*q2*q3**4*q4*q5**2 - 45*p**5*q2*q3**4*q4*q5*q6 + 360*p**5*q2*q3**4*q4*q5 - 45*p**5*q2*q3**3*q4**3*q5 - 45*p**5*q2*q3**3*q4**2*q5**2 - 45*p**5*q2*q3**3*q4**2*q5*q6 + 360*p**5*q2*q3**3*q4**2*q5 - 45*p**5*q2*q3**3*q4*q5**3 - 45*p**5*q2*q3**3*q4*q5**2*q6 + 360*p**5*q2*q3**3*q4*q5**2 - 45*p**5*q2*q3**3*q4*q5*q6**2 + 360*p**5*q2*q3**3*q4*q5*q6 - 1260*p**5*q2*q3**3*q4*q5 - 45*p**5*q2*q3**2*q4**4*q5 - 45*p**5*q2*q3**2*q4**3*q5**2 - 45*p**5*q2*q3**2*q4**3*q5*q6 + 360*p**5*q2*q3**2*q4**3*q5 - 45*p**5*q2*q3**2*q4**2*q5**3 - 45*p**5*q2*q3**2*q4**2*q5**2*q6 + 360*p**5*q2*q3**2*q4**2*q5**2 - 45*p**5*q2*q3**2*q4**2*q5*q6**2 + 360*p**5*q2*q3**2*q4**2*q5*q6 - 1260*p**5*q2*q3**2*q4**2*q5 - 45*p**5*q2*q3**2*q4*q5**4 - 45*p**5*q2*q3**2*q4*q5**3*q6 + 360*p**5*q2*q3**2*q4*q5**3 - 45*p**5*q2*q3**2*q4*q5**2*q6**2 + 360*p**5*q2*q3**2*q4*q5**2*q6 - 1260*p**5*q2*q3**2*q4*q5**2 - 45*p**5*q2*q3**2*q4*q5*q6**3 + 360*p**5*q2*q3**2*q4*q5*q6**2 - 1260*p**5*q2*q3**2*q4*q5*q6 + 2520*p**5*q2*q3**2*q4*q5 - 45*p**5*q2*q3*q4**5*q5 - 45*p**5*q2*q3*q4**4*q5**2 - 45*p**5*q2*q3*q4**4*q5*q6 + 360*p**5*q2*q3*q4**4*q5 - 45*p**5*q2*q3*q4**3*q5**3 - 45*p**5*q2*q3*q4**3*q5**2*q6 + 360*p**5*q2*q3*q4**3*q5**2 - 45*p**5*q2*q3*q4**3*q5*q6**2 + 360*p**5*q2*q3*q4**3*q5*q6 - 1260*p**5*q2*q3*q4**3*q5 - 45*p**5*q2*q3*q4**2*q5**4 - 45*p**5*q2*q3*q4**2*q5**3*q6 + 360*p**5*q2*q3*q4**2*q5**3 - 45*p**5*q2*q3*q4**2*q5**2*q6**2 + 360*p**5*q2*q3*q4**2*q5**2*q6 - 1260*p**5*q2*q3*q4**2*q5**2 - 45*p**5*q2*q3*q4**2*q5*q6**3 + 360*p**5*q2*q3*q4**2*q5*q6**2 - 1260*p**5*q2*q3*q4**2*q5*q6 + 2520*p**5*q2*q3*q4**2*q5 - 45*p**5*q2*q3*q4*q5**5 - 45*p**5*q2*q3*q4*q5**4*q6 + 360*p**5*q2*q3*q4*q5**4 - 45*p**5*q2*q3*q4*q5**3*q6**2 + 360*p**5*q2*q3*q4*q5**3*q6 - 1260*p**5*q2*q3*q4*q5**3 - 45*p**5*q2*q3*q4*q5**2*q6**3 + 360*p**5*q2*q3*q4*q5**2*q6**2 - 1260*p**5*q2*q3*q4*q5**2*q6 + 2520*p**5*q2*q3*q4*q5**2 - 45*p**5*q2*q3*q4*q5*q6**4 + 360*p**5*q2*q3*q4*q5*q6**3 - 1260*p**5*q2*q3*q4*q5*q6**2 + 2520*p**5*q2*q3*q4*q5*q6 - 3150*p**5*q2*q3*q4*q5 + 120*p**5*q3**5*q4*q5 + 120*p**5*q3**4*q4**2*q5 + 120*p**5*q3**4*q4*q5**2 + 120*p**5*q3**4*q4*q5*q6 - 840*p**5*q3**4*q4*q5 + 120*p**5*q3**3*q4**3*q5 + 120*p**5*q3**3*q4**2*q5**2 + 120*p**5*q3**3*q4**2*q5*q6 - 840*p**5*q3**3*q4**2*q5 + 120*p**5*q3**3*q4*q5**3 + 120*p**5*q3**3*q4*q5**2*q6 - 840*p**5*q3**3*q4*q5**2 + 120*p**5*q3**3*q4*q5*q6**2 - 840*p**5*q3**3*q4*q5*q6 + 2520*p**5*q3**3*q4*q5 + 120*p**5*q3**2*q4**4*q5 + 120*p**5*q3**2*q4**3*q5**2 + 120*p**5*q3**2*q4**3*q5*q6 - 840*p**5*q3**2*q4**3*q5 + 120*p**5*q3**2*q4**2*q5**3 + 120*p**5*q3**2*q4**2*q5**2*q6 - 840*p**5*q3**2*q4**2*q5**2 + 120*p**5*q3**2*q4**2*q5*q6**2 - 840*p**5*q3**2*q4**2*q5*q6 + 2520*p**5*q3**2*q4**2*q5 + 120*p**5*q3**2*q4*q5**4 + 120*p**5*q3**2*q4*q5**3*q6 - 840*p**5*q3**2*q4*q5**3 + 120*p**5*q3**2*q4*q5**2*q6**2 - 840*p**5*q3**2*q4*q5**2*q6 + 2520*p**5*q3**2*q4*q5**2 + 120*p**5*q3**2*q4*q5*q6**3 - 840*p**5*q3**2*q4*q5*q6**2 + 2520*p**5*q3**2*q4*q5*q6 - 4200*p**5*q3**2*q4*q5 + 120*p**5*q3*q4**5*q5 + 120*p**5*q3*q4**4*q5**2 + 120*p**5*q3*q4**4*q5*q6 - 840*p**5*q3*q4**4*q5 + 120*p**5*q3*q4**3*q5**3 + 120*p**5*q3*q4**3*q5**2*q6 - 840*p**5*q3*q4**3*q5**2 + 120*p**5*q3*q4**3*q5*q6**2 - 840*p**5*q3*q4**3*q5*q6 + 2520*p**5*q3*q4**3*q5 + 120*p**5*q3*q4**2*q5**4 + 120*p**5*q3*q4**2*q5**3*q6 - 840*p**5*q3*q4**2*q5**3 + 120*p**5*q3*q4**2*q5**2*q6**2 - 840*p**5*q3*q4**2*q5**2*q6 + 2520*p**5*q3*q4**2*q5**2 + 120*p**5*q3*q4**2*q5*q6**3 - 840*p**5*q3*q4**2*q5*q6**2 + 2520*p**5*q3*q4**2*q5*q6 - 4200*p**5*q3*q4**2*q5 + 120*p**5*q3*q4*q5**5 + 120*p**5*q3*q4*q5**4*q6 - 840*p**5*q3*q4*q5**4 + 120*p**5*q3*q4*q5**3*q6**2 - 840*p**5*q3*q4*q5**3*q6 + 2520*p**5*q3*q4*q5**3 + 120*p**5*q3*q4*q5**2*q6**3 - 840*p**5*q3*q4*q5**2*q6**2 + 2520*p**5*q3*q4*q5**2*q6 - 4200*p**5*q3*q4*q5**2 + 120*p**5*q3*q4*q5*q6**4 - 840*p**5*q3*q4*q5*q6**3 + 2520*p**5*q3*q4*q5*q6**2 - 4200*p**5*q3*q4*q5*q6 + 4200*p**5*q3*q4*q5 - 210*p**5*q4**5*q5 - 210*p**5*q4**4*q5**2 - 210*p**5*q4**4*q5*q6 + 1260*p**5*q4**4*q5 - 210*p**5*q4**3*q5**3 - 210*p**5*q4**3*q5**2*q6 + 1260*p**5*q4**3*q5**2 - 210*p**5*q4**3*q5*q6**2 + 1260*p**5*q4**3*q5*q6 - 3150*p**5*q4**3*q5 - 210*p**5*q4**2*q5**4 - 210*p**5*q4**2*q5**3*q6 + 1260*p**5*q4**2*q5**3 - 210*p**5*q4**2*q5**2*q6**2 + 1260*p**5*q4**2*q5**2*q6 - 3150*p**5*q4**2*q5**2 - 210*p**5*q4**2*q5*q6**3 + 1260*p**5*q4**2*q5*q6**2 - 3150*p**5*q4**2*q5*q6 + 4200*p**5*q4**2*q5 - 210*p**5*q4*q5**5 - 210*p**5*q4*q5**4*q6 + 1260*p**5*q4*q5**4 - 210*p**5*q4*q5**3*q6**2 + 1260*p**5*q4*q5**3*q6 - 3150*p**5*q4*q5**3 - 210*p**5*q4*q5**2*q6**3 + 1260*p**5*q4*q5**2*q6**2 - 3150*p**5*q4*q5**2*q6 + 4200*p**5*q4*q5**2 - 210*p**5*q4*q5*q6**4 + 1260*p**5*q4*q5*q6**3 - 3150*p**5*q4*q5*q6**2 + 4200*p**5*q4*q5*q6 - 3150*p**5*q4*q5 + 252*p**5*q5**5 + 252*p**5*q5**4*q6 - 1260*p**5*q5**4 + 252*p**5*q5**3*q6**2 - 1260*p**5*q5**3*q6 + 2520*p**5*q5**3 + 252*p**5*q5**2*q6**3 - 1260*p**5*q5**2*q6**2 + 2520*p**5*q5**2*q6 - 2520*p**5*q5**2 + 252*p**5*q5*q6**4 - 1260*p**5*q5*q6**3 + 2520*p**5*q5*q6**2 - 2520*p**5*q5*q6 + 1260*p**5*q5 - 210*p**5*q6**4 + 840*p**5*q6**3 - 1260*p**5*q6**2 + 840*p**5*q6 - 210*p**5 - 50*p**4*q1**5*q2*q3*q4*q5 - 50*p**4*q1**4*q2**2*q3*q4*q5 - 50*p**4*q1**4*q2*q3**2*q4*q5 - 50*p**4*q1**4*q2*q3*q4**2*q5 - 50*p**4*q1**4*q2*q3*q4*q5**2 - 50*p**4*q1**4*q2*q3*q4*q5*q6 + 450*p**4*q1**4*q2*q3*q4*q5 - 50*p**4*q1**3*q2**3*q3*q4*q5 - 50*p**4*q1**3*q2**2*q3**2*q4*q5 - 50*p**4*q1**3*q2**2*q3*q4**2*q5 - 50*p**4*q1**3*q2**2*q3*q4*q5**2 - 50*p**4*q1**3*q2**2*q3*q4*q5*q6 + 450*p**4*q1**3*q2**2*q3*q4*q5 - 50*p**4*q1**3*q2*q3**3*q4*q5 - 50*p**4*q1**3*q2*q3**2*q4**2*q5 - 50*p**4*q1**3*q2*q3**2*q4*q5**2 - 50*p**4*q1**3*q2*q3**2*q4*q5*q6 + 450*p**4*q1**3*q2*q3**2*q4*q5 - 50*p**4*q1**3*q2*q3*q4**3*q5 - 50*p**4*q1**3*q2*q3*q4**2*q5**2 - 50*p**4*q1**3*q2*q3*q4**2*q5*q6 + 450*p**4*q1**3*q2*q3*q4**2*q5 - 50*p**4*q1**3*q2*q3*q4*q5**3 - 50*p**4*q1**3*q2*q3*q4*q5**2*q6 + 450*p**4*q1**3*q2*q3*q4*q5**2 - 50*p**4*q1**3*q2*q3*q4*q5*q6**2 + 450*p**4*q1**3*q2*q3*q4*q5*q6 - 1800*p**4*q1**3*q2*q3*q4*q5 - 50*p**4*q1**2*q2**4*q3*q4*q5 - 50*p**4*q1**2*q2**3*q3**2*q4*q5 - 50*p**4*q1**2*q2**3*q3*q4**2*q5 - 50*p**4*q1**2*q2**3*q3*q4*q5**2 - 50*p**4*q1**2*q2**3*q3*q4*q5*q6 + 450*p**4*q1**2*q2**3*q3*q4*q5 - 50*p**4*q1**2*q2**2*q3**3*q4*q5 - 50*p**4*q1**2*q2**2*q3**2*q4**2*q5 - 50*p**4*q1**2*q2**2*q3**2*q4*q5**2 - 50*p**4*q1**2*q2**2*q3**2*q4*q5*q6 + 450*p**4*q1**2*q2**2*q3**2*q4*q5 - 50*p**4*q1**2*q2**2*q3*q4**3*q5 - 50*p**4*q1**2*q2**2*q3*q4**2*q5**2 - 50*p**4*q1**2*q2**2*q3*q4**2*q5*q6 + 450*p**4*q1**2*q2**2*q3*q4**2*q5 - 50*p**4*q1**2*q2**2*q3*q4*q5**3 - 50*p**4*q1**2*q2**2*q3*q4*q5**2*q6 + 450*p**4*q1**2*q2**2*q3*q4*q5**2 - 50*p**4*q1**2*q2**2*q3*q4*q5*q6**2 + 450*p**4*q1**2*q2**2*q3*q4*q5*q6 - 1800*p**4*q1**2*q2**2*q3*q4*q5 - 50*p**4*q1**2*q2*q3**4*q4*q5 - 50*p**4*q1**2*q2*q3**3*q4**2*q5 - 50*p**4*q1**2*q2*q3**3*q4*q5**2 - 50*p**4*q1**2*q2*q3**3*q4*q5*q6 + 450*p**4*q1**2*q2*q3**3*q4*q5 - 50*p**4*q1**2*q2*q3**2*q4**3*q5 - 50*p**4*q1**2*q2*q3**2*q4**2*q5**2 - 50*p**4*q1**2*q2*q3**2*q4**2*q5*q6 + 450*p**4*q1**2*q2*q3**2*q4**2*q5 - 50*p**4*q1**2*q2*q3**2*q4*q5**3 - 50*p**4*q1**2*q2*q3**2*q4*q5**2*q6 + 450*p**4*q1**2*q2*q3**2*q4*q5**2 - 50*p**4*q1**2*q2*q3**2*q4*q5*q6**2 + 450*p**4*q1**2*q2*q3**2*q4*q5*q6 - 1800*p**4*q1**2*q2*q3**2*q4*q5 - 50*p**4*q1**2*q2*q3*q4**4*q5 - 50*p**4*q1**2*q2*q3*q4**3*q5**2 - 50*p**4*q1**2*q2*q3*q4**3*q5*q6 + 450*p**4*q1**2*q2*q3*q4**3*q5 - 50*p**4*q1**2*q2*q3*q4**2*q5**3 - 50*p**4*q1**2*q2*q3*q4**2*q5**2*q6 + 450*p**4*q1**2*q2*q3*q4**2*q5**2 - 50*p**4*q1**2*q2*q3*q4**2*q5*q6**2 + 450*p**4*q1**2*q2*q3*q4**2*q5*q6 - 1800*p**4*q1**2*q2*q3*q4**2*q5 - 50*p**4*q1**2*q2*q3*q4*q5**4 - 50*p**4*q1**2*q2*q3*q4*q5**3*q6 + 450*p**4*q1**2*q2*q3*q4*q5**3 - 50*p**4*q1**2*q2*q3*q4*q5**2*q6**2 + 450*p**4*q1**2*q2*q3*q4*q5**2*q6 - 1800*p**4*q1**2*q2*q3*q4*q5**2 - 50*p**4*q1**2*q2*q3*q4*q5*q6**3 + 450*p**4*q1**2*q2*q3*q4*q5*q6**2 - 1800*p**4*q1**2*q2*q3*q4*q5*q6 + 4200*p**4*q1**2*q2*q3*q4*q5 - 50*p**4*q1*q2**5*q3*q4*q5 - 50*p**4*q1*q2**4*q3**2*q4*q5 - 50*p**4*q1*q2**4*q3*q4**2*q5 - 50*p**4*q1*q2**4*q3*q4*q5**2 - 50*p**4*q1*q2**4*q3*q4*q5*q6 + 450*p**4*q1*q2**4*q3*q4*q5 - 50*p**4*q1*q2**3*q3**3*q4*q5 - 50*p**4*q1*q2**3*q3**2*q4**2*q5 - 50*p**4*q1*q2**3*q3**2*q4*q5**2 - 50*p**4*q1*q2**3*q3**2*q4*q5*q6 + 450*p**4*q1*q2**3*q3**2*q4*q5 - 50*p**4*q1*q2**3*q3*q4**3*q5 - 50*p**4*q1*q2**3*q3*q4**2*q5**2 - 50*p**4*q1*q2**3*q3*q4**2*q5*q6 + 450*p**4*q1*q2**3*q3*q4**2*q5 - 50*p**4*q1*q2**3*q3*q4*q5**3 - 50*p**4*q1*q2**3*q3*q4*q5**2*q6 + 450*p**4*q1*q2**3*q3*q4*q5**2 - 50*p**4*q1*q2**3*q3*q4*q5*q6**2 + 450*p**4*q1*q2**3*q3*q4*q5*q6 - 1800*p**4*q1*q2**3*q3*q4*q5 - 50*p**4*q1*q2**2*q3**4*q4*q5 - 50*p**4*q1*q2**2*q3**3*q4**2*q5 - 50*p**4*q1*q2**2*q3**3*q4*q5**2 - 50*p**4*q1*q2**2*q3**3*q4*q5*q6 + 450*p**4*q1*q2**2*q3**3*q4*q5 - 50*p**4*q1*q2**2*q3**2*q4**3*q5 - 50*p**4*q1*q2**2*q3**2*q4**2*q5**2 - 50*p**4*q1*q2**2*q3**2*q4**2*q5*q6 + 450*p**4*q1*q2**2*q3**2*q4**2*q5 - 50*p**4*q1*q2**2*q3**2*q4*q5**3 - 50*p**4*q1*q2**2*q3**2*q4*q5**2*q6 + 450*p**4*q1*q2**2*q3**2*q4*q5**2 - 50*p**4*q1*q2**2*q3**2*q4*q5*q6**2 + 450*p**4*q1*q2**2*q3**2*q4*q5*q6 - 1800*p**4*q1*q2**2*q3**2*q4*q5 - 50*p**4*q1*q2**2*q3*q4**4*q5 - 50*p**4*q1*q2**2*q3*q4**3*q5**2 - 50*p**4*q1*q2**2*q3*q4**3*q5*q6 + 450*p**4*q1*q2**2*q3*q4**3*q5 - 50*p**4*q1*q2**2*q3*q4**2*q5**3 - 50*p**4*q1*q2**2*q3*q4**2*q5**2*q6 + 450*p**4*q1*q2**2*q3*q4**2*q5**2 - 50*p**4*q1*q2**2*q3*q4**2*q5*q6**2 + 450*p**4*q1*q2**2*q3*q4**2*q5*q6 - 1800*p**4*q1*q2**2*q3*q4**2*q5 - 50*p**4*q1*q2**2*q3*q4*q5**4 - 50*p**4*q1*q2**2*q3*q4*q5**3*q6 + 450*p**4*q1*q2**2*q3*q4*q5**3 - 50*p**4*q1*q2**2*q3*q4*q5**2*q6**2 + 450*p**4*q1*q2**2*q3*q4*q5**2*q6 - 1800*p**4*q1*q2**2*q3*q4*q5**2 - 50*p**4*q1*q2**2*q3*q4*q5*q6**3 + 450*p**4*q1*q2**2*q3*q4*q5*q6**2 - 1800*p**4*q1*q2**2*q3*q4*q5*q6 + 4200*p**4*q1*q2**2*q3*q4*q5 - 50*p**4*q1*q2*q3**5*q4*q5 - 50*p**4*q1*q2*q3**4*q4**2*q5 - 50*p**4*q1*q2*q3**4*q4*q5**2 - 50*p**4*q1*q2*q3**4*q4*q5*q6 + 450*p**4*q1*q2*q3**4*q4*q5 - 50*p**4*q1*q2*q3**3*q4**3*q5 - 50*p**4*q1*q2*q3**3*q4**2*q5**2 - 50*p**4*q1*q2*q3**3*q4**2*q5*q6 + 450*p**4*q1*q2*q3**3*q4**2*q5 - 50*p**4*q1*q2*q3**3*q4*q5**3 - 50*p**4*q1*q2*q3**3*q4*q5**2*q6 + 450*p**4*q1*q2*q3**3*q4*q5**2 - 50*p**4*q1*q2*q3**3*q4*q5*q6**2 + 450*p**4*q1*q2*q3**3*q4*q5*q6 - 1800*p**4*q1*q2*q3**3*q4*q5 - 50*p**4*q1*q2*q3**2*q4**4*q5 - 50*p**4*q1*q2*q3**2*q4**3*q5**2 - 50*p**4*q1*q2*q3**2*q4**3*q5*q6 + 450*p**4*q1*q2*q3**2*q4**3*q5 - 50*p**4*q1*q2*q3**2*q4**2*q5**3 - 50*p**4*q1*q2*q3**2*q4**2*q5**2*q6 + 450*p**4*q1*q2*q3**2*q4**2*q5**2 - 50*p**4*q1*q2*q3**2*q4**2*q5*q6**2 + 450*p**4*q1*q2*q3**2*q4**2*q5*q6 - 1800*p**4*q1*q2*q3**2*q4**2*q5 - 50*p**4*q1*q2*q3**2*q4*q5**4 - 50*p**4*q1*q2*q3**2*q4*q5**3*q6 + 450*p**4*q1*q2*q3**2*q4*q5**3 - 50*p**4*q1*q2*q3**2*q4*q5**2*q6**2 + 450*p**4*q1*q2*q3**2*q4*q5**2*q6 - 1800*p**4*q1*q2*q3**2*q4*q5**2 - 50*p**4*q1*q2*q3**2*q4*q5*q6**3 + 450*p**4*q1*q2*q3**2*q4*q5*q6**2 - 1800*p**4*q1*q2*q3**2*q4*q5*q6 + 4200*p**4*q1*q2*q3**2*q4*q5 - 50*p**4*q1*q2*q3*q4**5*q5 - 50*p**4*q1*q2*q3*q4**4*q5**2 - 50*p**4*q1*q2*q3*q4**4*q5*q6 + 450*p**4*q1*q2*q3*q4**4*q5 - 50*p**4*q1*q2*q3*q4**3*q5**3 - 50*p**4*q1*q2*q3*q4**3*q5**2*q6 + 450*p**4*q1*q2*q3*q4**3*q5**2 - 50*p**4*q1*q2*q3*q4**3*q5*q6**2 + 450*p**4*q1*q2*q3*q4**3*q5*q6 - 1800*p**4*q1*q2*q3*q4**3*q5 - 50*p**4*q1*q2*q3*q4**2*q5**4 - 50*p**4*q1*q2*q3*q4**2*q5**3*q6 + 450*p**4*q1*q2*q3*q4**2*q5**3 - 50*p**4*q1*q2*q3*q4**2*q5**2*q6**2 + 450*p**4*q1*q2*q3*q4**2*q5**2*q6 - 1800*p**4*q1*q2*q3*q4**2*q5**2 - 50*p**4*q1*q2*q3*q4**2*q5*q6**3 + 450*p**4*q1*q2*q3*q4**2*q5*q6**2 - 1800*p**4*q1*q2*q3*q4**2*q5*q6 + 4200*p**4*q1*q2*q3*q4**2*q5 - 50*p**4*q1*q2*q3*q4*q5**5 - 50*p**4*q1*q2*q3*q4*q5**4*q6 + 450*p**4*q1*q2*q3*q4*q5**4 - 50*p**4*q1*q2*q3*q4*q5**3*q6**2 + 450*p**4*q1*q2*q3*q4*q5**3*q6 - 1800*p**4*q1*q2*q3*q4*q5**3 - 50*p**4*q1*q2*q3*q4*q5**2*q6**3 + 450*p**4*q1*q2*q3*q4*q5**2*q6**2 - 1800*p**4*q1*q2*q3*q4*q5**2*q6 + 4200*p**4*q1*q2*q3*q4*q5**2 - 50*p**4*q1*q2*q3*q4*q5*q6**4 + 450*p**4*q1*q2*q3*q4*q5*q6**3 - 1800*p**4*q1*q2*q3*q4*q5*q6**2 + 4200*p**4*q1*q2*q3*q4*q5*q6 - 6300*p**4*q1*q2*q3*q4*q5 + 180*p**4*q2**5*q3*q4*q5 + 180*p**4*q2**4*q3**2*q4*q5 + 180*p**4*q2**4*q3*q4**2*q5 + 180*p**4*q2**4*q3*q4*q5**2 + 180*p**4*q2**4*q3*q4*q5*q6 - 1440*p**4*q2**4*q3*q4*q5 + 180*p**4*q2**3*q3**3*q4*q5 + 180*p**4*q2**3*q3**2*q4**2*q5 + 180*p**4*q2**3*q3**2*q4*q5**2 + 180*p**4*q2**3*q3**2*q4*q5*q6 - 1440*p**4*q2**3*q3**2*q4*q5 + 180*p**4*q2**3*q3*q4**3*q5 + 180*p**4*q2**3*q3*q4**2*q5**2 + 180*p**4*q2**3*q3*q4**2*q5*q6 - 1440*p**4*q2**3*q3*q4**2*q5 + 180*p**4*q2**3*q3*q4*q5**3 + 180*p**4*q2**3*q3*q4*q5**2*q6 - 1440*p**4*q2**3*q3*q4*q5**2 + 180*p**4*q2**3*q3*q4*q5*q6**2 - 1440*p**4*q2**3*q3*q4*q5*q6 + 5040*p**4*q2**3*q3*q4*q5 + 180*p**4*q2**2*q3**4*q4*q5 + 180*p**4*q2**2*q3**3*q4**2*q5 + 180*p**4*q2**2*q3**3*q4*q5**2 + 180*p**4*q2**2*q3**3*q4*q5*q6 - 1440*p**4*q2**2*q3**3*q4*q5 + 180*p**4*q2**2*q3**2*q4**3*q5 + 180*p**4*q2**2*q3**2*q4**2*q5**2 + 180*p**4*q2**2*q3**2*q4**2*q5*q6 - 1440*p**4*q2**2*q3**2*q4**2*q5 + 180*p**4*q2**2*q3**2*q4*q5**3 + 180*p**4*q2**2*q3**2*q4*q5**2*q6 - 1440*p**4*q2**2*q3**2*q4*q5**2 + 180*p**4*q2**2*q3**2*q4*q5*q6**2 - 1440*p**4*q2**2*q3**2*q4*q5*q6 + 5040*p**4*q2**2*q3**2*q4*q5 + 180*p**4*q2**2*q3*q4**4*q5 + 180*p**4*q2**2*q3*q4**3*q5**2 + 180*p**4*q2**2*q3*q4**3*q5*q6 - 1440*p**4*q2**2*q3*q4**3*q5 + 180*p**4*q2**2*q3*q4**2*q5**3 + 180*p**4*q2**2*q3*q4**2*q5**2*q6 - 1440*p**4*q2**2*q3*q4**2*q5**2 + 180*p**4*q2**2*q3*q4**2*q5*q6**2 - 1440*p**4*q2**2*q3*q4**2*q5*q6 + 5040*p**4*q2**2*q3*q4**2*q5 + 180*p**4*q2**2*q3*q4*q5**4 + 180*p**4*q2**2*q3*q4*q5**3*q6 - 1440*p**4*q2**2*q3*q4*q5**3 + 180*p**4*q2**2*q3*q4*q5**2*q6**2 - 1440*p**4*q2**2*q3*q4*q5**2*q6 + 5040*p**4*q2**2*q3*q4*q5**2 + 180*p**4*q2**2*q3*q4*q5*q6**3 - 1440*p**4*q2**2*q3*q4*q5*q6**2 + 5040*p**4*q2**2*q3*q4*q5*q6 - 10080*p**4*q2**2*q3*q4*q5 + 180*p**4*q2*q3**5*q4*q5 + 180*p**4*q2*q3**4*q4**2*q5 + 180*p**4*q2*q3**4*q4*q5**2 + 180*p**4*q2*q3**4*q4*q5*q6 - 1440*p**4*q2*q3**4*q4*q5 + 180*p**4*q2*q3**3*q4**3*q5 + 180*p**4*q2*q3**3*q4**2*q5**2 + 180*p**4*q2*q3**3*q4**2*q5*q6 - 1440*p**4*q2*q3**3*q4**2*q5 + 180*p**4*q2*q3**3*q4*q5**3 + 180*p**4*q2*q3**3*q4*q5**2*q6 - 1440*p**4*q2*q3**3*q4*q5**2 + 180*p**4*q2*q3**3*q4*q5*q6**2 - 1440*p**4*q2*q3**3*q4*q5*q6 + 5040*p**4*q2*q3**3*q4*q5 + 180*p**4*q2*q3**2*q4**4*q5 + 180*p**4*q2*q3**2*q4**3*q5**2 + 180*p**4*q2*q3**2*q4**3*q5*q6 - 1440*p**4*q2*q3**2*q4**3*q5 + 180*p**4*q2*q3**2*q4**2*q5**3 + 180*p**4*q2*q3**2*q4**2*q5**2*q6 - 1440*p**4*q2*q3**2*q4**2*q5**2 + 180*p**4*q2*q3**2*q4**2*q5*q6**2 - 1440*p**4*q2*q3**2*q4**2*q5*q6 + 5040*p**4*q2*q3**2*q4**2*q5 + 180*p**4*q2*q3**2*q4*q5**4 + 180*p**4*q2*q3**2*q4*q5**3*q6 - 1440*p**4*q2*q3**2*q4*q5**3 + 180*p**4*q2*q3**2*q4*q5**2*q6**2 - 1440*p**4*q2*q3**2*q4*q5**2*q6 + 5040*p**4*q2*q3**2*q4*q5**2 + 180*p**4*q2*q3**2*q4*q5*q6**3 - 1440*p**4*q2*q3**2*q4*q5*q6**2 + 5040*p**4*q2*q3**2*q4*q5*q6 - 10080*p**4*q2*q3**2*q4*q5 + 180*p**4*q2*q3*q4**5*q5 + 180*p**4*q2*q3*q4**4*q5**2 + 180*p**4*q2*q3*q4**4*q5*q6 - 1440*p**4*q2*q3*q4**4*q5 + 180*p**4*q2*q3*q4**3*q5**3 + 180*p**4*q2*q3*q4**3*q5**2*q6 - 1440*p**4*q2*q3*q4**3*q5**2 + 180*p**4*q2*q3*q4**3*q5*q6**2 - 1440*p**4*q2*q3*q4**3*q5*q6 + 5040*p**4*q2*q3*q4**3*q5 + 180*p**4*q2*q3*q4**2*q5**4 + 180*p**4*q2*q3*q4**2*q5**3*q6 - 1440*p**4*q2*q3*q4**2*q5**3 + 180*p**4*q2*q3*q4**2*q5**2*q6**2 - 1440*p**4*q2*q3*q4**2*q5**2*q6 + 5040*p**4*q2*q3*q4**2*q5**2 + 180*p**4*q2*q3*q4**2*q5*q6**3 - 1440*p**4*q2*q3*q4**2*q5*q6**2 + 5040*p**4*q2*q3*q4**2*q5*q6 - 10080*p**4*q2*q3*q4**2*q5 + 180*p**4*q2*q3*q4*q5**5 + 180*p**4*q2*q3*q4*q5**4*q6 - 1440*p**4*q2*q3*q4*q5**4 + 180*p**4*q2*q3*q4*q5**3*q6**2 - 1440*p**4*q2*q3*q4*q5**3*q6 + 5040*p**4*q2*q3*q4*q5**3 + 180*p**4*q2*q3*q4*q5**2*q6**3 - 1440*p**4*q2*q3*q4*q5**2*q6**2 + 5040*p**4*q2*q3*q4*q5**2*q6 - 10080*p**4*q2*q3*q4*q5**2 + 180*p**4*q2*q3*q4*q5*q6**4 - 1440*p**4*q2*q3*q4*q5*q6**3 + 5040*p**4*q2*q3*q4*q5*q6**2 - 10080*p**4*q2*q3*q4*q5*q6 + 12600*p**4*q2*q3*q4*q5 - 360*p**4*q3**5*q4*q5 - 360*p**4*q3**4*q4**2*q5 - 360*p**4*q3**4*q4*q5**2 - 360*p**4*q3**4*q4*q5*q6 + 2520*p**4*q3**4*q4*q5 - 360*p**4*q3**3*q4**3*q5 - 360*p**4*q3**3*q4**2*q5**2 - 360*p**4*q3**3*q4**2*q5*q6 + 2520*p**4*q3**3*q4**2*q5 - 360*p**4*q3**3*q4*q5**3 - 360*p**4*q3**3*q4*q5**2*q6 + 2520*p**4*q3**3*q4*q5**2 - 360*p**4*q3**3*q4*q5*q6**2 + 2520*p**4*q3**3*q4*q5*q6 - 7560*p**4*q3**3*q4*q5 - 360*p**4*q3**2*q4**4*q5 - 360*p**4*q3**2*q4**3*q5**2 - 360*p**4*q3**2*q4**3*q5*q6 + 2520*p**4*q3**2*q4**3*q5 - 360*p**4*q3**2*q4**2*q5**3 - 360*p**4*q3**2*q4**2*q5**2*q6 + 2520*p**4*q3**2*q4**2*q5**2 - 360*p**4*q3**2*q4**2*q5*q6**2 + 2520*p**4*q3**2*q4**2*q5*q6 - 7560*p**4*q3**2*q4**2*q5 - 360*p**4*q3**2*q4*q5**4 - 360*p**4*q3**2*q4*q5**3*q6 + 2520*p**4*q3**2*q4*q5**3 - 360*p**4*q3**2*q4*q5**2*q6**2 + 2520*p**4*q3**2*q4*q5**2*q6 - 7560*p**4*q3**2*q4*q5**2 - 360*p**4*q3**2*q4*q5*q6**3 + 2520*p**4*q3**2*q4*q5*q6**2 - 7560*p**4*q3**2*q4*q5*q6 + 12600*p**4*q3**2*q4*q5 - 360*p**4*q3*q4**5*q5 - 360*p**4*q3*q4**4*q5**2 - 360*p**4*q3*q4**4*q5*q6 + 2520*p**4*q3*q4**4*q5 - 360*p**4*q3*q4**3*q5**3 - 360*p**4*q3*q4**3*q5**2*q6 + 2520*p**4*q3*q4**3*q5**2 - 360*p**4*q3*q4**3*q5*q6**2 + 2520*p**4*q3*q4**3*q5*q6 - 7560*p**4*q3*q4**3*q5 - 360*p**4*q3*q4**2*q5**4 - 360*p**4*q3*q4**2*q5**3*q6 + 2520*p**4*q3*q4**2*q5**3 - 360*p**4*q3*q4**2*q5**2*q6**2 + 2520*p**4*q3*q4**2*q5**2*q6 - 7560*p**4*q3*q4**2*q5**2 - 360*p**4*q3*q4**2*q5*q6**3 + 2520*p**4*q3*q4**2*q5*q6**2 - 7560*p**4*q3*q4**2*q5*q6 + 12600*p**4*q3*q4**2*q5 - 360*p**4*q3*q4*q5**5 - 360*p**4*q3*q4*q5**4*q6 + 2520*p**4*q3*q4*q5**4 - 360*p**4*q3*q4*q5**3*q6**2 + 2520*p**4*q3*q4*q5**3*q6 - 7560*p**4*q3*q4*q5**3 - 360*p**4*q3*q4*q5**2*q6**3 + 2520*p**4*q3*q4*q5**2*q6**2 - 7560*p**4*q3*q4*q5**2*q6 + 12600*p**4*q3*q4*q5**2 - 360*p**4*q3*q4*q5*q6**4 + 2520*p**4*q3*q4*q5*q6**3 - 7560*p**4*q3*q4*q5*q6**2 + 12600*p**4*q3*q4*q5*q6 - 12600*p**4*q3*q4*q5 + 420*p**4*q4**5*q5 + 420*p**4*q4**4*q5**2 + 420*p**4*q4**4*q5*q6 - 2520*p**4*q4**4*q5 + 420*p**4*q4**3*q5**3 + 420*p**4*q4**3*q5**2*q6 - 2520*p**4*q4**3*q5**2 + 420*p**4*q4**3*q5*q6**2 - 2520*p**4*q4**3*q5*q6 + 6300*p**4*q4**3*q5 + 420*p**4*q4**2*q5**4 + 420*p**4*q4**2*q5**3*q6 - 2520*p**4*q4**2*q5**3 + 420*p**4*q4**2*q5**2*q6**2 - 2520*p**4*q4**2*q5**2*q6 + 6300*p**4*q4**2*q5**2 + 420*p**4*q4**2*q5*q6**3 - 2520*p**4*q4**2*q5*q6**2 + 6300*p**4*q4**2*q5*q6 - 8400*p**4*q4**2*q5 + 420*p**4*q4*q5**5 + 420*p**4*q4*q5**4*q6 - 2520*p**4*q4*q5**4 + 420*p**4*q4*q5**3*q6**2 - 2520*p**4*q4*q5**3*q6 + 6300*p**4*q4*q5**3 + 420*p**4*q4*q5**2*q6**3 - 2520*p**4*q4*q5**2*q6**2 + 6300*p**4*q4*q5**2*q6 - 8400*p**4*q4*q5**2 + 420*p**4*q4*q5*q6**4 - 2520*p**4*q4*q5*q6**3 + 6300*p**4*q4*q5*q6**2 - 8400*p**4*q4*q5*q6 + 6300*p**4*q4*q5 - 252*p**4*q5**5 - 252*p**4*q5**4*q6 + 1260*p**4*q5**4 - 252*p**4*q5**3*q6**2 + 1260*p**4*q5**3*q6 - 2520*p**4*q5**3 - 252*p**4*q5**2*q6**3 + 1260*p**4*q5**2*q6**2 - 2520*p**4*q5**2*q6 + 2520*p**4*q5**2 - 252*p**4*q5*q6**4 + 1260*p**4*q5*q6**3 - 2520*p**4*q5*q6**2 + 2520*p**4*q5*q6 - 1260*p**4*q5 + 100*p**3*q1**5*q2*q3*q4*q5 + 100*p**3*q1**4*q2**2*q3*q4*q5 + 100*p**3*q1**4*q2*q3**2*q4*q5 + 100*p**3*q1**4*q2*q3*q4**2*q5 + 100*p**3*q1**4*q2*q3*q4*q5**2 + 100*p**3*q1**4*q2*q3*q4*q5*q6 - 900*p**3*q1**4*q2*q3*q4*q5 + 100*p**3*q1**3*q2**3*q3*q4*q5 + 100*p**3*q1**3*q2**2*q3**2*q4*q5 + 100*p**3*q1**3*q2**2*q3*q4**2*q5 + 100*p**3*q1**3*q2**2*q3*q4*q5**2 + 100*p**3*q1**3*q2**2*q3*q4*q5*q6 - 900*p**3*q1**3*q2**2*q3*q4*q5 + 100*p**3*q1**3*q2*q3**3*q4*q5 + 100*p**3*q1**3*q2*q3**2*q4**2*q5 + 100*p**3*q1**3*q2*q3**2*q4*q5**2 + 100*p**3*q1**3*q2*q3**2*q4*q5*q6 - 900*p**3*q1**3*q2*q3**2*q4*q5 + 100*p**3*q1**3*q2*q3*q4**3*q5 + 100*p**3*q1**3*q2*q3*q4**2*q5**2 + 100*p**3*q1**3*q2*q3*q4**2*q5*q6 - 900*p**3*q1**3*q2*q3*q4**2*q5 + 100*p**3*q1**3*q2*q3*q4*q5**3 + 100*p**3*q1**3*q2*q3*q4*q5**2*q6 - 900*p**3*q1**3*q2*q3*q4*q5**2 + 100*p**3*q1**3*q2*q3*q4*q5*q6**2 - 900*p**3*q1**3*q2*q3*q4*q5*q6 + 3600*p**3*q1**3*q2*q3*q4*q5 + 100*p**3*q1**2*q2**4*q3*q4*q5 + 100*p**3*q1**2*q2**3*q3**2*q4*q5 + 100*p**3*q1**2*q2**3*q3*q4**2*q5 + 100*p**3*q1**2*q2**3*q3*q4*q5**2 + 100*p**3*q1**2*q2**3*q3*q4*q5*q6 - 900*p**3*q1**2*q2**3*q3*q4*q5 + 100*p**3*q1**2*q2**2*q3**3*q4*q5 + 100*p**3*q1**2*q2**2*q3**2*q4**2*q5 + 100*p**3*q1**2*q2**2*q3**2*q4*q5**2 + 100*p**3*q1**2*q2**2*q3**2*q4*q5*q6 - 900*p**3*q1**2*q2**2*q3**2*q4*q5 + 100*p**3*q1**2*q2**2*q3*q4**3*q5 + 100*p**3*q1**2*q2**2*q3*q4**2*q5**2 + 100*p**3*q1**2*q2**2*q3*q4**2*q5*q6 - 900*p**3*q1**2*q2**2*q3*q4**2*q5 + 100*p**3*q1**2*q2**2*q3*q4*q5**3 + 100*p**3*q1**2*q2**2*q3*q4*q5**2*q6 - 900*p**3*q1**2*q2**2*q3*q4*q5**2 + 100*p**3*q1**2*q2**2*q3*q4*q5*q6**2 - 900*p**3*q1**2*q2**2*q3*q4*q5*q6 + 3600*p**3*q1**2*q2**2*q3*q4*q5 + 100*p**3*q1**2*q2*q3**4*q4*q5 + 100*p**3*q1**2*q2*q3**3*q4**2*q5 + 100*p**3*q1**2*q2*q3**3*q4*q5**2 + 100*p**3*q1**2*q2*q3**3*q4*q5*q6 - 900*p**3*q1**2*q2*q3**3*q4*q5 + 100*p**3*q1**2*q2*q3**2*q4**3*q5 + 100*p**3*q1**2*q2*q3**2*q4**2*q5**2 + 100*p**3*q1**2*q2*q3**2*q4**2*q5*q6 - 900*p**3*q1**2*q2*q3**2*q4**2*q5 + 100*p**3*q1**2*q2*q3**2*q4*q5**3 + 100*p**3*q1**2*q2*q3**2*q4*q5**2*q6 - 900*p**3*q1**2*q2*q3**2*q4*q5**2 + 100*p**3*q1**2*q2*q3**2*q4*q5*q6**2 - 900*p**3*q1**2*q2*q3**2*q4*q5*q6 + 3600*p**3*q1**2*q2*q3**2*q4*q5 + 100*p**3*q1**2*q2*q3*q4**4*q5 + 100*p**3*q1**2*q2*q3*q4**3*q5**2 + 100*p**3*q1**2*q2*q3*q4**3*q5*q6 - 900*p**3*q1**2*q2*q3*q4**3*q5 + 100*p**3*q1**2*q2*q3*q4**2*q5**3 + 100*p**3*q1**2*q2*q3*q4**2*q5**2*q6 - 900*p**3*q1**2*q2*q3*q4**2*q5**2 + 100*p**3*q1**2*q2*q3*q4**2*q5*q6**2 - 900*p**3*q1**2*q2*q3*q4**2*q5*q6 + 3600*p**3*q1**2*q2*q3*q4**2*q5 + 100*p**3*q1**2*q2*q3*q4*q5**4 + 100*p**3*q1**2*q2*q3*q4*q5**3*q6 - 900*p**3*q1**2*q2*q3*q4*q5**3 + 100*p**3*q1**2*q2*q3*q4*q5**2*q6**2 - 900*p**3*q1**2*q2*q3*q4*q5**2*q6 + 3600*p**3*q1**2*q2*q3*q4*q5**2 + 100*p**3*q1**2*q2*q3*q4*q5*q6**3 - 900*p**3*q1**2*q2*q3*q4*q5*q6**2 + 3600*p**3*q1**2*q2*q3*q4*q5*q6 - 8400*p**3*q1**2*q2*q3*q4*q5 + 100*p**3*q1*q2**5*q3*q4*q5 + 100*p**3*q1*q2**4*q3**2*q4*q5 + 100*p**3*q1*q2**4*q3*q4**2*q5 + 100*p**3*q1*q2**4*q3*q4*q5**2 + 100*p**3*q1*q2**4*q3*q4*q5*q6 - 900*p**3*q1*q2**4*q3*q4*q5 + 100*p**3*q1*q2**3*q3**3*q4*q5 + 100*p**3*q1*q2**3*q3**2*q4**2*q5 + 100*p**3*q1*q2**3*q3**2*q4*q5**2 + 100*p**3*q1*q2**3*q3**2*q4*q5*q6 - 900*p**3*q1*q2**3*q3**2*q4*q5 + 100*p**3*q1*q2**3*q3*q4**3*q5 + 100*p**3*q1*q2**3*q3*q4**2*q5**2 + 100*p**3*q1*q2**3*q3*q4**2*q5*q6 - 900*p**3*q1*q2**3*q3*q4**2*q5 + 100*p**3*q1*q2**3*q3*q4*q5**3 + 100*p**3*q1*q2**3*q3*q4*q5**2*q6 - 900*p**3*q1*q2**3*q3*q4*q5**2 + 100*p**3*q1*q2**3*q3*q4*q5*q6**2 - 900*p**3*q1*q2**3*q3*q4*q5*q6 + 3600*p**3*q1*q2**3*q3*q4*q5 + 100*p**3*q1*q2**2*q3**4*q4*q5 + 100*p**3*q1*q2**2*q3**3*q4**2*q5 + 100*p**3*q1*q2**2*q3**3*q4*q5**2 + 100*p**3*q1*q2**2*q3**3*q4*q5*q6 - 900*p**3*q1*q2**2*q3**3*q4*q5 + 100*p**3*q1*q2**2*q3**2*q4**3*q5 + 100*p**3*q1*q2**2*q3**2*q4**2*q5**2 + 100*p**3*q1*q2**2*q3**2*q4**2*q5*q6 - 900*p**3*q1*q2**2*q3**2*q4**2*q5 + 100*p**3*q1*q2**2*q3**2*q4*q5**3 + 100*p**3*q1*q2**2*q3**2*q4*q5**2*q6 - 900*p**3*q1*q2**2*q3**2*q4*q5**2 + 100*p**3*q1*q2**2*q3**2*q4*q5*q6**2 - 900*p**3*q1*q2**2*q3**2*q4*q5*q6 + 3600*p**3*q1*q2**2*q3**2*q4*q5 + 100*p**3*q1*q2**2*q3*q4**4*q5 + 100*p**3*q1*q2**2*q3*q4**3*q5**2 + 100*p**3*q1*q2**2*q3*q4**3*q5*q6 - 900*p**3*q1*q2**2*q3*q4**3*q5 + 100*p**3*q1*q2**2*q3*q4**2*q5**3 + 100*p**3*q1*q2**2*q3*q4**2*q5**2*q6 - 900*p**3*q1*q2**2*q3*q4**2*q5**2 + 100*p**3*q1*q2**2*q3*q4**2*q5*q6**2 - 900*p**3*q1*q2**2*q3*q4**2*q5*q6 + 3600*p**3*q1*q2**2*q3*q4**2*q5 + 100*p**3*q1*q2**2*q3*q4*q5**4 + 100*p**3*q1*q2**2*q3*q4*q5**3*q6 - 900*p**3*q1*q2**2*q3*q4*q5**3 + 100*p**3*q1*q2**2*q3*q4*q5**2*q6**2 - 900*p**3*q1*q2**2*q3*q4*q5**2*q6 + 3600*p**3*q1*q2**2*q3*q4*q5**2 + 100*p**3*q1*q2**2*q3*q4*q5*q6**3 - 900*p**3*q1*q2**2*q3*q4*q5*q6**2 + 3600*p**3*q1*q2**2*q3*q4*q5*q6 - 8400*p**3*q1*q2**2*q3*q4*q5 + 100*p**3*q1*q2*q3**5*q4*q5 + 100*p**3*q1*q2*q3**4*q4**2*q5 + 100*p**3*q1*q2*q3**4*q4*q5**2 + 100*p**3*q1*q2*q3**4*q4*q5*q6 - 900*p**3*q1*q2*q3**4*q4*q5 + 100*p**3*q1*q2*q3**3*q4**3*q5 + 100*p**3*q1*q2*q3**3*q4**2*q5**2 + 100*p**3*q1*q2*q3**3*q4**2*q5*q6 - 900*p**3*q1*q2*q3**3*q4**2*q5 + 100*p**3*q1*q2*q3**3*q4*q5**3 + 100*p**3*q1*q2*q3**3*q4*q5**2*q6 - 900*p**3*q1*q2*q3**3*q4*q5**2 + 100*p**3*q1*q2*q3**3*q4*q5*q6**2 - 900*p**3*q1*q2*q3**3*q4*q5*q6 + 3600*p**3*q1*q2*q3**3*q4*q5 + 100*p**3*q1*q2*q3**2*q4**4*q5 + 100*p**3*q1*q2*q3**2*q4**3*q5**2 + 100*p**3*q1*q2*q3**2*q4**3*q5*q6 - 900*p**3*q1*q2*q3**2*q4**3*q5 + 100*p**3*q1*q2*q3**2*q4**2*q5**3 + 100*p**3*q1*q2*q3**2*q4**2*q5**2*q6 - 900*p**3*q1*q2*q3**2*q4**2*q5**2 + 100*p**3*q1*q2*q3**2*q4**2*q5*q6**2 - 900*p**3*q1*q2*q3**2*q4**2*q5*q6 + 3600*p**3*q1*q2*q3**2*q4**2*q5 + 100*p**3*q1*q2*q3**2*q4*q5**4 + 100*p**3*q1*q2*q3**2*q4*q5**3*q6 - 900*p**3*q1*q2*q3**2*q4*q5**3 + 100*p**3*q1*q2*q3**2*q4*q5**2*q6**2 - 900*p**3*q1*q2*q3**2*q4*q5**2*q6 + 3600*p**3*q1*q2*q3**2*q4*q5**2 + 100*p**3*q1*q2*q3**2*q4*q5*q6**3 - 900*p**3*q1*q2*q3**2*q4*q5*q6**2 + 3600*p**3*q1*q2*q3**2*q4*q5*q6 - 8400*p**3*q1*q2*q3**2*q4*q5 + 100*p**3*q1*q2*q3*q4**5*q5 + 100*p**3*q1*q2*q3*q4**4*q5**2 + 100*p**3*q1*q2*q3*q4**4*q5*q6 - 900*p**3*q1*q2*q3*q4**4*q5 + 100*p**3*q1*q2*q3*q4**3*q5**3 + 100*p**3*q1*q2*q3*q4**3*q5**2*q6 - 900*p**3*q1*q2*q3*q4**3*q5**2 + 100*p**3*q1*q2*q3*q4**3*q5*q6**2 - 900*p**3*q1*q2*q3*q4**3*q5*q6 + 3600*p**3*q1*q2*q3*q4**3*q5 + 100*p**3*q1*q2*q3*q4**2*q5**4 + 100*p**3*q1*q2*q3*q4**2*q5**3*q6 - 900*p**3*q1*q2*q3*q4**2*q5**3 + 100*p**3*q1*q2*q3*q4**2*q5**2*q6**2 - 900*p**3*q1*q2*q3*q4**2*q5**2*q6 + 3600*p**3*q1*q2*q3*q4**2*q5**2 + 100*p**3*q1*q2*q3*q4**2*q5*q6**3 - 900*p**3*q1*q2*q3*q4**2*q5*q6**2 + 3600*p**3*q1*q2*q3*q4**2*q5*q6 - 8400*p**3*q1*q2*q3*q4**2*q5 + 100*p**3*q1*q2*q3*q4*q5**5 + 100*p**3*q1*q2*q3*q4*q5**4*q6 - 900*p**3*q1*q2*q3*q4*q5**4 + 100*p**3*q1*q2*q3*q4*q5**3*q6**2 - 900*p**3*q1*q2*q3*q4*q5**3*q6 + 3600*p**3*q1*q2*q3*q4*q5**3 + 100*p**3*q1*q2*q3*q4*q5**2*q6**3 - 900*p**3*q1*q2*q3*q4*q5**2*q6**2 + 3600*p**3*q1*q2*q3*q4*q5**2*q6 - 8400*p**3*q1*q2*q3*q4*q5**2 + 100*p**3*q1*q2*q3*q4*q5*q6**4 - 900*p**3*q1*q2*q3*q4*q5*q6**3 + 3600*p**3*q1*q2*q3*q4*q5*q6**2 - 8400*p**3*q1*q2*q3*q4*q5*q6 + 12600*p**3*q1*q2*q3*q4*q5 - 270*p**3*q2**5*q3*q4*q5 - 270*p**3*q2**4*q3**2*q4*q5 - 270*p**3*q2**4*q3*q4**2*q5 - 270*p**3*q2**4*q3*q4*q5**2 - 270*p**3*q2**4*q3*q4*q5*q6 + 2160*p**3*q2**4*q3*q4*q5 - 270*p**3*q2**3*q3**3*q4*q5 - 270*p**3*q2**3*q3**2*q4**2*q5 - 270*p**3*q2**3*q3**2*q4*q5**2 - 270*p**3*q2**3*q3**2*q4*q5*q6 + 2160*p**3*q2**3*q3**2*q4*q5 - 270*p**3*q2**3*q3*q4**3*q5 - 270*p**3*q2**3*q3*q4**2*q5**2 - 270*p**3*q2**3*q3*q4**2*q5*q6 + 2160*p**3*q2**3*q3*q4**2*q5 - 270*p**3*q2**3*q3*q4*q5**3 - 270*p**3*q2**3*q3*q4*q5**2*q6 + 2160*p**3*q2**3*q3*q4*q5**2 - 270*p**3*q2**3*q3*q4*q5*q6**2 + 2160*p**3*q2**3*q3*q4*q5*q6 - 7560*p**3*q2**3*q3*q4*q5 - 270*p**3*q2**2*q3**4*q4*q5 - 270*p**3*q2**2*q3**3*q4**2*q5 - 270*p**3*q2**2*q3**3*q4*q5**2 - 270*p**3*q2**2*q3**3*q4*q5*q6 + 2160*p**3*q2**2*q3**3*q4*q5 - 270*p**3*q2**2*q3**2*q4**3*q5 - 270*p**3*q2**2*q3**2*q4**2*q5**2 - 270*p**3*q2**2*q3**2*q4**2*q5*q6 + 2160*p**3*q2**2*q3**2*q4**2*q5 - 270*p**3*q2**2*q3**2*q4*q5**3 - 270*p**3*q2**2*q3**2*q4*q5**2*q6 + 2160*p**3*q2**2*q3**2*q4*q5**2 - 270*p**3*q2**2*q3**2*q4*q5*q6**2 + 2160*p**3*q2**2*q3**2*q4*q5*q6 - 7560*p**3*q2**2*q3**2*q4*q5 - 270*p**3*q2**2*q3*q4**4*q5 - 270*p**3*q2**2*q3*q4**3*q5**2 - 270*p**3*q2**2*q3*q4**3*q5*q6 + 2160*p**3*q2**2*q3*q4**3*q5 - 270*p**3*q2**2*q3*q4**2*q5**3 - 270*p**3*q2**2*q3*q4**2*q5**2*q6 + 2160*p**3*q2**2*q3*q4**2*q5**2 - 270*p**3*q2**2*q3*q4**2*q5*q6**2 + 2160*p**3*q2**2*q3*q4**2*q5*q6 - 7560*p**3*q2**2*q3*q4**2*q5 - 270*p**3*q2**2*q3*q4*q5**4 - 270*p**3*q2**2*q3*q4*q5**3*q6 + 2160*p**3*q2**2*q3*q4*q5**3 - 270*p**3*q2**2*q3*q4*q5**2*q6**2 + 2160*p**3*q2**2*q3*q4*q5**2*q6 - 7560*p**3*q2**2*q3*q4*q5**2 - 270*p**3*q2**2*q3*q4*q5*q6**3 + 2160*p**3*q2**2*q3*q4*q5*q6**2 - 7560*p**3*q2**2*q3*q4*q5*q6 + 15120*p**3*q2**2*q3*q4*q5 - 270*p**3*q2*q3**5*q4*q5 - 270*p**3*q2*q3**4*q4**2*q5 - 270*p**3*q2*q3**4*q4*q5**2 - 270*p**3*q2*q3**4*q4*q5*q6 + 2160*p**3*q2*q3**4*q4*q5 - 270*p**3*q2*q3**3*q4**3*q5 - 270*p**3*q2*q3**3*q4**2*q5**2 - 270*p**3*q2*q3**3*q4**2*q5*q6 + 2160*p**3*q2*q3**3*q4**2*q5 - 270*p**3*q2*q3**3*q4*q5**3 - 270*p**3*q2*q3**3*q4*q5**2*q6 + 2160*p**3*q2*q3**3*q4*q5**2 - 270*p**3*q2*q3**3*q4*q5*q6**2 + 2160*p**3*q2*q3**3*q4*q5*q6 - 7560*p**3*q2*q3**3*q4*q5 - 270*p**3*q2*q3**2*q4**4*q5 - 270*p**3*q2*q3**2*q4**3*q5**2 - 270*p**3*q2*q3**2*q4**3*q5*q6 + 2160*p**3*q2*q3**2*q4**3*q5 - 270*p**3*q2*q3**2*q4**2*q5**3 - 270*p**3*q2*q3**2*q4**2*q5**2*q6 + 2160*p**3*q2*q3**2*q4**2*q5**2 - 270*p**3*q2*q3**2*q4**2*q5*q6**2 + 2160*p**3*q2*q3**2*q4**2*q5*q6 - 7560*p**3*q2*q3**2*q4**2*q5 - 270*p**3*q2*q3**2*q4*q5**4 - 270*p**3*q2*q3**2*q4*q5**3*q6 + 2160*p**3*q2*q3**2*q4*q5**3 - 270*p**3*q2*q3**2*q4*q5**2*q6**2 + 2160*p**3*q2*q3**2*q4*q5**2*q6 - 7560*p**3*q2*q3**2*q4*q5**2 - 270*p**3*q2*q3**2*q4*q5*q6**3 + 2160*p**3*q2*q3**2*q4*q5*q6**2 - 7560*p**3*q2*q3**2*q4*q5*q6 + 15120*p**3*q2*q3**2*q4*q5 - 270*p**3*q2*q3*q4**5*q5 - 270*p**3*q2*q3*q4**4*q5**2 - 270*p**3*q2*q3*q4**4*q5*q6 + 2160*p**3*q2*q3*q4**4*q5 - 270*p**3*q2*q3*q4**3*q5**3 - 270*p**3*q2*q3*q4**3*q5**2*q6 + 2160*p**3*q2*q3*q4**3*q5**2 - 270*p**3*q2*q3*q4**3*q5*q6**2 + 2160*p**3*q2*q3*q4**3*q5*q6 - 7560*p**3*q2*q3*q4**3*q5 - 270*p**3*q2*q3*q4**2*q5**4 - 270*p**3*q2*q3*q4**2*q5**3*q6 + 2160*p**3*q2*q3*q4**2*q5**3 - 270*p**3*q2*q3*q4**2*q5**2*q6**2 + 2160*p**3*q2*q3*q4**2*q5**2*q6 - 7560*p**3*q2*q3*q4**2*q5**2 - 270*p**3*q2*q3*q4**2*q5*q6**3 + 2160*p**3*q2*q3*q4**2*q5*q6**2 - 7560*p**3*q2*q3*q4**2*q5*q6 + 15120*p**3*q2*q3*q4**2*q5 - 270*p**3*q2*q3*q4*q5**5 - 270*p**3*q2*q3*q4*q5**4*q6 + 2160*p**3*q2*q3*q4*q5**4 - 270*p**3*q2*q3*q4*q5**3*q6**2 + 2160*p**3*q2*q3*q4*q5**3*q6 - 7560*p**3*q2*q3*q4*q5**3 - 270*p**3*q2*q3*q4*q5**2*q6**3 + 2160*p**3*q2*q3*q4*q5**2*q6**2 - 7560*p**3*q2*q3*q4*q5**2*q6 + 15120*p**3*q2*q3*q4*q5**2 - 270*p**3*q2*q3*q4*q5*q6**4 + 2160*p**3*q2*q3*q4*q5*q6**3 - 7560*p**3*q2*q3*q4*q5*q6**2 + 15120*p**3*q2*q3*q4*q5*q6 - 18900*p**3*q2*q3*q4*q5 + 360*p**3*q3**5*q4*q5 + 360*p**3*q3**4*q4**2*q5 + 360*p**3*q3**4*q4*q5**2 + 360*p**3*q3**4*q4*q5*q6 - 2520*p**3*q3**4*q4*q5 + 360*p**3*q3**3*q4**3*q5 + 360*p**3*q3**3*q4**2*q5**2 + 360*p**3*q3**3*q4**2*q5*q6 - 2520*p**3*q3**3*q4**2*q5 + 360*p**3*q3**3*q4*q5**3 + 360*p**3*q3**3*q4*q5**2*q6 - 2520*p**3*q3**3*q4*q5**2 + 360*p**3*q3**3*q4*q5*q6**2 - 2520*p**3*q3**3*q4*q5*q6 + 7560*p**3*q3**3*q4*q5 + 360*p**3*q3**2*q4**4*q5 + 360*p**3*q3**2*q4**3*q5**2 + 360*p**3*q3**2*q4**3*q5*q6 - 2520*p**3*q3**2*q4**3*q5 + 360*p**3*q3**2*q4**2*q5**3 + 360*p**3*q3**2*q4**2*q5**2*q6 - 2520*p**3*q3**2*q4**2*q5**2 + 360*p**3*q3**2*q4**2*q5*q6**2 - 2520*p**3*q3**2*q4**2*q5*q6 + 7560*p**3*q3**2*q4**2*q5 + 360*p**3*q3**2*q4*q5**4 + 360*p**3*q3**2*q4*q5**3*q6 - 2520*p**3*q3**2*q4*q5**3 + 360*p**3*q3**2*q4*q5**2*q6**2 - 2520*p**3*q3**2*q4*q5**2*q6 + 7560*p**3*q3**2*q4*q5**2 + 360*p**3*q3**2*q4*q5*q6**3 - 2520*p**3*q3**2*q4*q5*q6**2 + 7560*p**3*q3**2*q4*q5*q6 - 12600*p**3*q3**2*q4*q5 + 360*p**3*q3*q4**5*q5 + 360*p**3*q3*q4**4*q5**2 + 360*p**3*q3*q4**4*q5*q6 - 2520*p**3*q3*q4**4*q5 + 360*p**3*q3*q4**3*q5**3 + 360*p**3*q3*q4**3*q5**2*q6 - 2520*p**3*q3*q4**3*q5**2 + 360*p**3*q3*q4**3*q5*q6**2 - 2520*p**3*q3*q4**3*q5*q6 + 7560*p**3*q3*q4**3*q5 + 360*p**3*q3*q4**2*q5**4 + 360*p**3*q3*q4**2*q5**3*q6 - 2520*p**3*q3*q4**2*q5**3 + 360*p**3*q3*q4**2*q5**2*q6**2 - 2520*p**3*q3*q4**2*q5**2*q6 + 7560*p**3*q3*q4**2*q5**2 + 360*p**3*q3*q4**2*q5*q6**3 - 2520*p**3*q3*q4**2*q5*q6**2 + 7560*p**3*q3*q4**2*q5*q6 - 12600*p**3*q3*q4**2*q5 + 360*p**3*q3*q4*q5**5 + 360*p**3*q3*q4*q5**4*q6 - 2520*p**3*q3*q4*q5**4 + 360*p**3*q3*q4*q5**3*q6**2 - 2520*p**3*q3*q4*q5**3*q6 + 7560*p**3*q3*q4*q5**3 + 360*p**3*q3*q4*q5**2*q6**3 - 2520*p**3*q3*q4*q5**2*q6**2 + 7560*p**3*q3*q4*q5**2*q6 - 12600*p**3*q3*q4*q5**2 + 360*p**3*q3*q4*q5*q6**4 - 2520*p**3*q3*q4*q5*q6**3 + 7560*p**3*q3*q4*q5*q6**2 - 12600*p**3*q3*q4*q5*q6 + 12600*p**3*q3*q4*q5 - 210*p**3*q4**5*q5 - 210*p**3*q4**4*q5**2 - 210*p**3*q4**4*q5*q6 + 1260*p**3*q4**4*q5 - 210*p**3*q4**3*q5**3 - 210*p**3*q4**3*q5**2*q6 + 1260*p**3*q4**3*q5**2 - 210*p**3*q4**3*q5*q6**2 + 1260*p**3*q4**3*q5*q6 - 3150*p**3*q4**3*q5 - 210*p**3*q4**2*q5**4 - 210*p**3*q4**2*q5**3*q6 + 1260*p**3*q4**2*q5**3 - 210*p**3*q4**2*q5**2*q6**2 + 1260*p**3*q4**2*q5**2*q6 - 3150*p**3*q4**2*q5**2 - 210*p**3*q4**2*q5*q6**3 + 1260*p**3*q4**2*q5*q6**2 - 3150*p**3*q4**2*q5*q6 + 4200*p**3*q4**2*q5 - 210*p**3*q4*q5**5 - 210*p**3*q4*q5**4*q6 + 1260*p**3*q4*q5**4 - 210*p**3*q4*q5**3*q6**2 + 1260*p**3*q4*q5**3*q6 - 3150*p**3*q4*q5**3 - 210*p**3*q4*q5**2*q6**3 + 1260*p**3*q4*q5**2*q6**2 - 3150*p**3*q4*q5**2*q6 + 4200*p**3*q4*q5**2 - 210*p**3*q4*q5*q6**4 + 1260*p**3*q4*q5*q6**3 - 3150*p**3*q4*q5*q6**2 + 4200*p**3*q4*q5*q6 - 3150*p**3*q4*q5 - 100*p**2*q1**5*q2*q3*q4*q5 - 100*p**2*q1**4*q2**2*q3*q4*q5 - 100*p**2*q1**4*q2*q3**2*q4*q5 - 100*p**2*q1**4*q2*q3*q4**2*q5 - 100*p**2*q1**4*q2*q3*q4*q5**2 - 100*p**2*q1**4*q2*q3*q4*q5*q6 + 900*p**2*q1**4*q2*q3*q4*q5 - 100*p**2*q1**3*q2**3*q3*q4*q5 - 100*p**2*q1**3*q2**2*q3**2*q4*q5 - 100*p**2*q1**3*q2**2*q3*q4**2*q5 - 100*p**2*q1**3*q2**2*q3*q4*q5**2 - 100*p**2*q1**3*q2**2*q3*q4*q5*q6 + 900*p**2*q1**3*q2**2*q3*q4*q5 - 100*p**2*q1**3*q2*q3**3*q4*q5 - 100*p**2*q1**3*q2*q3**2*q4**2*q5 - 100*p**2*q1**3*q2*q3**2*q4*q5**2 - 100*p**2*q1**3*q2*q3**2*q4*q5*q6 + 900*p**2*q1**3*q2*q3**2*q4*q5 - 100*p**2*q1**3*q2*q3*q4**3*q5 - 100*p**2*q1**3*q2*q3*q4**2*q5**2 - 100*p**2*q1**3*q2*q3*q4**2*q5*q6 + 900*p**2*q1**3*q2*q3*q4**2*q5 - 100*p**2*q1**3*q2*q3*q4*q5**3 - 100*p**2*q1**3*q2*q3*q4*q5**2*q6 + 900*p**2*q1**3*q2*q3*q4*q5**2 - 100*p**2*q1**3*q2*q3*q4*q5*q6**2 + 900*p**2*q1**3*q2*q3*q4*q5*q6 - 3600*p**2*q1**3*q2*q3*q4*q5 - 100*p**2*q1**2*q2**4*q3*q4*q5 - 100*p**2*q1**2*q2**3*q3**2*q4*q5 - 100*p**2*q1**2*q2**3*q3*q4**2*q5 - 100*p**2*q1**2*q2**3*q3*q4*q5**2 - 100*p**2*q1**2*q2**3*q3*q4*q5*q6 + 900*p**2*q1**2*q2**3*q3*q4*q5 - 100*p**2*q1**2*q2**2*q3**3*q4*q5 - 100*p**2*q1**2*q2**2*q3**2*q4**2*q5 - 100*p**2*q1**2*q2**2*q3**2*q4*q5**2 - 100*p**2*q1**2*q2**2*q3**2*q4*q5*q6 + 900*p**2*q1**2*q2**2*q3**2*q4*q5 - 100*p**2*q1**2*q2**2*q3*q4**3*q5 - 100*p**2*q1**2*q2**2*q3*q4**2*q5**2 - 100*p**2*q1**2*q2**2*q3*q4**2*q5*q6 + 900*p**2*q1**2*q2**2*q3*q4**2*q5 - 100*p**2*q1**2*q2**2*q3*q4*q5**3 - 100*p**2*q1**2*q2**2*q3*q4*q5**2*q6 + 900*p**2*q1**2*q2**2*q3*q4*q5**2 - 100*p**2*q1**2*q2**2*q3*q4*q5*q6**2 + 900*p**2*q1**2*q2**2*q3*q4*q5*q6 - 3600*p**2*q1**2*q2**2*q3*q4*q5 - 100*p**2*q1**2*q2*q3**4*q4*q5 - 100*p**2*q1**2*q2*q3**3*q4**2*q5 - 100*p**2*q1**2*q2*q3**3*q4*q5**2 - 100*p**2*q1**2*q2*q3**3*q4*q5*q6 + 900*p**2*q1**2*q2*q3**3*q4*q5 - 100*p**2*q1**2*q2*q3**2*q4**3*q5 - 100*p**2*q1**2*q2*q3**2*q4**2*q5**2 - 100*p**2*q1**2*q2*q3**2*q4**2*q5*q6 + 900*p**2*q1**2*q2*q3**2*q4**2*q5 - 100*p**2*q1**2*q2*q3**2*q4*q5**3 - 100*p**2*q1**2*q2*q3**2*q4*q5**2*q6 + 900*p**2*q1**2*q2*q3**2*q4*q5**2 - 100*p**2*q1**2*q2*q3**2*q4*q5*q6**2 + 900*p**2*q1**2*q2*q3**2*q4*q5*q6 - 3600*p**2*q1**2*q2*q3**2*q4*q5 - 100*p**2*q1**2*q2*q3*q4**4*q5 - 100*p**2*q1**2*q2*q3*q4**3*q5**2 - 100*p**2*q1**2*q2*q3*q4**3*q5*q6 + 900*p**2*q1**2*q2*q3*q4**3*q5 - 100*p**2*q1**2*q2*q3*q4**2*q5**3 - 100*p**2*q1**2*q2*q3*q4**2*q5**2*q6 + 900*p**2*q1**2*q2*q3*q4**2*q5**2 - 100*p**2*q1**2*q2*q3*q4**2*q5*q6**2 + 900*p**2*q1**2*q2*q3*q4**2*q5*q6 - 3600*p**2*q1**2*q2*q3*q4**2*q5 - 100*p**2*q1**2*q2*q3*q4*q5**4 - 100*p**2*q1**2*q2*q3*q4*q5**3*q6 + 900*p**2*q1**2*q2*q3*q4*q5**3 - 100*p**2*q1**2*q2*q3*q4*q5**2*q6**2 + 900*p**2*q1**2*q2*q3*q4*q5**2*q6 - 3600*p**2*q1**2*q2*q3*q4*q5**2 - 100*p**2*q1**2*q2*q3*q4*q5*q6**3 + 900*p**2*q1**2*q2*q3*q4*q5*q6**2 - 3600*p**2*q1**2*q2*q3*q4*q5*q6 + 8400*p**2*q1**2*q2*q3*q4*q5 - 100*p**2*q1*q2**5*q3*q4*q5 - 100*p**2*q1*q2**4*q3**2*q4*q5 - 100*p**2*q1*q2**4*q3*q4**2*q5 - 100*p**2*q1*q2**4*q3*q4*q5**2 - 100*p**2*q1*q2**4*q3*q4*q5*q6 + 900*p**2*q1*q2**4*q3*q4*q5 - 100*p**2*q1*q2**3*q3**3*q4*q5 - 100*p**2*q1*q2**3*q3**2*q4**2*q5 - 100*p**2*q1*q2**3*q3**2*q4*q5**2 - 100*p**2*q1*q2**3*q3**2*q4*q5*q6 + 900*p**2*q1*q2**3*q3**2*q4*q5 - 100*p**2*q1*q2**3*q3*q4**3*q5 - 100*p**2*q1*q2**3*q3*q4**2*q5**2 - 100*p**2*q1*q2**3*q3*q4**2*q5*q6 + 900*p**2*q1*q2**3*q3*q4**2*q5 - 100*p**2*q1*q2**3*q3*q4*q5**3 - 100*p**2*q1*q2**3*q3*q4*q5**2*q6 + 900*p**2*q1*q2**3*q3*q4*q5**2 - 100*p**2*q1*q2**3*q3*q4*q5*q6**2 + 900*p**2*q1*q2**3*q3*q4*q5*q6 - 3600*p**2*q1*q2**3*q3*q4*q5 - 100*p**2*q1*q2**2*q3**4*q4*q5 - 100*p**2*q1*q2**2*q3**3*q4**2*q5 - 100*p**2*q1*q2**2*q3**3*q4*q5**2 - 100*p**2*q1*q2**2*q3**3*q4*q5*q6 + 900*p**2*q1*q2**2*q3**3*q4*q5 - 100*p**2*q1*q2**2*q3**2*q4**3*q5 - 100*p**2*q1*q2**2*q3**2*q4**2*q5**2 - 100*p**2*q1*q2**2*q3**2*q4**2*q5*q6 + 900*p**2*q1*q2**2*q3**2*q4**2*q5 - 100*p**2*q1*q2**2*q3**2*q4*q5**3 - 100*p**2*q1*q2**2*q3**2*q4*q5**2*q6 + 900*p**2*q1*q2**2*q3**2*q4*q5**2 - 100*p**2*q1*q2**2*q3**2*q4*q5*q6**2 + 900*p**2*q1*q2**2*q3**2*q4*q5*q6 - 3600*p**2*q1*q2**2*q3**2*q4*q5 - 100*p**2*q1*q2**2*q3*q4**4*q5 - 100*p**2*q1*q2**2*q3*q4**3*q5**2 - 100*p**2*q1*q2**2*q3*q4**3*q5*q6 + 900*p**2*q1*q2**2*q3*q4**3*q5 - 100*p**2*q1*q2**2*q3*q4**2*q5**3 - 100*p**2*q1*q2**2*q3*q4**2*q5**2*q6 + 900*p**2*q1*q2**2*q3*q4**2*q5**2 - 100*p**2*q1*q2**2*q3*q4**2*q5*q6**2 + 900*p**2*q1*q2**2*q3*q4**2*q5*q6 - 3600*p**2*q1*q2**2*q3*q4**2*q5 - 100*p**2*q1*q2**2*q3*q4*q5**4 - 100*p**2*q1*q2**2*q3*q4*q5**3*q6 + 900*p**2*q1*q2**2*q3*q4*q5**3 - 100*p**2*q1*q2**2*q3*q4*q5**2*q6**2 + 900*p**2*q1*q2**2*q3*q4*q5**2*q6 - 3600*p**2*q1*q2**2*q3*q4*q5**2 - 100*p**2*q1*q2**2*q3*q4*q5*q6**3 + 900*p**2*q1*q2**2*q3*q4*q5*q6**2 - 3600*p**2*q1*q2**2*q3*q4*q5*q6 + 8400*p**2*q1*q2**2*q3*q4*q5 - 100*p**2*q1*q2*q3**5*q4*q5 - 100*p**2*q1*q2*q3**4*q4**2*q5 - 100*p**2*q1*q2*q3**4*q4*q5**2 - 100*p**2*q1*q2*q3**4*q4*q5*q6 + 900*p**2*q1*q2*q3**4*q4*q5 - 100*p**2*q1*q2*q3**3*q4**3*q5 - 100*p**2*q1*q2*q3**3*q4**2*q5**2 - 100*p**2*q1*q2*q3**3*q4**2*q5*q6 + 900*p**2*q1*q2*q3**3*q4**2*q5 - 100*p**2*q1*q2*q3**3*q4*q5**3 - 100*p**2*q1*q2*q3**3*q4*q5**2*q6 + 900*p**2*q1*q2*q3**3*q4*q5**2 - 100*p**2*q1*q2*q3**3*q4*q5*q6**2 + 900*p**2*q1*q2*q3**3*q4*q5*q6 - 3600*p**2*q1*q2*q3**3*q4*q5 - 100*p**2*q1*q2*q3**2*q4**4*q5 - 100*p**2*q1*q2*q3**2*q4**3*q5**2 - 100*p**2*q1*q2*q3**2*q4**3*q5*q6 + 900*p**2*q1*q2*q3**2*q4**3*q5 - 100*p**2*q1*q2*q3**2*q4**2*q5**3 - 100*p**2*q1*q2*q3**2*q4**2*q5**2*q6 + 900*p**2*q1*q2*q3**2*q4**2*q5**2 - 100*p**2*q1*q2*q3**2*q4**2*q5*q6**2 + 900*p**2*q1*q2*q3**2*q4**2*q5*q6 - 3600*p**2*q1*q2*q3**2*q4**2*q5 - 100*p**2*q1*q2*q3**2*q4*q5**4 - 100*p**2*q1*q2*q3**2*q4*q5**3*q6 + 900*p**2*q1*q2*q3**2*q4*q5**3 - 100*p**2*q1*q2*q3**2*q4*q5**2*q6**2 + 900*p**2*q1*q2*q3**2*q4*q5**2*q6 - 3600*p**2*q1*q2*q3**2*q4*q5**2 - 100*p**2*q1*q2*q3**2*q4*q5*q6**3 + 900*p**2*q1*q2*q3**2*q4*q5*q6**2 - 3600*p**2*q1*q2*q3**2*q4*q5*q6 + 8400*p**2*q1*q2*q3**2*q4*q5 - 100*p**2*q1*q2*q3*q4**5*q5 - 100*p**2*q1*q2*q3*q4**4*q5**2 - 100*p**2*q1*q2*q3*q4**4*q5*q6 + 900*p**2*q1*q2*q3*q4**4*q5 - 100*p**2*q1*q2*q3*q4**3*q5**3 - 100*p**2*q1*q2*q3*q4**3*q5**2*q6 + 900*p**2*q1*q2*q3*q4**3*q5**2 - 100*p**2*q1*q2*q3*q4**3*q5*q6**2 + 900*p**2*q1*q2*q3*q4**3*q5*q6 - 3600*p**2*q1*q2*q3*q4**3*q5 - 100*p**2*q1*q2*q3*q4**2*q5**4 - 100*p**2*q1*q2*q3*q4**2*q5**3*q6 + 900*p**2*q1*q2*q3*q4**2*q5**3 - 100*p**2*q1*q2*q3*q4**2*q5**2*q6**2 + 900*p**2*q1*q2*q3*q4**2*q5**2*q6 - 3600*p**2*q1*q2*q3*q4**2*q5**2 - 100*p**2*q1*q2*q3*q4**2*q5*q6**3 + 900*p**2*q1*q2*q3*q4**2*q5*q6**2 - 3600*p**2*q1*q2*q3*q4**2*q5*q6 + 8400*p**2*q1*q2*q3*q4**2*q5 - 100*p**2*q1*q2*q3*q4*q5**5 - 100*p**2*q1*q2*q3*q4*q5**4*q6 + 900*p**2*q1*q2*q3*q4*q5**4 - 100*p**2*q1*q2*q3*q4*q5**3*q6**2 + 900*p**2*q1*q2*q3*q4*q5**3*q6 - 3600*p**2*q1*q2*q3*q4*q5**3 - 100*p**2*q1*q2*q3*q4*q5**2*q6**3 + 900*p**2*q1*q2*q3*q4*q5**2*q6**2 - 3600*p**2*q1*q2*q3*q4*q5**2*q6 + 8400*p**2*q1*q2*q3*q4*q5**2 - 100*p**2*q1*q2*q3*q4*q5*q6**4 + 900*p**2*q1*q2*q3*q4*q5*q6**3 - 3600*p**2*q1*q2*q3*q4*q5*q6**2 + 8400*p**2*q1*q2*q3*q4*q5*q6 - 12600*p**2*q1*q2*q3*q4*q5 + 180*p**2*q2**5*q3*q4*q5 + 180*p**2*q2**4*q3**2*q4*q5 + 180*p**2*q2**4*q3*q4**2*q5 + 180*p**2*q2**4*q3*q4*q5**2 + 180*p**2*q2**4*q3*q4*q5*q6 - 1440*p**2*q2**4*q3*q4*q5 + 180*p**2*q2**3*q3**3*q4*q5 + 180*p**2*q2**3*q3**2*q4**2*q5 + 180*p**2*q2**3*q3**2*q4*q5**2 + 180*p**2*q2**3*q3**2*q4*q5*q6 - 1440*p**2*q2**3*q3**2*q4*q5 + 180*p**2*q2**3*q3*q4**3*q5 + 180*p**2*q2**3*q3*q4**2*q5**2 + 180*p**2*q2**3*q3*q4**2*q5*q6 - 1440*p**2*q2**3*q3*q4**2*q5 + 180*p**2*q2**3*q3*q4*q5**3 + 180*p**2*q2**3*q3*q4*q5**2*q6 - 1440*p**2*q2**3*q3*q4*q5**2 + 180*p**2*q2**3*q3*q4*q5*q6**2 - 1440*p**2*q2**3*q3*q4*q5*q6 + 5040*p**2*q2**3*q3*q4*q5 + 180*p**2*q2**2*q3**4*q4*q5 + 180*p**2*q2**2*q3**3*q4**2*q5 + 180*p**2*q2**2*q3**3*q4*q5**2 + 180*p**2*q2**2*q3**3*q4*q5*q6 - 1440*p**2*q2**2*q3**3*q4*q5 + 180*p**2*q2**2*q3**2*q4**3*q5 + 180*p**2*q2**2*q3**2*q4**2*q5**2 + 180*p**2*q2**2*q3**2*q4**2*q5*q6 - 1440*p**2*q2**2*q3**2*q4**2*q5 + 180*p**2*q2**2*q3**2*q4*q5**3 + 180*p**2*q2**2*q3**2*q4*q5**2*q6 - 1440*p**2*q2**2*q3**2*q4*q5**2 + 180*p**2*q2**2*q3**2*q4*q5*q6**2 - 1440*p**2*q2**2*q3**2*q4*q5*q6 + 5040*p**2*q2**2*q3**2*q4*q5 + 180*p**2*q2**2*q3*q4**4*q5 + 180*p**2*q2**2*q3*q4**3*q5**2 + 180*p**2*q2**2*q3*q4**3*q5*q6 - 1440*p**2*q2**2*q3*q4**3*q5 + 180*p**2*q2**2*q3*q4**2*q5**3 + 180*p**2*q2**2*q3*q4**2*q5**2*q6 - 1440*p**2*q2**2*q3*q4**2*q5**2 + 180*p**2*q2**2*q3*q4**2*q5*q6**2 - 1440*p**2*q2**2*q3*q4**2*q5*q6 + 5040*p**2*q2**2*q3*q4**2*q5 + 180*p**2*q2**2*q3*q4*q5**4 + 180*p**2*q2**2*q3*q4*q5**3*q6 - 1440*p**2*q2**2*q3*q4*q5**3 + 180*p**2*q2**2*q3*q4*q5**2*q6**2 - 1440*p**2*q2**2*q3*q4*q5**2*q6 + 5040*p**2*q2**2*q3*q4*q5**2 + 180*p**2*q2**2*q3*q4*q5*q6**3 - 1440*p**2*q2**2*q3*q4*q5*q6**2 + 5040*p**2*q2**2*q3*q4*q5*q6 - 10080*p**2*q2**2*q3*q4*q5 + 180*p**2*q2*q3**5*q4*q5 + 180*p**2*q2*q3**4*q4**2*q5 + 180*p**2*q2*q3**4*q4*q5**2 + 180*p**2*q2*q3**4*q4*q5*q6 - 1440*p**2*q2*q3**4*q4*q5 + 180*p**2*q2*q3**3*q4**3*q5 + 180*p**2*q2*q3**3*q4**2*q5**2 + 180*p**2*q2*q3**3*q4**2*q5*q6 - 1440*p**2*q2*q3**3*q4**2*q5 + 180*p**2*q2*q3**3*q4*q5**3 + 180*p**2*q2*q3**3*q4*q5**2*q6 - 1440*p**2*q2*q3**3*q4*q5**2 + 180*p**2*q2*q3**3*q4*q5*q6**2 - 1440*p**2*q2*q3**3*q4*q5*q6 + 5040*p**2*q2*q3**3*q4*q5 + 180*p**2*q2*q3**2*q4**4*q5 + 180*p**2*q2*q3**2*q4**3*q5**2 + 180*p**2*q2*q3**2*q4**3*q5*q6 - 1440*p**2*q2*q3**2*q4**3*q5 + 180*p**2*q2*q3**2*q4**2*q5**3 + 180*p**2*q2*q3**2*q4**2*q5**2*q6 - 1440*p**2*q2*q3**2*q4**2*q5**2 + 180*p**2*q2*q3**2*q4**2*q5*q6**2 - 1440*p**2*q2*q3**2*q4**2*q5*q6 + 5040*p**2*q2*q3**2*q4**2*q5 + 180*p**2*q2*q3**2*q4*q5**4 + 180*p**2*q2*q3**2*q4*q5**3*q6 - 1440*p**2*q2*q3**2*q4*q5**3 + 180*p**2*q2*q3**2*q4*q5**2*q6**2 - 1440*p**2*q2*q3**2*q4*q5**2*q6 + 5040*p**2*q2*q3**2*q4*q5**2 + 180*p**2*q2*q3**2*q4*q5*q6**3 - 1440*p**2*q2*q3**2*q4*q5*q6**2 + 5040*p**2*q2*q3**2*q4*q5*q6 - 10080*p**2*q2*q3**2*q4*q5 + 180*p**2*q2*q3*q4**5*q5 + 180*p**2*q2*q3*q4**4*q5**2 + 180*p**2*q2*q3*q4**4*q5*q6 - 1440*p**2*q2*q3*q4**4*q5 + 180*p**2*q2*q3*q4**3*q5**3 + 180*p**2*q2*q3*q4**3*q5**2*q6 - 1440*p**2*q2*q3*q4**3*q5**2 + 180*p**2*q2*q3*q4**3*q5*q6**2 - 1440*p**2*q2*q3*q4**3*q5*q6 + 5040*p**2*q2*q3*q4**3*q5 + 180*p**2*q2*q3*q4**2*q5**4 + 180*p**2*q2*q3*q4**2*q5**3*q6 - 1440*p**2*q2*q3*q4**2*q5**3 + 180*p**2*q2*q3*q4**2*q5**2*q6**2 - 1440*p**2*q2*q3*q4**2*q5**2*q6 + 5040*p**2*q2*q3*q4**2*q5**2 + 180*p**2*q2*q3*q4**2*q5*q6**3 - 1440*p**2*q2*q3*q4**2*q5*q6**2 + 5040*p**2*q2*q3*q4**2*q5*q6 - 10080*p**2*q2*q3*q4**2*q5 + 180*p**2*q2*q3*q4*q5**5 + 180*p**2*q2*q3*q4*q5**4*q6 - 1440*p**2*q2*q3*q4*q5**4 + 180*p**2*q2*q3*q4*q5**3*q6**2 - 1440*p**2*q2*q3*q4*q5**3*q6 + 5040*p**2*q2*q3*q4*q5**3 + 180*p**2*q2*q3*q4*q5**2*q6**3 - 1440*p**2*q2*q3*q4*q5**2*q6**2 + 5040*p**2*q2*q3*q4*q5**2*q6 - 10080*p**2*q2*q3*q4*q5**2 + 180*p**2*q2*q3*q4*q5*q6**4 - 1440*p**2*q2*q3*q4*q5*q6**3 + 5040*p**2*q2*q3*q4*q5*q6**2 - 10080*p**2*q2*q3*q4*q5*q6 + 12600*p**2*q2*q3*q4*q5 - 120*p**2*q3**5*q4*q5 - 120*p**2*q3**4*q4**2*q5 - 120*p**2*q3**4*q4*q5**2 - 120*p**2*q3**4*q4*q5*q6 + 840*p**2*q3**4*q4*q5 - 120*p**2*q3**3*q4**3*q5 - 120*p**2*q3**3*q4**2*q5**2 - 120*p**2*q3**3*q4**2*q5*q6 + 840*p**2*q3**3*q4**2*q5 - 120*p**2*q3**3*q4*q5**3 - 120*p**2*q3**3*q4*q5**2*q6 + 840*p**2*q3**3*q4*q5**2 - 120*p**2*q3**3*q4*q5*q6**2 + 840*p**2*q3**3*q4*q5*q6 - 2520*p**2*q3**3*q4*q5 - 120*p**2*q3**2*q4**4*q5 - 120*p**2*q3**2*q4**3*q5**2 - 120*p**2*q3**2*q4**3*q5*q6 + 840*p**2*q3**2*q4**3*q5 - 120*p**2*q3**2*q4**2*q5**3 - 120*p**2*q3**2*q4**2*q5**2*q6 + 840*p**2*q3**2*q4**2*q5**2 - 120*p**2*q3**2*q4**2*q5*q6**2 + 840*p**2*q3**2*q4**2*q5*q6 - 2520*p**2*q3**2*q4**2*q5 - 120*p**2*q3**2*q4*q5**4 - 120*p**2*q3**2*q4*q5**3*q6 + 840*p**2*q3**2*q4*q5**3 - 120*p**2*q3**2*q4*q5**2*q6**2 + 840*p**2*q3**2*q4*q5**2*q6 - 2520*p**2*q3**2*q4*q5**2 - 120*p**2*q3**2*q4*q5*q6**3 + 840*p**2*q3**2*q4*q5*q6**2 - 2520*p**2*q3**2*q4*q5*q6 + 4200*p**2*q3**2*q4*q5 - 120*p**2*q3*q4**5*q5 - 120*p**2*q3*q4**4*q5**2 - 120*p**2*q3*q4**4*q5*q6 + 840*p**2*q3*q4**4*q5 - 120*p**2*q3*q4**3*q5**3 - 120*p**2*q3*q4**3*q5**2*q6 + 840*p**2*q3*q4**3*q5**2 - 120*p**2*q3*q4**3*q5*q6**2 + 840*p**2*q3*q4**3*q5*q6 - 2520*p**2*q3*q4**3*q5 - 120*p**2*q3*q4**2*q5**4 - 120*p**2*q3*q4**2*q5**3*q6 + 840*p**2*q3*q4**2*q5**3 - 120*p**2*q3*q4**2*q5**2*q6**2 + 840*p**2*q3*q4**2*q5**2*q6 - 2520*p**2*q3*q4**2*q5**2 - 120*p**2*q3*q4**2*q5*q6**3 + 840*p**2*q3*q4**2*q5*q6**2 - 2520*p**2*q3*q4**2*q5*q6 + 4200*p**2*q3*q4**2*q5 - 120*p**2*q3*q4*q5**5 - 120*p**2*q3*q4*q5**4*q6 + 840*p**2*q3*q4*q5**4 - 120*p**2*q3*q4*q5**3*q6**2 + 840*p**2*q3*q4*q5**3*q6 - 2520*p**2*q3*q4*q5**3 - 120*p**2*q3*q4*q5**2*q6**3 + 840*p**2*q3*q4*q5**2*q6**2 - 2520*p**2*q3*q4*q5**2*q6 + 4200*p**2*q3*q4*q5**2 - 120*p**2*q3*q4*q5*q6**4 + 840*p**2*q3*q4*q5*q6**3 - 2520*p**2*q3*q4*q5*q6**2 + 4200*p**2*q3*q4*q5*q6 - 4200*p**2*q3*q4*q5 + 50*p*q1**5*q2*q3*q4*q5 + 50*p*q1**4*q2**2*q3*q4*q5 + 50*p*q1**4*q2*q3**2*q4*q5 + 50*p*q1**4*q2*q3*q4**2*q5 + 50*p*q1**4*q2*q3*q4*q5**2 + 50*p*q1**4*q2*q3*q4*q5*q6 - 450*p*q1**4*q2*q3*q4*q5 + 50*p*q1**3*q2**3*q3*q4*q5 + 50*p*q1**3*q2**2*q3**2*q4*q5 + 50*p*q1**3*q2**2*q3*q4**2*q5 + 50*p*q1**3*q2**2*q3*q4*q5**2 + 50*p*q1**3*q2**2*q3*q4*q5*q6 - 450*p*q1**3*q2**2*q3*q4*q5 + 50*p*q1**3*q2*q3**3*q4*q5 + 50*p*q1**3*q2*q3**2*q4**2*q5 + 50*p*q1**3*q2*q3**2*q4*q5**2 + 50*p*q1**3*q2*q3**2*q4*q5*q6 - 450*p*q1**3*q2*q3**2*q4*q5 + 50*p*q1**3*q2*q3*q4**3*q5 + 50*p*q1**3*q2*q3*q4**2*q5**2 + 50*p*q1**3*q2*q3*q4**2*q5*q6 - 450*p*q1**3*q2*q3*q4**2*q5 + 50*p*q1**3*q2*q3*q4*q5**3 + 50*p*q1**3*q2*q3*q4*q5**2*q6 - 450*p*q1**3*q2*q3*q4*q5**2 + 50*p*q1**3*q2*q3*q4*q5*q6**2 - 450*p*q1**3*q2*q3*q4*q5*q6 + 1800*p*q1**3*q2*q3*q4*q5 + 50*p*q1**2*q2**4*q3*q4*q5 + 50*p*q1**2*q2**3*q3**2*q4*q5 + 50*p*q1**2*q2**3*q3*q4**2*q5 + 50*p*q1**2*q2**3*q3*q4*q5**2 + 50*p*q1**2*q2**3*q3*q4*q5*q6 - 450*p*q1**2*q2**3*q3*q4*q5 + 50*p*q1**2*q2**2*q3**3*q4*q5 + 50*p*q1**2*q2**2*q3**2*q4**2*q5 + 50*p*q1**2*q2**2*q3**2*q4*q5**2 + 50*p*q1**2*q2**2*q3**2*q4*q5*q6 - 450*p*q1**2*q2**2*q3**2*q4*q5 + 50*p*q1**2*q2**2*q3*q4**3*q5 + 50*p*q1**2*q2**2*q3*q4**2*q5**2 + 50*p*q1**2*q2**2*q3*q4**2*q5*q6 - 450*p*q1**2*q2**2*q3*q4**2*q5 + 50*p*q1**2*q2**2*q3*q4*q5**3 + 50*p*q1**2*q2**2*q3*q4*q5**2*q6 - 450*p*q1**2*q2**2*q3*q4*q5**2 + 50*p*q1**2*q2**2*q3*q4*q5*q6**2 - 450*p*q1**2*q2**2*q3*q4*q5*q6 + 1800*p*q1**2*q2**2*q3*q4*q5 + 50*p*q1**2*q2*q3**4*q4*q5 + 50*p*q1**2*q2*q3**3*q4**2*q5 + 50*p*q1**2*q2*q3**3*q4*q5**2 + 50*p*q1**2*q2*q3**3*q4*q5*q6 - 450*p*q1**2*q2*q3**3*q4*q5 + 50*p*q1**2*q2*q3**2*q4**3*q5 + 50*p*q1**2*q2*q3**2*q4**2*q5**2 + 50*p*q1**2*q2*q3**2*q4**2*q5*q6 - 450*p*q1**2*q2*q3**2*q4**2*q5 + 50*p*q1**2*q2*q3**2*q4*q5**3 + 50*p*q1**2*q2*q3**2*q4*q5**2*q6 - 450*p*q1**2*q2*q3**2*q4*q5**2 + 50*p*q1**2*q2*q3**2*q4*q5*q6**2 - 450*p*q1**2*q2*q3**2*q4*q5*q6 + 1800*p*q1**2*q2*q3**2*q4*q5 + 50*p*q1**2*q2*q3*q4**4*q5 + 50*p*q1**2*q2*q3*q4**3*q5**2 + 50*p*q1**2*q2*q3*q4**3*q5*q6 - 450*p*q1**2*q2*q3*q4**3*q5 + 50*p*q1**2*q2*q3*q4**2*q5**3 + 50*p*q1**2*q2*q3*q4**2*q5**2*q6 - 450*p*q1**2*q2*q3*q4**2*q5**2 + 50*p*q1**2*q2*q3*q4**2*q5*q6**2 - 450*p*q1**2*q2*q3*q4**2*q5*q6 + 1800*p*q1**2*q2*q3*q4**2*q5 + 50*p*q1**2*q2*q3*q4*q5**4 + 50*p*q1**2*q2*q3*q4*q5**3*q6 - 450*p*q1**2*q2*q3*q4*q5**3 + 50*p*q1**2*q2*q3*q4*q5**2*q6**2 - 450*p*q1**2*q2*q3*q4*q5**2*q6 + 1800*p*q1**2*q2*q3*q4*q5**2 + 50*p*q1**2*q2*q3*q4*q5*q6**3 - 450*p*q1**2*q2*q3*q4*q5*q6**2 + 1800*p*q1**2*q2*q3*q4*q5*q6 - 4200*p*q1**2*q2*q3*q4*q5 + 50*p*q1*q2**5*q3*q4*q5 + 50*p*q1*q2**4*q3**2*q4*q5 + 50*p*q1*q2**4*q3*q4**2*q5 + 50*p*q1*q2**4*q3*q4*q5**2 + 50*p*q1*q2**4*q3*q4*q5*q6 - 450*p*q1*q2**4*q3*q4*q5 + 50*p*q1*q2**3*q3**3*q4*q5 + 50*p*q1*q2**3*q3**2*q4**2*q5 + 50*p*q1*q2**3*q3**2*q4*q5**2 + 50*p*q1*q2**3*q3**2*q4*q5*q6 - 450*p*q1*q2**3*q3**2*q4*q5 + 50*p*q1*q2**3*q3*q4**3*q5 + 50*p*q1*q2**3*q3*q4**2*q5**2 + 50*p*q1*q2**3*q3*q4**2*q5*q6 - 450*p*q1*q2**3*q3*q4**2*q5 + 50*p*q1*q2**3*q3*q4*q5**3 + 50*p*q1*q2**3*q3*q4*q5**2*q6 - 450*p*q1*q2**3*q3*q4*q5**2 + 50*p*q1*q2**3*q3*q4*q5*q6**2 - 450*p*q1*q2**3*q3*q4*q5*q6 + 1800*p*q1*q2**3*q3*q4*q5 + 50*p*q1*q2**2*q3**4*q4*q5 + 50*p*q1*q2**2*q3**3*q4**2*q5 + 50*p*q1*q2**2*q3**3*q4*q5**2 + 50*p*q1*q2**2*q3**3*q4*q5*q6 - 450*p*q1*q2**2*q3**3*q4*q5 + 50*p*q1*q2**2*q3**2*q4**3*q5 + 50*p*q1*q2**2*q3**2*q4**2*q5**2 + 50*p*q1*q2**2*q3**2*q4**2*q5*q6 - 450*p*q1*q2**2*q3**2*q4**2*q5 + 50*p*q1*q2**2*q3**2*q4*q5**3 + 50*p*q1*q2**2*q3**2*q4*q5**2*q6 - 450*p*q1*q2**2*q3**2*q4*q5**2 + 50*p*q1*q2**2*q3**2*q4*q5*q6**2 - 450*p*q1*q2**2*q3**2*q4*q5*q6 + 1800*p*q1*q2**2*q3**2*q4*q5 + 50*p*q1*q2**2*q3*q4**4*q5 + 50*p*q1*q2**2*q3*q4**3*q5**2 + 50*p*q1*q2**2*q3*q4**3*q5*q6 - 450*p*q1*q2**2*q3*q4**3*q5 + 50*p*q1*q2**2*q3*q4**2*q5**3 + 50*p*q1*q2**2*q3*q4**2*q5**2*q6 - 450*p*q1*q2**2*q3*q4**2*q5**2 + 50*p*q1*q2**2*q3*q4**2*q5*q6**2 - 450*p*q1*q2**2*q3*q4**2*q5*q6 + 1800*p*q1*q2**2*q3*q4**2*q5 + 50*p*q1*q2**2*q3*q4*q5**4 + 50*p*q1*q2**2*q3*q4*q5**3*q6 - 450*p*q1*q2**2*q3*q4*q5**3 + 50*p*q1*q2**2*q3*q4*q5**2*q6**2 - 450*p*q1*q2**2*q3*q4*q5**2*q6 + 1800*p*q1*q2**2*q3*q4*q5**2 + 50*p*q1*q2**2*q3*q4*q5*q6**3 - 450*p*q1*q2**2*q3*q4*q5*q6**2 + 1800*p*q1*q2**2*q3*q4*q5*q6 - 4200*p*q1*q2**2*q3*q4*q5 + 50*p*q1*q2*q3**5*q4*q5 + 50*p*q1*q2*q3**4*q4**2*q5 + 50*p*q1*q2*q3**4*q4*q5**2 + 50*p*q1*q2*q3**4*q4*q5*q6 - 450*p*q1*q2*q3**4*q4*q5 + 50*p*q1*q2*q3**3*q4**3*q5 + 50*p*q1*q2*q3**3*q4**2*q5**2 + 50*p*q1*q2*q3**3*q4**2*q5*q6 - 450*p*q1*q2*q3**3*q4**2*q5 + 50*p*q1*q2*q3**3*q4*q5**3 + 50*p*q1*q2*q3**3*q4*q5**2*q6 - 450*p*q1*q2*q3**3*q4*q5**2 + 50*p*q1*q2*q3**3*q4*q5*q6**2 - 450*p*q1*q2*q3**3*q4*q5*q6 + 1800*p*q1*q2*q3**3*q4*q5 + 50*p*q1*q2*q3**2*q4**4*q5 + 50*p*q1*q2*q3**2*q4**3*q5**2 + 50*p*q1*q2*q3**2*q4**3*q5*q6 - 450*p*q1*q2*q3**2*q4**3*q5 + 50*p*q1*q2*q3**2*q4**2*q5**3 + 50*p*q1*q2*q3**2*q4**2*q5**2*q6 - 450*p*q1*q2*q3**2*q4**2*q5**2 + 50*p*q1*q2*q3**2*q4**2*q5*q6**2 - 450*p*q1*q2*q3**2*q4**2*q5*q6 + 1800*p*q1*q2*q3**2*q4**2*q5 + 50*p*q1*q2*q3**2*q4*q5**4 + 50*p*q1*q2*q3**2*q4*q5**3*q6 - 450*p*q1*q2*q3**2*q4*q5**3 + 50*p*q1*q2*q3**2*q4*q5**2*q6**2 - 450*p*q1*q2*q3**2*q4*q5**2*q6 + 1800*p*q1*q2*q3**2*q4*q5**2 + 50*p*q1*q2*q3**2*q4*q5*q6**3 - 450*p*q1*q2*q3**2*q4*q5*q6**2 + 1800*p*q1*q2*q3**2*q4*q5*q6 - 4200*p*q1*q2*q3**2*q4*q5 + 50*p*q1*q2*q3*q4**5*q5 + 50*p*q1*q2*q3*q4**4*q5**2 + 50*p*q1*q2*q3*q4**4*q5*q6 - 450*p*q1*q2*q3*q4**4*q5 + 50*p*q1*q2*q3*q4**3*q5**3 + 50*p*q1*q2*q3*q4**3*q5**2*q6 - 450*p*q1*q2*q3*q4**3*q5**2 + 50*p*q1*q2*q3*q4**3*q5*q6**2 - 450*p*q1*q2*q3*q4**3*q5*q6 + 1800*p*q1*q2*q3*q4**3*q5 + 50*p*q1*q2*q3*q4**2*q5**4 + 50*p*q1*q2*q3*q4**2*q5**3*q6 - 450*p*q1*q2*q3*q4**2*q5**3 + 50*p*q1*q2*q3*q4**2*q5**2*q6**2 - 450*p*q1*q2*q3*q4**2*q5**2*q6 + 1800*p*q1*q2*q3*q4**2*q5**2 + 50*p*q1*q2*q3*q4**2*q5*q6**3 - 450*p*q1*q2*q3*q4**2*q5*q6**2 + 1800*p*q1*q2*q3*q4**2*q5*q6 - 4200*p*q1*q2*q3*q4**2*q5 + 50*p*q1*q2*q3*q4*q5**5 + 50*p*q1*q2*q3*q4*q5**4*q6 - 450*p*q1*q2*q3*q4*q5**4 + 50*p*q1*q2*q3*q4*q5**3*q6**2 - 450*p*q1*q2*q3*q4*q5**3*q6 + 1800*p*q1*q2*q3*q4*q5**3 + 50*p*q1*q2*q3*q4*q5**2*q6**3 - 450*p*q1*q2*q3*q4*q5**2*q6**2 + 1800*p*q1*q2*q3*q4*q5**2*q6 - 4200*p*q1*q2*q3*q4*q5**2 + 50*p*q1*q2*q3*q4*q5*q6**4 - 450*p*q1*q2*q3*q4*q5*q6**3 + 1800*p*q1*q2*q3*q4*q5*q6**2 - 4200*p*q1*q2*q3*q4*q5*q6 + 6300*p*q1*q2*q3*q4*q5 - 45*p*q2**5*q3*q4*q5 - 45*p*q2**4*q3**2*q4*q5 - 45*p*q2**4*q3*q4**2*q5 - 45*p*q2**4*q3*q4*q5**2 - 45*p*q2**4*q3*q4*q5*q6 + 360*p*q2**4*q3*q4*q5 - 45*p*q2**3*q3**3*q4*q5 - 45*p*q2**3*q3**2*q4**2*q5 - 45*p*q2**3*q3**2*q4*q5**2 - 45*p*q2**3*q3**2*q4*q5*q6 + 360*p*q2**3*q3**2*q4*q5 - 45*p*q2**3*q3*q4**3*q5 - 45*p*q2**3*q3*q4**2*q5**2 - 45*p*q2**3*q3*q4**2*q5*q6 + 360*p*q2**3*q3*q4**2*q5 - 45*p*q2**3*q3*q4*q5**3 - 45*p*q2**3*q3*q4*q5**2*q6 + 360*p*q2**3*q3*q4*q5**2 - 45*p*q2**3*q3*q4*q5*q6**2 + 360*p*q2**3*q3*q4*q5*q6 - 1260*p*q2**3*q3*q4*q5 - 45*p*q2**2*q3**4*q4*q5 - 45*p*q2**2*q3**3*q4**2*q5 - 45*p*q2**2*q3**3*q4*q5**2 - 45*p*q2**2*q3**3*q4*q5*q6 + 360*p*q2**2*q3**3*q4*q5 - 45*p*q2**2*q3**2*q4**3*q5 - 45*p*q2**2*q3**2*q4**2*q5**2 - 45*p*q2**2*q3**2*q4**2*q5*q6 + 360*p*q2**2*q3**2*q4**2*q5 - 45*p*q2**2*q3**2*q4*q5**3 - 45*p*q2**2*q3**2*q4*q5**2*q6 + 360*p*q2**2*q3**2*q4*q5**2 - 45*p*q2**2*q3**2*q4*q5*q6**2 + 360*p*q2**2*q3**2*q4*q5*q6 - 1260*p*q2**2*q3**2*q4*q5 - 45*p*q2**2*q3*q4**4*q5 - 45*p*q2**2*q3*q4**3*q5**2 - 45*p*q2**2*q3*q4**3*q5*q6 + 360*p*q2**2*q3*q4**3*q5 - 45*p*q2**2*q3*q4**2*q5**3 - 45*p*q2**2*q3*q4**2*q5**2*q6 + 360*p*q2**2*q3*q4**2*q5**2 - 45*p*q2**2*q3*q4**2*q5*q6**2 + 360*p*q2**2*q3*q4**2*q5*q6 - 1260*p*q2**2*q3*q4**2*q5 - 45*p*q2**2*q3*q4*q5**4 - 45*p*q2**2*q3*q4*q5**3*q6 + 360*p*q2**2*q3*q4*q5**3 - 45*p*q2**2*q3*q4*q5**2*q6**2 + 360*p*q2**2*q3*q4*q5**2*q6 - 1260*p*q2**2*q3*q4*q5**2 - 45*p*q2**2*q3*q4*q5*q6**3 + 360*p*q2**2*q3*q4*q5*q6**2 - 1260*p*q2**2*q3*q4*q5*q6 + 2520*p*q2**2*q3*q4*q5 - 45*p*q2*q3**5*q4*q5 - 45*p*q2*q3**4*q4**2*q5 - 45*p*q2*q3**4*q4*q5**2 - 45*p*q2*q3**4*q4*q5*q6 + 360*p*q2*q3**4*q4*q5 - 45*p*q2*q3**3*q4**3*q5 - 45*p*q2*q3**3*q4**2*q5**2 - 45*p*q2*q3**3*q4**2*q5*q6 + 360*p*q2*q3**3*q4**2*q5 - 45*p*q2*q3**3*q4*q5**3 - 45*p*q2*q3**3*q4*q5**2*q6 + 360*p*q2*q3**3*q4*q5**2 - 45*p*q2*q3**3*q4*q5*q6**2 + 360*p*q2*q3**3*q4*q5*q6 - 1260*p*q2*q3**3*q4*q5 - 45*p*q2*q3**2*q4**4*q5 - 45*p*q2*q3**2*q4**3*q5**2 - 45*p*q2*q3**2*q4**3*q5*q6 + 360*p*q2*q3**2*q4**3*q5 - 45*p*q2*q3**2*q4**2*q5**3 - 45*p*q2*q3**2*q4**2*q5**2*q6 + 360*p*q2*q3**2*q4**2*q5**2 - 45*p*q2*q3**2*q4**2*q5*q6**2 + 360*p*q2*q3**2*q4**2*q5*q6 - 1260*p*q2*q3**2*q4**2*q5 - 45*p*q2*q3**2*q4*q5**4 - 45*p*q2*q3**2*q4*q5**3*q6 + 360*p*q2*q3**2*q4*q5**3 - 45*p*q2*q3**2*q4*q5**2*q6**2 + 360*p*q2*q3**2*q4*q5**2*q6 - 1260*p*q2*q3**2*q4*q5**2 - 45*p*q2*q3**2*q4*q5*q6**3 + 360*p*q2*q3**2*q4*q5*q6**2 - 1260*p*q2*q3**2*q4*q5*q6 + 2520*p*q2*q3**2*q4*q5 - 45*p*q2*q3*q4**5*q5 - 45*p*q2*q3*q4**4*q5**2 - 45*p*q2*q3*q4**4*q5*q6 + 360*p*q2*q3*q4**4*q5 - 45*p*q2*q3*q4**3*q5**3 - 45*p*q2*q3*q4**3*q5**2*q6 + 360*p*q2*q3*q4**3*q5**2 - 45*p*q2*q3*q4**3*q5*q6**2 + 360*p*q2*q3*q4**3*q5*q6 - 1260*p*q2*q3*q4**3*q5 - 45*p*q2*q3*q4**2*q5**4 - 45*p*q2*q3*q4**2*q5**3*q6 + 360*p*q2*q3*q4**2*q5**3 - 45*p*q2*q3*q4**2*q5**2*q6**2 + 360*p*q2*q3*q4**2*q5**2*q6 - 1260*p*q2*q3*q4**2*q5**2 - 45*p*q2*q3*q4**2*q5*q6**3 + 360*p*q2*q3*q4**2*q5*q6**2 - 1260*p*q2*q3*q4**2*q5*q6 + 2520*p*q2*q3*q4**2*q5 - 45*p*q2*q3*q4*q5**5 - 45*p*q2*q3*q4*q5**4*q6 + 360*p*q2*q3*q4*q5**4 - 45*p*q2*q3*q4*q5**3*q6**2 + 360*p*q2*q3*q4*q5**3*q6 - 1260*p*q2*q3*q4*q5**3 - 45*p*q2*q3*q4*q5**2*q6**3 + 360*p*q2*q3*q4*q5**2*q6**2 - 1260*p*q2*q3*q4*q5**2*q6 + 2520*p*q2*q3*q4*q5**2 - 45*p*q2*q3*q4*q5*q6**4 + 360*p*q2*q3*q4*q5*q6**3 - 1260*p*q2*q3*q4*q5*q6**2 + 2520*p*q2*q3*q4*q5*q6 - 3150*p*q2*q3*q4*q5 - 10*q1**5*q2*q3*q4*q5 - 10*q1**4*q2**2*q3*q4*q5 - 10*q1**4*q2*q3**2*q4*q5 - 10*q1**4*q2*q3*q4**2*q5 - 10*q1**4*q2*q3*q4*q5**2 - 10*q1**4*q2*q3*q4*q5*q6 + 90*q1**4*q2*q3*q4*q5 - 10*q1**3*q2**3*q3*q4*q5 - 10*q1**3*q2**2*q3**2*q4*q5 - 10*q1**3*q2**2*q3*q4**2*q5 - 10*q1**3*q2**2*q3*q4*q5**2 - 10*q1**3*q2**2*q3*q4*q5*q6 + 90*q1**3*q2**2*q3*q4*q5 - 10*q1**3*q2*q3**3*q4*q5 - 10*q1**3*q2*q3**2*q4**2*q5 - 10*q1**3*q2*q3**2*q4*q5**2 - 10*q1**3*q2*q3**2*q4*q5*q6 + 90*q1**3*q2*q3**2*q4*q5 - 10*q1**3*q2*q3*q4**3*q5 - 10*q1**3*q2*q3*q4**2*q5**2 - 10*q1**3*q2*q3*q4**2*q5*q6 + 90*q1**3*q2*q3*q4**2*q5 - 10*q1**3*q2*q3*q4*q5**3 - 10*q1**3*q2*q3*q4*q5**2*q6 + 90*q1**3*q2*q3*q4*q5**2 - 10*q1**3*q2*q3*q4*q5*q6**2 + 90*q1**3*q2*q3*q4*q5*q6 - 360*q1**3*q2*q3*q4*q5 - 10*q1**2*q2**4*q3*q4*q5 - 10*q1**2*q2**3*q3**2*q4*q5 - 10*q1**2*q2**3*q3*q4**2*q5 - 10*q1**2*q2**3*q3*q4*q5**2 - 10*q1**2*q2**3*q3*q4*q5*q6 + 90*q1**2*q2**3*q3*q4*q5 - 10*q1**2*q2**2*q3**3*q4*q5 - 10*q1**2*q2**2*q3**2*q4**2*q5 - 10*q1**2*q2**2*q3**2*q4*q5**2 - 10*q1**2*q2**2*q3**2*q4*q5*q6 + 90*q1**2*q2**2*q3**2*q4*q5 - 10*q1**2*q2**2*q3*q4**3*q5 - 10*q1**2*q2**2*q3*q4**2*q5**2 - 10*q1**2*q2**2*q3*q4**2*q5*q6 + 90*q1**2*q2**2*q3*q4**2*q5 - 10*q1**2*q2**2*q3*q4*q5**3 - 10*q1**2*q2**2*q3*q4*q5**2*q6 + 90*q1**2*q2**2*q3*q4*q5**2 - 10*q1**2*q2**2*q3*q4*q5*q6**2 + 90*q1**2*q2**2*q3*q4*q5*q6 - 360*q1**2*q2**2*q3*q4*q5 - 10*q1**2*q2*q3**4*q4*q5 - 10*q1**2*q2*q3**3*q4**2*q5 - 10*q1**2*q2*q3**3*q4*q5**2 - 10*q1**2*q2*q3**3*q4*q5*q6 + 90*q1**2*q2*q3**3*q4*q5 - 10*q1**2*q2*q3**2*q4**3*q5 - 10*q1**2*q2*q3**2*q4**2*q5**2 - 10*q1**2*q2*q3**2*q4**2*q5*q6 + 90*q1**2*q2*q3**2*q4**2*q5 - 10*q1**2*q2*q3**2*q4*q5**3 - 10*q1**2*q2*q3**2*q4*q5**2*q6 + 90*q1**2*q2*q3**2*q4*q5**2 - 10*q1**2*q2*q3**2*q4*q5*q6**2 + 90*q1**2*q2*q3**2*q4*q5*q6 - 360*q1**2*q2*q3**2*q4*q5 - 10*q1**2*q2*q3*q4**4*q5 - 10*q1**2*q2*q3*q4**3*q5**2 - 10*q1**2*q2*q3*q4**3*q5*q6 + 90*q1**2*q2*q3*q4**3*q5 - 10*q1**2*q2*q3*q4**2*q5**3 - 10*q1**2*q2*q3*q4**2*q5**2*q6 + 90*q1**2*q2*q3*q4**2*q5**2 - 10*q1**2*q2*q3*q4**2*q5*q6**2 + 90*q1**2*q2*q3*q4**2*q5*q6 - 360*q1**2*q2*q3*q4**2*q5 - 10*q1**2*q2*q3*q4*q5**4 - 10*q1**2*q2*q3*q4*q5**3*q6 + 90*q1**2*q2*q3*q4*q5**3 - 10*q1**2*q2*q3*q4*q5**2*q6**2 + 90*q1**2*q2*q3*q4*q5**2*q6 - 360*q1**2*q2*q3*q4*q5**2 - 10*q1**2*q2*q3*q4*q5*q6**3 + 90*q1**2*q2*q3*q4*q5*q6**2 - 360*q1**2*q2*q3*q4*q5*q6 + 840*q1**2*q2*q3*q4*q5 - 10*q1*q2**5*q3*q4*q5 - 10*q1*q2**4*q3**2*q4*q5 - 10*q1*q2**4*q3*q4**2*q5 - 10*q1*q2**4*q3*q4*q5**2 - 10*q1*q2**4*q3*q4*q5*q6 + 90*q1*q2**4*q3*q4*q5 - 10*q1*q2**3*q3**3*q4*q5 - 10*q1*q2**3*q3**2*q4**2*q5 - 10*q1*q2**3*q3**2*q4*q5**2 - 10*q1*q2**3*q3**2*q4*q5*q6 + 90*q1*q2**3*q3**2*q4*q5 - 10*q1*q2**3*q3*q4**3*q5 - 10*q1*q2**3*q3*q4**2*q5**2 - 10*q1*q2**3*q3*q4**2*q5*q6 + 90*q1*q2**3*q3*q4**2*q5 - 10*q1*q2**3*q3*q4*q5**3 - 10*q1*q2**3*q3*q4*q5**2*q6 + 90*q1*q2**3*q3*q4*q5**2 - 10*q1*q2**3*q3*q4*q5*q6**2 + 90*q1*q2**3*q3*q4*q5*q6 - 360*q1*q2**3*q3*q4*q5 - 10*q1*q2**2*q3**4*q4*q5 - 10*q1*q2**2*q3**3*q4**2*q5 - 10*q1*q2**2*q3**3*q4*q5**2 - 10*q1*q2**2*q3**3*q4*q5*q6 + 90*q1*q2**2*q3**3*q4*q5 - 10*q1*q2**2*q3**2*q4**3*q5 - 10*q1*q2**2*q3**2*q4**2*q5**2 - 10*q1*q2**2*q3**2*q4**2*q5*q6 + 90*q1*q2**2*q3**2*q4**2*q5 - 10*q1*q2**2*q3**2*q4*q5**3 - 10*q1*q2**2*q3**2*q4*q5**2*q6 + 90*q1*q2**2*q3**2*q4*q5**2 - 10*q1*q2**2*q3**2*q4*q5*q6**2 + 90*q1*q2**2*q3**2*q4*q5*q6 - 360*q1*q2**2*q3**2*q4*q5 - 10*q1*q2**2*q3*q4**4*q5 - 10*q1*q2**2*q3*q4**3*q5**2 - 10*q1*q2**2*q3*q4**3*q5*q6 + 90*q1*q2**2*q3*q4**3*q5 - 10*q1*q2**2*q3*q4**2*q5**3 - 10*q1*q2**2*q3*q4**2*q5**2*q6 + 90*q1*q2**2*q3*q4**2*q5**2 - 10*q1*q2**2*q3*q4**2*q5*q6**2 + 90*q1*q2**2*q3*q4**2*q5*q6 - 360*q1*q2**2*q3*q4**2*q5 - 10*q1*q2**2*q3*q4*q5**4 - 10*q1*q2**2*q3*q4*q5**3*q6 + 90*q1*q2**2*q3*q4*q5**3 - 10*q1*q2**2*q3*q4*q5**2*q6**2 + 90*q1*q2**2*q3*q4*q5**2*q6 - 360*q1*q2**2*q3*q4*q5**2 - 10*q1*q2**2*q3*q4*q5*q6**3 + 90*q1*q2**2*q3*q4*q5*q6**2 - 360*q1*q2**2*q3*q4*q5*q6 + 840*q1*q2**2*q3*q4*q5 - 10*q1*q2*q3**5*q4*q5 - 10*q1*q2*q3**4*q4**2*q5 - 10*q1*q2*q3**4*q4*q5**2 - 10*q1*q2*q3**4*q4*q5*q6 + 90*q1*q2*q3**4*q4*q5 - 10*q1*q2*q3**3*q4**3*q5 - 10*q1*q2*q3**3*q4**2*q5**2 - 10*q1*q2*q3**3*q4**2*q5*q6 + 90*q1*q2*q3**3*q4**2*q5 - 10*q1*q2*q3**3*q4*q5**3 - 10*q1*q2*q3**3*q4*q5**2*q6 + 90*q1*q2*q3**3*q4*q5**2 - 10*q1*q2*q3**3*q4*q5*q6**2 + 90*q1*q2*q3**3*q4*q5*q6 - 360*q1*q2*q3**3*q4*q5 - 10*q1*q2*q3**2*q4**4*q5 - 10*q1*q2*q3**2*q4**3*q5**2 - 10*q1*q2*q3**2*q4**3*q5*q6 + 90*q1*q2*q3**2*q4**3*q5 - 10*q1*q2*q3**2*q4**2*q5**3 - 10*q1*q2*q3**2*q4**2*q5**2*q6 + 90*q1*q2*q3**2*q4**2*q5**2 - 10*q1*q2*q3**2*q4**2*q5*q6**2 + 90*q1*q2*q3**2*q4**2*q5*q6 - 360*q1*q2*q3**2*q4**2*q5 - 10*q1*q2*q3**2*q4*q5**4 - 10*q1*q2*q3**2*q4*q5**3*q6 + 90*q1*q2*q3**2*q4*q5**3 - 10*q1*q2*q3**2*q4*q5**2*q6**2 + 90*q1*q2*q3**2*q4*q5**2*q6 - 360*q1*q2*q3**2*q4*q5**2 - 10*q1*q2*q3**2*q4*q5*q6**3 + 90*q1*q2*q3**2*q4*q5*q6**2 - 360*q1*q2*q3**2*q4*q5*q6 + 840*q1*q2*q3**2*q4*q5 - 10*q1*q2*q3*q4**5*q5 - 10*q1*q2*q3*q4**4*q5**2 - 10*q1*q2*q3*q4**4*q5*q6 + 90*q1*q2*q3*q4**4*q5 - 10*q1*q2*q3*q4**3*q5**3 - 10*q1*q2*q3*q4**3*q5**2*q6 + 90*q1*q2*q3*q4**3*q5**2 - 10*q1*q2*q3*q4**3*q5*q6**2 + 90*q1*q2*q3*q4**3*q5*q6 - 360*q1*q2*q3*q4**3*q5 - 10*q1*q2*q3*q4**2*q5**4 - 10*q1*q2*q3*q4**2*q5**3*q6 + 90*q1*q2*q3*q4**2*q5**3 - 10*q1*q2*q3*q4**2*q5**2*q6**2 + 90*q1*q2*q3*q4**2*q5**2*q6 - 360*q1*q2*q3*q4**2*q5**2 - 10*q1*q2*q3*q4**2*q5*q6**3 + 90*q1*q2*q3*q4**2*q5*q6**2 - 360*q1*q2*q3*q4**2*q5*q6 + 840*q1*q2*q3*q4**2*q5 - 10*q1*q2*q3*q4*q5**5 - 10*q1*q2*q3*q4*q5**4*q6 + 90*q1*q2*q3*q4*q5**4 - 10*q1*q2*q3*q4*q5**3*q6**2 + 90*q1*q2*q3*q4*q5**3*q6 - 360*q1*q2*q3*q4*q5**3 - 10*q1*q2*q3*q4*q5**2*q6**3 + 90*q1*q2*q3*q4*q5**2*q6**2 - 360*q1*q2*q3*q4*q5**2*q6 + 840*q1*q2*q3*q4*q5**2 - 10*q1*q2*q3*q4*q5*q6**4 + 90*q1*q2*q3*q4*q5*q6**3 - 360*q1*q2*q3*q4*q5*q6**2 + 840*q1*q2*q3*q4*q5*q6 - 1260*q1*q2*q3*q4*q5)'
        f_multiparam[10][7] = 'p*(p - 1)**3*(10*p**6*q1**4*q2*q3*q4*q5*q6 + 10*p**6*q1**3*q2**2*q3*q4*q5*q6 + 10*p**6*q1**3*q2*q3**2*q4*q5*q6 + 10*p**6*q1**3*q2*q3*q4**2*q5*q6 + 10*p**6*q1**3*q2*q3*q4*q5**2*q6 + 10*p**6*q1**3*q2*q3*q4*q5*q6**2 + 10*p**6*q1**3*q2*q3*q4*q5*q6*q7 - 90*p**6*q1**3*q2*q3*q4*q5*q6 + 10*p**6*q1**2*q2**3*q3*q4*q5*q6 + 10*p**6*q1**2*q2**2*q3**2*q4*q5*q6 + 10*p**6*q1**2*q2**2*q3*q4**2*q5*q6 + 10*p**6*q1**2*q2**2*q3*q4*q5**2*q6 + 10*p**6*q1**2*q2**2*q3*q4*q5*q6**2 + 10*p**6*q1**2*q2**2*q3*q4*q5*q6*q7 - 90*p**6*q1**2*q2**2*q3*q4*q5*q6 + 10*p**6*q1**2*q2*q3**3*q4*q5*q6 + 10*p**6*q1**2*q2*q3**2*q4**2*q5*q6 + 10*p**6*q1**2*q2*q3**2*q4*q5**2*q6 + 10*p**6*q1**2*q2*q3**2*q4*q5*q6**2 + 10*p**6*q1**2*q2*q3**2*q4*q5*q6*q7 - 90*p**6*q1**2*q2*q3**2*q4*q5*q6 + 10*p**6*q1**2*q2*q3*q4**3*q5*q6 + 10*p**6*q1**2*q2*q3*q4**2*q5**2*q6 + 10*p**6*q1**2*q2*q3*q4**2*q5*q6**2 + 10*p**6*q1**2*q2*q3*q4**2*q5*q6*q7 - 90*p**6*q1**2*q2*q3*q4**2*q5*q6 + 10*p**6*q1**2*q2*q3*q4*q5**3*q6 + 10*p**6*q1**2*q2*q3*q4*q5**2*q6**2 + 10*p**6*q1**2*q2*q3*q4*q5**2*q6*q7 - 90*p**6*q1**2*q2*q3*q4*q5**2*q6 + 10*p**6*q1**2*q2*q3*q4*q5*q6**3 + 10*p**6*q1**2*q2*q3*q4*q5*q6**2*q7 - 90*p**6*q1**2*q2*q3*q4*q5*q6**2 + 10*p**6*q1**2*q2*q3*q4*q5*q6*q7**2 - 90*p**6*q1**2*q2*q3*q4*q5*q6*q7 + 360*p**6*q1**2*q2*q3*q4*q5*q6 + 10*p**6*q1*q2**4*q3*q4*q5*q6 + 10*p**6*q1*q2**3*q3**2*q4*q5*q6 + 10*p**6*q1*q2**3*q3*q4**2*q5*q6 + 10*p**6*q1*q2**3*q3*q4*q5**2*q6 + 10*p**6*q1*q2**3*q3*q4*q5*q6**2 + 10*p**6*q1*q2**3*q3*q4*q5*q6*q7 - 90*p**6*q1*q2**3*q3*q4*q5*q6 + 10*p**6*q1*q2**2*q3**3*q4*q5*q6 + 10*p**6*q1*q2**2*q3**2*q4**2*q5*q6 + 10*p**6*q1*q2**2*q3**2*q4*q5**2*q6 + 10*p**6*q1*q2**2*q3**2*q4*q5*q6**2 + 10*p**6*q1*q2**2*q3**2*q4*q5*q6*q7 - 90*p**6*q1*q2**2*q3**2*q4*q5*q6 + 10*p**6*q1*q2**2*q3*q4**3*q5*q6 + 10*p**6*q1*q2**2*q3*q4**2*q5**2*q6 + 10*p**6*q1*q2**2*q3*q4**2*q5*q6**2 + 10*p**6*q1*q2**2*q3*q4**2*q5*q6*q7 - 90*p**6*q1*q2**2*q3*q4**2*q5*q6 + 10*p**6*q1*q2**2*q3*q4*q5**3*q6 + 10*p**6*q1*q2**2*q3*q4*q5**2*q6**2 + 10*p**6*q1*q2**2*q3*q4*q5**2*q6*q7 - 90*p**6*q1*q2**2*q3*q4*q5**2*q6 + 10*p**6*q1*q2**2*q3*q4*q5*q6**3 + 10*p**6*q1*q2**2*q3*q4*q5*q6**2*q7 - 90*p**6*q1*q2**2*q3*q4*q5*q6**2 + 10*p**6*q1*q2**2*q3*q4*q5*q6*q7**2 - 90*p**6*q1*q2**2*q3*q4*q5*q6*q7 + 360*p**6*q1*q2**2*q3*q4*q5*q6 + 10*p**6*q1*q2*q3**4*q4*q5*q6 + 10*p**6*q1*q2*q3**3*q4**2*q5*q6 + 10*p**6*q1*q2*q3**3*q4*q5**2*q6 + 10*p**6*q1*q2*q3**3*q4*q5*q6**2 + 10*p**6*q1*q2*q3**3*q4*q5*q6*q7 - 90*p**6*q1*q2*q3**3*q4*q5*q6 + 10*p**6*q1*q2*q3**2*q4**3*q5*q6 + 10*p**6*q1*q2*q3**2*q4**2*q5**2*q6 + 10*p**6*q1*q2*q3**2*q4**2*q5*q6**2 + 10*p**6*q1*q2*q3**2*q4**2*q5*q6*q7 - 90*p**6*q1*q2*q3**2*q4**2*q5*q6 + 10*p**6*q1*q2*q3**2*q4*q5**3*q6 + 10*p**6*q1*q2*q3**2*q4*q5**2*q6**2 + 10*p**6*q1*q2*q3**2*q4*q5**2*q6*q7 - 90*p**6*q1*q2*q3**2*q4*q5**2*q6 + 10*p**6*q1*q2*q3**2*q4*q5*q6**3 + 10*p**6*q1*q2*q3**2*q4*q5*q6**2*q7 - 90*p**6*q1*q2*q3**2*q4*q5*q6**2 + 10*p**6*q1*q2*q3**2*q4*q5*q6*q7**2 - 90*p**6*q1*q2*q3**2*q4*q5*q6*q7 + 360*p**6*q1*q2*q3**2*q4*q5*q6 + 10*p**6*q1*q2*q3*q4**4*q5*q6 + 10*p**6*q1*q2*q3*q4**3*q5**2*q6 + 10*p**6*q1*q2*q3*q4**3*q5*q6**2 + 10*p**6*q1*q2*q3*q4**3*q5*q6*q7 - 90*p**6*q1*q2*q3*q4**3*q5*q6 + 10*p**6*q1*q2*q3*q4**2*q5**3*q6 + 10*p**6*q1*q2*q3*q4**2*q5**2*q6**2 + 10*p**6*q1*q2*q3*q4**2*q5**2*q6*q7 - 90*p**6*q1*q2*q3*q4**2*q5**2*q6 + 10*p**6*q1*q2*q3*q4**2*q5*q6**3 + 10*p**6*q1*q2*q3*q4**2*q5*q6**2*q7 - 90*p**6*q1*q2*q3*q4**2*q5*q6**2 + 10*p**6*q1*q2*q3*q4**2*q5*q6*q7**2 - 90*p**6*q1*q2*q3*q4**2*q5*q6*q7 + 360*p**6*q1*q2*q3*q4**2*q5*q6 + 10*p**6*q1*q2*q3*q4*q5**4*q6 + 10*p**6*q1*q2*q3*q4*q5**3*q6**2 + 10*p**6*q1*q2*q3*q4*q5**3*q6*q7 - 90*p**6*q1*q2*q3*q4*q5**3*q6 + 10*p**6*q1*q2*q3*q4*q5**2*q6**3 + 10*p**6*q1*q2*q3*q4*q5**2*q6**2*q7 - 90*p**6*q1*q2*q3*q4*q5**2*q6**2 + 10*p**6*q1*q2*q3*q4*q5**2*q6*q7**2 - 90*p**6*q1*q2*q3*q4*q5**2*q6*q7 + 360*p**6*q1*q2*q3*q4*q5**2*q6 + 10*p**6*q1*q2*q3*q4*q5*q6**4 + 10*p**6*q1*q2*q3*q4*q5*q6**3*q7 - 90*p**6*q1*q2*q3*q4*q5*q6**3 + 10*p**6*q1*q2*q3*q4*q5*q6**2*q7**2 - 90*p**6*q1*q2*q3*q4*q5*q6**2*q7 + 360*p**6*q1*q2*q3*q4*q5*q6**2 + 10*p**6*q1*q2*q3*q4*q5*q6*q7**3 - 90*p**6*q1*q2*q3*q4*q5*q6*q7**2 + 360*p**6*q1*q2*q3*q4*q5*q6*q7 - 840*p**6*q1*q2*q3*q4*q5*q6 - 45*p**6*q2**4*q3*q4*q5*q6 - 45*p**6*q2**3*q3**2*q4*q5*q6 - 45*p**6*q2**3*q3*q4**2*q5*q6 - 45*p**6*q2**3*q3*q4*q5**2*q6 - 45*p**6*q2**3*q3*q4*q5*q6**2 - 45*p**6*q2**3*q3*q4*q5*q6*q7 + 360*p**6*q2**3*q3*q4*q5*q6 - 45*p**6*q2**2*q3**3*q4*q5*q6 - 45*p**6*q2**2*q3**2*q4**2*q5*q6 - 45*p**6*q2**2*q3**2*q4*q5**2*q6 - 45*p**6*q2**2*q3**2*q4*q5*q6**2 - 45*p**6*q2**2*q3**2*q4*q5*q6*q7 + 360*p**6*q2**2*q3**2*q4*q5*q6 - 45*p**6*q2**2*q3*q4**3*q5*q6 - 45*p**6*q2**2*q3*q4**2*q5**2*q6 - 45*p**6*q2**2*q3*q4**2*q5*q6**2 - 45*p**6*q2**2*q3*q4**2*q5*q6*q7 + 360*p**6*q2**2*q3*q4**2*q5*q6 - 45*p**6*q2**2*q3*q4*q5**3*q6 - 45*p**6*q2**2*q3*q4*q5**2*q6**2 - 45*p**6*q2**2*q3*q4*q5**2*q6*q7 + 360*p**6*q2**2*q3*q4*q5**2*q6 - 45*p**6*q2**2*q3*q4*q5*q6**3 - 45*p**6*q2**2*q3*q4*q5*q6**2*q7 + 360*p**6*q2**2*q3*q4*q5*q6**2 - 45*p**6*q2**2*q3*q4*q5*q6*q7**2 + 360*p**6*q2**2*q3*q4*q5*q6*q7 - 1260*p**6*q2**2*q3*q4*q5*q6 - 45*p**6*q2*q3**4*q4*q5*q6 - 45*p**6*q2*q3**3*q4**2*q5*q6 - 45*p**6*q2*q3**3*q4*q5**2*q6 - 45*p**6*q2*q3**3*q4*q5*q6**2 - 45*p**6*q2*q3**3*q4*q5*q6*q7 + 360*p**6*q2*q3**3*q4*q5*q6 - 45*p**6*q2*q3**2*q4**3*q5*q6 - 45*p**6*q2*q3**2*q4**2*q5**2*q6 - 45*p**6*q2*q3**2*q4**2*q5*q6**2 - 45*p**6*q2*q3**2*q4**2*q5*q6*q7 + 360*p**6*q2*q3**2*q4**2*q5*q6 - 45*p**6*q2*q3**2*q4*q5**3*q6 - 45*p**6*q2*q3**2*q4*q5**2*q6**2 - 45*p**6*q2*q3**2*q4*q5**2*q6*q7 + 360*p**6*q2*q3**2*q4*q5**2*q6 - 45*p**6*q2*q3**2*q4*q5*q6**3 - 45*p**6*q2*q3**2*q4*q5*q6**2*q7 + 360*p**6*q2*q3**2*q4*q5*q6**2 - 45*p**6*q2*q3**2*q4*q5*q6*q7**2 + 360*p**6*q2*q3**2*q4*q5*q6*q7 - 1260*p**6*q2*q3**2*q4*q5*q6 - 45*p**6*q2*q3*q4**4*q5*q6 - 45*p**6*q2*q3*q4**3*q5**2*q6 - 45*p**6*q2*q3*q4**3*q5*q6**2 - 45*p**6*q2*q3*q4**3*q5*q6*q7 + 360*p**6*q2*q3*q4**3*q5*q6 - 45*p**6*q2*q3*q4**2*q5**3*q6 - 45*p**6*q2*q3*q4**2*q5**2*q6**2 - 45*p**6*q2*q3*q4**2*q5**2*q6*q7 + 360*p**6*q2*q3*q4**2*q5**2*q6 - 45*p**6*q2*q3*q4**2*q5*q6**3 - 45*p**6*q2*q3*q4**2*q5*q6**2*q7 + 360*p**6*q2*q3*q4**2*q5*q6**2 - 45*p**6*q2*q3*q4**2*q5*q6*q7**2 + 360*p**6*q2*q3*q4**2*q5*q6*q7 - 1260*p**6*q2*q3*q4**2*q5*q6 - 45*p**6*q2*q3*q4*q5**4*q6 - 45*p**6*q2*q3*q4*q5**3*q6**2 - 45*p**6*q2*q3*q4*q5**3*q6*q7 + 360*p**6*q2*q3*q4*q5**3*q6 - 45*p**6*q2*q3*q4*q5**2*q6**3 - 45*p**6*q2*q3*q4*q5**2*q6**2*q7 + 360*p**6*q2*q3*q4*q5**2*q6**2 - 45*p**6*q2*q3*q4*q5**2*q6*q7**2 + 360*p**6*q2*q3*q4*q5**2*q6*q7 - 1260*p**6*q2*q3*q4*q5**2*q6 - 45*p**6*q2*q3*q4*q5*q6**4 - 45*p**6*q2*q3*q4*q5*q6**3*q7 + 360*p**6*q2*q3*q4*q5*q6**3 - 45*p**6*q2*q3*q4*q5*q6**2*q7**2 + 360*p**6*q2*q3*q4*q5*q6**2*q7 - 1260*p**6*q2*q3*q4*q5*q6**2 - 45*p**6*q2*q3*q4*q5*q6*q7**3 + 360*p**6*q2*q3*q4*q5*q6*q7**2 - 1260*p**6*q2*q3*q4*q5*q6*q7 + 2520*p**6*q2*q3*q4*q5*q6 + 120*p**6*q3**4*q4*q5*q6 + 120*p**6*q3**3*q4**2*q5*q6 + 120*p**6*q3**3*q4*q5**2*q6 + 120*p**6*q3**3*q4*q5*q6**2 + 120*p**6*q3**3*q4*q5*q6*q7 - 840*p**6*q3**3*q4*q5*q6 + 120*p**6*q3**2*q4**3*q5*q6 + 120*p**6*q3**2*q4**2*q5**2*q6 + 120*p**6*q3**2*q4**2*q5*q6**2 + 120*p**6*q3**2*q4**2*q5*q6*q7 - 840*p**6*q3**2*q4**2*q5*q6 + 120*p**6*q3**2*q4*q5**3*q6 + 120*p**6*q3**2*q4*q5**2*q6**2 + 120*p**6*q3**2*q4*q5**2*q6*q7 - 840*p**6*q3**2*q4*q5**2*q6 + 120*p**6*q3**2*q4*q5*q6**3 + 120*p**6*q3**2*q4*q5*q6**2*q7 - 840*p**6*q3**2*q4*q5*q6**2 + 120*p**6*q3**2*q4*q5*q6*q7**2 - 840*p**6*q3**2*q4*q5*q6*q7 + 2520*p**6*q3**2*q4*q5*q6 + 120*p**6*q3*q4**4*q5*q6 + 120*p**6*q3*q4**3*q5**2*q6 + 120*p**6*q3*q4**3*q5*q6**2 + 120*p**6*q3*q4**3*q5*q6*q7 - 840*p**6*q3*q4**3*q5*q6 + 120*p**6*q3*q4**2*q5**3*q6 + 120*p**6*q3*q4**2*q5**2*q6**2 + 120*p**6*q3*q4**2*q5**2*q6*q7 - 840*p**6*q3*q4**2*q5**2*q6 + 120*p**6*q3*q4**2*q5*q6**3 + 120*p**6*q3*q4**2*q5*q6**2*q7 - 840*p**6*q3*q4**2*q5*q6**2 + 120*p**6*q3*q4**2*q5*q6*q7**2 - 840*p**6*q3*q4**2*q5*q6*q7 + 2520*p**6*q3*q4**2*q5*q6 + 120*p**6*q3*q4*q5**4*q6 + 120*p**6*q3*q4*q5**3*q6**2 + 120*p**6*q3*q4*q5**3*q6*q7 - 840*p**6*q3*q4*q5**3*q6 + 120*p**6*q3*q4*q5**2*q6**3 + 120*p**6*q3*q4*q5**2*q6**2*q7 - 840*p**6*q3*q4*q5**2*q6**2 + 120*p**6*q3*q4*q5**2*q6*q7**2 - 840*p**6*q3*q4*q5**2*q6*q7 + 2520*p**6*q3*q4*q5**2*q6 + 120*p**6*q3*q4*q5*q6**4 + 120*p**6*q3*q4*q5*q6**3*q7 - 840*p**6*q3*q4*q5*q6**3 + 120*p**6*q3*q4*q5*q6**2*q7**2 - 840*p**6*q3*q4*q5*q6**2*q7 + 2520*p**6*q3*q4*q5*q6**2 + 120*p**6*q3*q4*q5*q6*q7**3 - 840*p**6*q3*q4*q5*q6*q7**2 + 2520*p**6*q3*q4*q5*q6*q7 - 4200*p**6*q3*q4*q5*q6 - 210*p**6*q4**4*q5*q6 - 210*p**6*q4**3*q5**2*q6 - 210*p**6*q4**3*q5*q6**2 - 210*p**6*q4**3*q5*q6*q7 + 1260*p**6*q4**3*q5*q6 - 210*p**6*q4**2*q5**3*q6 - 210*p**6*q4**2*q5**2*q6**2 - 210*p**6*q4**2*q5**2*q6*q7 + 1260*p**6*q4**2*q5**2*q6 - 210*p**6*q4**2*q5*q6**3 - 210*p**6*q4**2*q5*q6**2*q7 + 1260*p**6*q4**2*q5*q6**2 - 210*p**6*q4**2*q5*q6*q7**2 + 1260*p**6*q4**2*q5*q6*q7 - 3150*p**6*q4**2*q5*q6 - 210*p**6*q4*q5**4*q6 - 210*p**6*q4*q5**3*q6**2 - 210*p**6*q4*q5**3*q6*q7 + 1260*p**6*q4*q5**3*q6 - 210*p**6*q4*q5**2*q6**3 - 210*p**6*q4*q5**2*q6**2*q7 + 1260*p**6*q4*q5**2*q6**2 - 210*p**6*q4*q5**2*q6*q7**2 + 1260*p**6*q4*q5**2*q6*q7 - 3150*p**6*q4*q5**2*q6 - 210*p**6*q4*q5*q6**4 - 210*p**6*q4*q5*q6**3*q7 + 1260*p**6*q4*q5*q6**3 - 210*p**6*q4*q5*q6**2*q7**2 + 1260*p**6*q4*q5*q6**2*q7 - 3150*p**6*q4*q5*q6**2 - 210*p**6*q4*q5*q6*q7**3 + 1260*p**6*q4*q5*q6*q7**2 - 3150*p**6*q4*q5*q6*q7 + 4200*p**6*q4*q5*q6 + 252*p**6*q5**4*q6 + 252*p**6*q5**3*q6**2 + 252*p**6*q5**3*q6*q7 - 1260*p**6*q5**3*q6 + 252*p**6*q5**2*q6**3 + 252*p**6*q5**2*q6**2*q7 - 1260*p**6*q5**2*q6**2 + 252*p**6*q5**2*q6*q7**2 - 1260*p**6*q5**2*q6*q7 + 2520*p**6*q5**2*q6 + 252*p**6*q5*q6**4 + 252*p**6*q5*q6**3*q7 - 1260*p**6*q5*q6**3 + 252*p**6*q5*q6**2*q7**2 - 1260*p**6*q5*q6**2*q7 + 2520*p**6*q5*q6**2 + 252*p**6*q5*q6*q7**3 - 1260*p**6*q5*q6*q7**2 + 2520*p**6*q5*q6*q7 - 2520*p**6*q5*q6 - 210*p**6*q6**4 - 210*p**6*q6**3*q7 + 840*p**6*q6**3 - 210*p**6*q6**2*q7**2 + 840*p**6*q6**2*q7 - 1260*p**6*q6**2 - 210*p**6*q6*q7**3 + 840*p**6*q6*q7**2 - 1260*p**6*q6*q7 + 840*p**6*q6 + 120*p**6*q7**3 - 360*p**6*q7**2 + 360*p**6*q7 - 120*p**6 - 60*p**5*q1**4*q2*q3*q4*q5*q6 - 60*p**5*q1**3*q2**2*q3*q4*q5*q6 - 60*p**5*q1**3*q2*q3**2*q4*q5*q6 - 60*p**5*q1**3*q2*q3*q4**2*q5*q6 - 60*p**5*q1**3*q2*q3*q4*q5**2*q6 - 60*p**5*q1**3*q2*q3*q4*q5*q6**2 - 60*p**5*q1**3*q2*q3*q4*q5*q6*q7 + 540*p**5*q1**3*q2*q3*q4*q5*q6 - 60*p**5*q1**2*q2**3*q3*q4*q5*q6 - 60*p**5*q1**2*q2**2*q3**2*q4*q5*q6 - 60*p**5*q1**2*q2**2*q3*q4**2*q5*q6 - 60*p**5*q1**2*q2**2*q3*q4*q5**2*q6 - 60*p**5*q1**2*q2**2*q3*q4*q5*q6**2 - 60*p**5*q1**2*q2**2*q3*q4*q5*q6*q7 + 540*p**5*q1**2*q2**2*q3*q4*q5*q6 - 60*p**5*q1**2*q2*q3**3*q4*q5*q6 - 60*p**5*q1**2*q2*q3**2*q4**2*q5*q6 - 60*p**5*q1**2*q2*q3**2*q4*q5**2*q6 - 60*p**5*q1**2*q2*q3**2*q4*q5*q6**2 - 60*p**5*q1**2*q2*q3**2*q4*q5*q6*q7 + 540*p**5*q1**2*q2*q3**2*q4*q5*q6 - 60*p**5*q1**2*q2*q3*q4**3*q5*q6 - 60*p**5*q1**2*q2*q3*q4**2*q5**2*q6 - 60*p**5*q1**2*q2*q3*q4**2*q5*q6**2 - 60*p**5*q1**2*q2*q3*q4**2*q5*q6*q7 + 540*p**5*q1**2*q2*q3*q4**2*q5*q6 - 60*p**5*q1**2*q2*q3*q4*q5**3*q6 - 60*p**5*q1**2*q2*q3*q4*q5**2*q6**2 - 60*p**5*q1**2*q2*q3*q4*q5**2*q6*q7 + 540*p**5*q1**2*q2*q3*q4*q5**2*q6 - 60*p**5*q1**2*q2*q3*q4*q5*q6**3 - 60*p**5*q1**2*q2*q3*q4*q5*q6**2*q7 + 540*p**5*q1**2*q2*q3*q4*q5*q6**2 - 60*p**5*q1**2*q2*q3*q4*q5*q6*q7**2 + 540*p**5*q1**2*q2*q3*q4*q5*q6*q7 - 2160*p**5*q1**2*q2*q3*q4*q5*q6 - 60*p**5*q1*q2**4*q3*q4*q5*q6 - 60*p**5*q1*q2**3*q3**2*q4*q5*q6 - 60*p**5*q1*q2**3*q3*q4**2*q5*q6 - 60*p**5*q1*q2**3*q3*q4*q5**2*q6 - 60*p**5*q1*q2**3*q3*q4*q5*q6**2 - 60*p**5*q1*q2**3*q3*q4*q5*q6*q7 + 540*p**5*q1*q2**3*q3*q4*q5*q6 - 60*p**5*q1*q2**2*q3**3*q4*q5*q6 - 60*p**5*q1*q2**2*q3**2*q4**2*q5*q6 - 60*p**5*q1*q2**2*q3**2*q4*q5**2*q6 - 60*p**5*q1*q2**2*q3**2*q4*q5*q6**2 - 60*p**5*q1*q2**2*q3**2*q4*q5*q6*q7 + 540*p**5*q1*q2**2*q3**2*q4*q5*q6 - 60*p**5*q1*q2**2*q3*q4**3*q5*q6 - 60*p**5*q1*q2**2*q3*q4**2*q5**2*q6 - 60*p**5*q1*q2**2*q3*q4**2*q5*q6**2 - 60*p**5*q1*q2**2*q3*q4**2*q5*q6*q7 + 540*p**5*q1*q2**2*q3*q4**2*q5*q6 - 60*p**5*q1*q2**2*q3*q4*q5**3*q6 - 60*p**5*q1*q2**2*q3*q4*q5**2*q6**2 - 60*p**5*q1*q2**2*q3*q4*q5**2*q6*q7 + 540*p**5*q1*q2**2*q3*q4*q5**2*q6 - 60*p**5*q1*q2**2*q3*q4*q5*q6**3 - 60*p**5*q1*q2**2*q3*q4*q5*q6**2*q7 + 540*p**5*q1*q2**2*q3*q4*q5*q6**2 - 60*p**5*q1*q2**2*q3*q4*q5*q6*q7**2 + 540*p**5*q1*q2**2*q3*q4*q5*q6*q7 - 2160*p**5*q1*q2**2*q3*q4*q5*q6 - 60*p**5*q1*q2*q3**4*q4*q5*q6 - 60*p**5*q1*q2*q3**3*q4**2*q5*q6 - 60*p**5*q1*q2*q3**3*q4*q5**2*q6 - 60*p**5*q1*q2*q3**3*q4*q5*q6**2 - 60*p**5*q1*q2*q3**3*q4*q5*q6*q7 + 540*p**5*q1*q2*q3**3*q4*q5*q6 - 60*p**5*q1*q2*q3**2*q4**3*q5*q6 - 60*p**5*q1*q2*q3**2*q4**2*q5**2*q6 - 60*p**5*q1*q2*q3**2*q4**2*q5*q6**2 - 60*p**5*q1*q2*q3**2*q4**2*q5*q6*q7 + 540*p**5*q1*q2*q3**2*q4**2*q5*q6 - 60*p**5*q1*q2*q3**2*q4*q5**3*q6 - 60*p**5*q1*q2*q3**2*q4*q5**2*q6**2 - 60*p**5*q1*q2*q3**2*q4*q5**2*q6*q7 + 540*p**5*q1*q2*q3**2*q4*q5**2*q6 - 60*p**5*q1*q2*q3**2*q4*q5*q6**3 - 60*p**5*q1*q2*q3**2*q4*q5*q6**2*q7 + 540*p**5*q1*q2*q3**2*q4*q5*q6**2 - 60*p**5*q1*q2*q3**2*q4*q5*q6*q7**2 + 540*p**5*q1*q2*q3**2*q4*q5*q6*q7 - 2160*p**5*q1*q2*q3**2*q4*q5*q6 - 60*p**5*q1*q2*q3*q4**4*q5*q6 - 60*p**5*q1*q2*q3*q4**3*q5**2*q6 - 60*p**5*q1*q2*q3*q4**3*q5*q6**2 - 60*p**5*q1*q2*q3*q4**3*q5*q6*q7 + 540*p**5*q1*q2*q3*q4**3*q5*q6 - 60*p**5*q1*q2*q3*q4**2*q5**3*q6 - 60*p**5*q1*q2*q3*q4**2*q5**2*q6**2 - 60*p**5*q1*q2*q3*q4**2*q5**2*q6*q7 + 540*p**5*q1*q2*q3*q4**2*q5**2*q6 - 60*p**5*q1*q2*q3*q4**2*q5*q6**3 - 60*p**5*q1*q2*q3*q4**2*q5*q6**2*q7 + 540*p**5*q1*q2*q3*q4**2*q5*q6**2 - 60*p**5*q1*q2*q3*q4**2*q5*q6*q7**2 + 540*p**5*q1*q2*q3*q4**2*q5*q6*q7 - 2160*p**5*q1*q2*q3*q4**2*q5*q6 - 60*p**5*q1*q2*q3*q4*q5**4*q6 - 60*p**5*q1*q2*q3*q4*q5**3*q6**2 - 60*p**5*q1*q2*q3*q4*q5**3*q6*q7 + 540*p**5*q1*q2*q3*q4*q5**3*q6 - 60*p**5*q1*q2*q3*q4*q5**2*q6**3 - 60*p**5*q1*q2*q3*q4*q5**2*q6**2*q7 + 540*p**5*q1*q2*q3*q4*q5**2*q6**2 - 60*p**5*q1*q2*q3*q4*q5**2*q6*q7**2 + 540*p**5*q1*q2*q3*q4*q5**2*q6*q7 - 2160*p**5*q1*q2*q3*q4*q5**2*q6 - 60*p**5*q1*q2*q3*q4*q5*q6**4 - 60*p**5*q1*q2*q3*q4*q5*q6**3*q7 + 540*p**5*q1*q2*q3*q4*q5*q6**3 - 60*p**5*q1*q2*q3*q4*q5*q6**2*q7**2 + 540*p**5*q1*q2*q3*q4*q5*q6**2*q7 - 2160*p**5*q1*q2*q3*q4*q5*q6**2 - 60*p**5*q1*q2*q3*q4*q5*q6*q7**3 + 540*p**5*q1*q2*q3*q4*q5*q6*q7**2 - 2160*p**5*q1*q2*q3*q4*q5*q6*q7 + 5040*p**5*q1*q2*q3*q4*q5*q6 + 225*p**5*q2**4*q3*q4*q5*q6 + 225*p**5*q2**3*q3**2*q4*q5*q6 + 225*p**5*q2**3*q3*q4**2*q5*q6 + 225*p**5*q2**3*q3*q4*q5**2*q6 + 225*p**5*q2**3*q3*q4*q5*q6**2 + 225*p**5*q2**3*q3*q4*q5*q6*q7 - 1800*p**5*q2**3*q3*q4*q5*q6 + 225*p**5*q2**2*q3**3*q4*q5*q6 + 225*p**5*q2**2*q3**2*q4**2*q5*q6 + 225*p**5*q2**2*q3**2*q4*q5**2*q6 + 225*p**5*q2**2*q3**2*q4*q5*q6**2 + 225*p**5*q2**2*q3**2*q4*q5*q6*q7 - 1800*p**5*q2**2*q3**2*q4*q5*q6 + 225*p**5*q2**2*q3*q4**3*q5*q6 + 225*p**5*q2**2*q3*q4**2*q5**2*q6 + 225*p**5*q2**2*q3*q4**2*q5*q6**2 + 225*p**5*q2**2*q3*q4**2*q5*q6*q7 - 1800*p**5*q2**2*q3*q4**2*q5*q6 + 225*p**5*q2**2*q3*q4*q5**3*q6 + 225*p**5*q2**2*q3*q4*q5**2*q6**2 + 225*p**5*q2**2*q3*q4*q5**2*q6*q7 - 1800*p**5*q2**2*q3*q4*q5**2*q6 + 225*p**5*q2**2*q3*q4*q5*q6**3 + 225*p**5*q2**2*q3*q4*q5*q6**2*q7 - 1800*p**5*q2**2*q3*q4*q5*q6**2 + 225*p**5*q2**2*q3*q4*q5*q6*q7**2 - 1800*p**5*q2**2*q3*q4*q5*q6*q7 + 6300*p**5*q2**2*q3*q4*q5*q6 + 225*p**5*q2*q3**4*q4*q5*q6 + 225*p**5*q2*q3**3*q4**2*q5*q6 + 225*p**5*q2*q3**3*q4*q5**2*q6 + 225*p**5*q2*q3**3*q4*q5*q6**2 + 225*p**5*q2*q3**3*q4*q5*q6*q7 - 1800*p**5*q2*q3**3*q4*q5*q6 + 225*p**5*q2*q3**2*q4**3*q5*q6 + 225*p**5*q2*q3**2*q4**2*q5**2*q6 + 225*p**5*q2*q3**2*q4**2*q5*q6**2 + 225*p**5*q2*q3**2*q4**2*q5*q6*q7 - 1800*p**5*q2*q3**2*q4**2*q5*q6 + 225*p**5*q2*q3**2*q4*q5**3*q6 + 225*p**5*q2*q3**2*q4*q5**2*q6**2 + 225*p**5*q2*q3**2*q4*q5**2*q6*q7 - 1800*p**5*q2*q3**2*q4*q5**2*q6 + 225*p**5*q2*q3**2*q4*q5*q6**3 + 225*p**5*q2*q3**2*q4*q5*q6**2*q7 - 1800*p**5*q2*q3**2*q4*q5*q6**2 + 225*p**5*q2*q3**2*q4*q5*q6*q7**2 - 1800*p**5*q2*q3**2*q4*q5*q6*q7 + 6300*p**5*q2*q3**2*q4*q5*q6 + 225*p**5*q2*q3*q4**4*q5*q6 + 225*p**5*q2*q3*q4**3*q5**2*q6 + 225*p**5*q2*q3*q4**3*q5*q6**2 + 225*p**5*q2*q3*q4**3*q5*q6*q7 - 1800*p**5*q2*q3*q4**3*q5*q6 + 225*p**5*q2*q3*q4**2*q5**3*q6 + 225*p**5*q2*q3*q4**2*q5**2*q6**2 + 225*p**5*q2*q3*q4**2*q5**2*q6*q7 - 1800*p**5*q2*q3*q4**2*q5**2*q6 + 225*p**5*q2*q3*q4**2*q5*q6**3 + 225*p**5*q2*q3*q4**2*q5*q6**2*q7 - 1800*p**5*q2*q3*q4**2*q5*q6**2 + 225*p**5*q2*q3*q4**2*q5*q6*q7**2 - 1800*p**5*q2*q3*q4**2*q5*q6*q7 + 6300*p**5*q2*q3*q4**2*q5*q6 + 225*p**5*q2*q3*q4*q5**4*q6 + 225*p**5*q2*q3*q4*q5**3*q6**2 + 225*p**5*q2*q3*q4*q5**3*q6*q7 - 1800*p**5*q2*q3*q4*q5**3*q6 + 225*p**5*q2*q3*q4*q5**2*q6**3 + 225*p**5*q2*q3*q4*q5**2*q6**2*q7 - 1800*p**5*q2*q3*q4*q5**2*q6**2 + 225*p**5*q2*q3*q4*q5**2*q6*q7**2 - 1800*p**5*q2*q3*q4*q5**2*q6*q7 + 6300*p**5*q2*q3*q4*q5**2*q6 + 225*p**5*q2*q3*q4*q5*q6**4 + 225*p**5*q2*q3*q4*q5*q6**3*q7 - 1800*p**5*q2*q3*q4*q5*q6**3 + 225*p**5*q2*q3*q4*q5*q6**2*q7**2 - 1800*p**5*q2*q3*q4*q5*q6**2*q7 + 6300*p**5*q2*q3*q4*q5*q6**2 + 225*p**5*q2*q3*q4*q5*q6*q7**3 - 1800*p**5*q2*q3*q4*q5*q6*q7**2 + 6300*p**5*q2*q3*q4*q5*q6*q7 - 12600*p**5*q2*q3*q4*q5*q6 - 480*p**5*q3**4*q4*q5*q6 - 480*p**5*q3**3*q4**2*q5*q6 - 480*p**5*q3**3*q4*q5**2*q6 - 480*p**5*q3**3*q4*q5*q6**2 - 480*p**5*q3**3*q4*q5*q6*q7 + 3360*p**5*q3**3*q4*q5*q6 - 480*p**5*q3**2*q4**3*q5*q6 - 480*p**5*q3**2*q4**2*q5**2*q6 - 480*p**5*q3**2*q4**2*q5*q6**2 - 480*p**5*q3**2*q4**2*q5*q6*q7 + 3360*p**5*q3**2*q4**2*q5*q6 - 480*p**5*q3**2*q4*q5**3*q6 - 480*p**5*q3**2*q4*q5**2*q6**2 - 480*p**5*q3**2*q4*q5**2*q6*q7 + 3360*p**5*q3**2*q4*q5**2*q6 - 480*p**5*q3**2*q4*q5*q6**3 - 480*p**5*q3**2*q4*q5*q6**2*q7 + 3360*p**5*q3**2*q4*q5*q6**2 - 480*p**5*q3**2*q4*q5*q6*q7**2 + 3360*p**5*q3**2*q4*q5*q6*q7 - 10080*p**5*q3**2*q4*q5*q6 - 480*p**5*q3*q4**4*q5*q6 - 480*p**5*q3*q4**3*q5**2*q6 - 480*p**5*q3*q4**3*q5*q6**2 - 480*p**5*q3*q4**3*q5*q6*q7 + 3360*p**5*q3*q4**3*q5*q6 - 480*p**5*q3*q4**2*q5**3*q6 - 480*p**5*q3*q4**2*q5**2*q6**2 - 480*p**5*q3*q4**2*q5**2*q6*q7 + 3360*p**5*q3*q4**2*q5**2*q6 - 480*p**5*q3*q4**2*q5*q6**3 - 480*p**5*q3*q4**2*q5*q6**2*q7 + 3360*p**5*q3*q4**2*q5*q6**2 - 480*p**5*q3*q4**2*q5*q6*q7**2 + 3360*p**5*q3*q4**2*q5*q6*q7 - 10080*p**5*q3*q4**2*q5*q6 - 480*p**5*q3*q4*q5**4*q6 - 480*p**5*q3*q4*q5**3*q6**2 - 480*p**5*q3*q4*q5**3*q6*q7 + 3360*p**5*q3*q4*q5**3*q6 - 480*p**5*q3*q4*q5**2*q6**3 - 480*p**5*q3*q4*q5**2*q6**2*q7 + 3360*p**5*q3*q4*q5**2*q6**2 - 480*p**5*q3*q4*q5**2*q6*q7**2 + 3360*p**5*q3*q4*q5**2*q6*q7 - 10080*p**5*q3*q4*q5**2*q6 - 480*p**5*q3*q4*q5*q6**4 - 480*p**5*q3*q4*q5*q6**3*q7 + 3360*p**5*q3*q4*q5*q6**3 - 480*p**5*q3*q4*q5*q6**2*q7**2 + 3360*p**5*q3*q4*q5*q6**2*q7 - 10080*p**5*q3*q4*q5*q6**2 - 480*p**5*q3*q4*q5*q6*q7**3 + 3360*p**5*q3*q4*q5*q6*q7**2 - 10080*p**5*q3*q4*q5*q6*q7 + 16800*p**5*q3*q4*q5*q6 + 630*p**5*q4**4*q5*q6 + 630*p**5*q4**3*q5**2*q6 + 630*p**5*q4**3*q5*q6**2 + 630*p**5*q4**3*q5*q6*q7 - 3780*p**5*q4**3*q5*q6 + 630*p**5*q4**2*q5**3*q6 + 630*p**5*q4**2*q5**2*q6**2 + 630*p**5*q4**2*q5**2*q6*q7 - 3780*p**5*q4**2*q5**2*q6 + 630*p**5*q4**2*q5*q6**3 + 630*p**5*q4**2*q5*q6**2*q7 - 3780*p**5*q4**2*q5*q6**2 + 630*p**5*q4**2*q5*q6*q7**2 - 3780*p**5*q4**2*q5*q6*q7 + 9450*p**5*q4**2*q5*q6 + 630*p**5*q4*q5**4*q6 + 630*p**5*q4*q5**3*q6**2 + 630*p**5*q4*q5**3*q6*q7 - 3780*p**5*q4*q5**3*q6 + 630*p**5*q4*q5**2*q6**3 + 630*p**5*q4*q5**2*q6**2*q7 - 3780*p**5*q4*q5**2*q6**2 + 630*p**5*q4*q5**2*q6*q7**2 - 3780*p**5*q4*q5**2*q6*q7 + 9450*p**5*q4*q5**2*q6 + 630*p**5*q4*q5*q6**4 + 630*p**5*q4*q5*q6**3*q7 - 3780*p**5*q4*q5*q6**3 + 630*p**5*q4*q5*q6**2*q7**2 - 3780*p**5*q4*q5*q6**2*q7 + 9450*p**5*q4*q5*q6**2 + 630*p**5*q4*q5*q6*q7**3 - 3780*p**5*q4*q5*q6*q7**2 + 9450*p**5*q4*q5*q6*q7 - 12600*p**5*q4*q5*q6 - 504*p**5*q5**4*q6 - 504*p**5*q5**3*q6**2 - 504*p**5*q5**3*q6*q7 + 2520*p**5*q5**3*q6 - 504*p**5*q5**2*q6**3 - 504*p**5*q5**2*q6**2*q7 + 2520*p**5*q5**2*q6**2 - 504*p**5*q5**2*q6*q7**2 + 2520*p**5*q5**2*q6*q7 - 5040*p**5*q5**2*q6 - 504*p**5*q5*q6**4 - 504*p**5*q5*q6**3*q7 + 2520*p**5*q5*q6**3 - 504*p**5*q5*q6**2*q7**2 + 2520*p**5*q5*q6**2*q7 - 5040*p**5*q5*q6**2 - 504*p**5*q5*q6*q7**3 + 2520*p**5*q5*q6*q7**2 - 5040*p**5*q5*q6*q7 + 5040*p**5*q5*q6 + 210*p**5*q6**4 + 210*p**5*q6**3*q7 - 840*p**5*q6**3 + 210*p**5*q6**2*q7**2 - 840*p**5*q6**2*q7 + 1260*p**5*q6**2 + 210*p**5*q6*q7**3 - 840*p**5*q6*q7**2 + 1260*p**5*q6*q7 - 840*p**5*q6 + 150*p**4*q1**4*q2*q3*q4*q5*q6 + 150*p**4*q1**3*q2**2*q3*q4*q5*q6 + 150*p**4*q1**3*q2*q3**2*q4*q5*q6 + 150*p**4*q1**3*q2*q3*q4**2*q5*q6 + 150*p**4*q1**3*q2*q3*q4*q5**2*q6 + 150*p**4*q1**3*q2*q3*q4*q5*q6**2 + 150*p**4*q1**3*q2*q3*q4*q5*q6*q7 - 1350*p**4*q1**3*q2*q3*q4*q5*q6 + 150*p**4*q1**2*q2**3*q3*q4*q5*q6 + 150*p**4*q1**2*q2**2*q3**2*q4*q5*q6 + 150*p**4*q1**2*q2**2*q3*q4**2*q5*q6 + 150*p**4*q1**2*q2**2*q3*q4*q5**2*q6 + 150*p**4*q1**2*q2**2*q3*q4*q5*q6**2 + 150*p**4*q1**2*q2**2*q3*q4*q5*q6*q7 - 1350*p**4*q1**2*q2**2*q3*q4*q5*q6 + 150*p**4*q1**2*q2*q3**3*q4*q5*q6 + 150*p**4*q1**2*q2*q3**2*q4**2*q5*q6 + 150*p**4*q1**2*q2*q3**2*q4*q5**2*q6 + 150*p**4*q1**2*q2*q3**2*q4*q5*q6**2 + 150*p**4*q1**2*q2*q3**2*q4*q5*q6*q7 - 1350*p**4*q1**2*q2*q3**2*q4*q5*q6 + 150*p**4*q1**2*q2*q3*q4**3*q5*q6 + 150*p**4*q1**2*q2*q3*q4**2*q5**2*q6 + 150*p**4*q1**2*q2*q3*q4**2*q5*q6**2 + 150*p**4*q1**2*q2*q3*q4**2*q5*q6*q7 - 1350*p**4*q1**2*q2*q3*q4**2*q5*q6 + 150*p**4*q1**2*q2*q3*q4*q5**3*q6 + 150*p**4*q1**2*q2*q3*q4*q5**2*q6**2 + 150*p**4*q1**2*q2*q3*q4*q5**2*q6*q7 - 1350*p**4*q1**2*q2*q3*q4*q5**2*q6 + 150*p**4*q1**2*q2*q3*q4*q5*q6**3 + 150*p**4*q1**2*q2*q3*q4*q5*q6**2*q7 - 1350*p**4*q1**2*q2*q3*q4*q5*q6**2 + 150*p**4*q1**2*q2*q3*q4*q5*q6*q7**2 - 1350*p**4*q1**2*q2*q3*q4*q5*q6*q7 + 5400*p**4*q1**2*q2*q3*q4*q5*q6 + 150*p**4*q1*q2**4*q3*q4*q5*q6 + 150*p**4*q1*q2**3*q3**2*q4*q5*q6 + 150*p**4*q1*q2**3*q3*q4**2*q5*q6 + 150*p**4*q1*q2**3*q3*q4*q5**2*q6 + 150*p**4*q1*q2**3*q3*q4*q5*q6**2 + 150*p**4*q1*q2**3*q3*q4*q5*q6*q7 - 1350*p**4*q1*q2**3*q3*q4*q5*q6 + 150*p**4*q1*q2**2*q3**3*q4*q5*q6 + 150*p**4*q1*q2**2*q3**2*q4**2*q5*q6 + 150*p**4*q1*q2**2*q3**2*q4*q5**2*q6 + 150*p**4*q1*q2**2*q3**2*q4*q5*q6**2 + 150*p**4*q1*q2**2*q3**2*q4*q5*q6*q7 - 1350*p**4*q1*q2**2*q3**2*q4*q5*q6 + 150*p**4*q1*q2**2*q3*q4**3*q5*q6 + 150*p**4*q1*q2**2*q3*q4**2*q5**2*q6 + 150*p**4*q1*q2**2*q3*q4**2*q5*q6**2 + 150*p**4*q1*q2**2*q3*q4**2*q5*q6*q7 - 1350*p**4*q1*q2**2*q3*q4**2*q5*q6 + 150*p**4*q1*q2**2*q3*q4*q5**3*q6 + 150*p**4*q1*q2**2*q3*q4*q5**2*q6**2 + 150*p**4*q1*q2**2*q3*q4*q5**2*q6*q7 - 1350*p**4*q1*q2**2*q3*q4*q5**2*q6 + 150*p**4*q1*q2**2*q3*q4*q5*q6**3 + 150*p**4*q1*q2**2*q3*q4*q5*q6**2*q7 - 1350*p**4*q1*q2**2*q3*q4*q5*q6**2 + 150*p**4*q1*q2**2*q3*q4*q5*q6*q7**2 - 1350*p**4*q1*q2**2*q3*q4*q5*q6*q7 + 5400*p**4*q1*q2**2*q3*q4*q5*q6 + 150*p**4*q1*q2*q3**4*q4*q5*q6 + 150*p**4*q1*q2*q3**3*q4**2*q5*q6 + 150*p**4*q1*q2*q3**3*q4*q5**2*q6 + 150*p**4*q1*q2*q3**3*q4*q5*q6**2 + 150*p**4*q1*q2*q3**3*q4*q5*q6*q7 - 1350*p**4*q1*q2*q3**3*q4*q5*q6 + 150*p**4*q1*q2*q3**2*q4**3*q5*q6 + 150*p**4*q1*q2*q3**2*q4**2*q5**2*q6 + 150*p**4*q1*q2*q3**2*q4**2*q5*q6**2 + 150*p**4*q1*q2*q3**2*q4**2*q5*q6*q7 - 1350*p**4*q1*q2*q3**2*q4**2*q5*q6 + 150*p**4*q1*q2*q3**2*q4*q5**3*q6 + 150*p**4*q1*q2*q3**2*q4*q5**2*q6**2 + 150*p**4*q1*q2*q3**2*q4*q5**2*q6*q7 - 1350*p**4*q1*q2*q3**2*q4*q5**2*q6 + 150*p**4*q1*q2*q3**2*q4*q5*q6**3 + 150*p**4*q1*q2*q3**2*q4*q5*q6**2*q7 - 1350*p**4*q1*q2*q3**2*q4*q5*q6**2 + 150*p**4*q1*q2*q3**2*q4*q5*q6*q7**2 - 1350*p**4*q1*q2*q3**2*q4*q5*q6*q7 + 5400*p**4*q1*q2*q3**2*q4*q5*q6 + 150*p**4*q1*q2*q3*q4**4*q5*q6 + 150*p**4*q1*q2*q3*q4**3*q5**2*q6 + 150*p**4*q1*q2*q3*q4**3*q5*q6**2 + 150*p**4*q1*q2*q3*q4**3*q5*q6*q7 - 1350*p**4*q1*q2*q3*q4**3*q5*q6 + 150*p**4*q1*q2*q3*q4**2*q5**3*q6 + 150*p**4*q1*q2*q3*q4**2*q5**2*q6**2 + 150*p**4*q1*q2*q3*q4**2*q5**2*q6*q7 - 1350*p**4*q1*q2*q3*q4**2*q5**2*q6 + 150*p**4*q1*q2*q3*q4**2*q5*q6**3 + 150*p**4*q1*q2*q3*q4**2*q5*q6**2*q7 - 1350*p**4*q1*q2*q3*q4**2*q5*q6**2 + 150*p**4*q1*q2*q3*q4**2*q5*q6*q7**2 - 1350*p**4*q1*q2*q3*q4**2*q5*q6*q7 + 5400*p**4*q1*q2*q3*q4**2*q5*q6 + 150*p**4*q1*q2*q3*q4*q5**4*q6 + 150*p**4*q1*q2*q3*q4*q5**3*q6**2 + 150*p**4*q1*q2*q3*q4*q5**3*q6*q7 - 1350*p**4*q1*q2*q3*q4*q5**3*q6 + 150*p**4*q1*q2*q3*q4*q5**2*q6**3 + 150*p**4*q1*q2*q3*q4*q5**2*q6**2*q7 - 1350*p**4*q1*q2*q3*q4*q5**2*q6**2 + 150*p**4*q1*q2*q3*q4*q5**2*q6*q7**2 - 1350*p**4*q1*q2*q3*q4*q5**2*q6*q7 + 5400*p**4*q1*q2*q3*q4*q5**2*q6 + 150*p**4*q1*q2*q3*q4*q5*q6**4 + 150*p**4*q1*q2*q3*q4*q5*q6**3*q7 - 1350*p**4*q1*q2*q3*q4*q5*q6**3 + 150*p**4*q1*q2*q3*q4*q5*q6**2*q7**2 - 1350*p**4*q1*q2*q3*q4*q5*q6**2*q7 + 5400*p**4*q1*q2*q3*q4*q5*q6**2 + 150*p**4*q1*q2*q3*q4*q5*q6*q7**3 - 1350*p**4*q1*q2*q3*q4*q5*q6*q7**2 + 5400*p**4*q1*q2*q3*q4*q5*q6*q7 - 12600*p**4*q1*q2*q3*q4*q5*q6 - 450*p**4*q2**4*q3*q4*q5*q6 - 450*p**4*q2**3*q3**2*q4*q5*q6 - 450*p**4*q2**3*q3*q4**2*q5*q6 - 450*p**4*q2**3*q3*q4*q5**2*q6 - 450*p**4*q2**3*q3*q4*q5*q6**2 - 450*p**4*q2**3*q3*q4*q5*q6*q7 + 3600*p**4*q2**3*q3*q4*q5*q6 - 450*p**4*q2**2*q3**3*q4*q5*q6 - 450*p**4*q2**2*q3**2*q4**2*q5*q6 - 450*p**4*q2**2*q3**2*q4*q5**2*q6 - 450*p**4*q2**2*q3**2*q4*q5*q6**2 - 450*p**4*q2**2*q3**2*q4*q5*q6*q7 + 3600*p**4*q2**2*q3**2*q4*q5*q6 - 450*p**4*q2**2*q3*q4**3*q5*q6 - 450*p**4*q2**2*q3*q4**2*q5**2*q6 - 450*p**4*q2**2*q3*q4**2*q5*q6**2 - 450*p**4*q2**2*q3*q4**2*q5*q6*q7 + 3600*p**4*q2**2*q3*q4**2*q5*q6 - 450*p**4*q2**2*q3*q4*q5**3*q6 - 450*p**4*q2**2*q3*q4*q5**2*q6**2 - 450*p**4*q2**2*q3*q4*q5**2*q6*q7 + 3600*p**4*q2**2*q3*q4*q5**2*q6 - 450*p**4*q2**2*q3*q4*q5*q6**3 - 450*p**4*q2**2*q3*q4*q5*q6**2*q7 + 3600*p**4*q2**2*q3*q4*q5*q6**2 - 450*p**4*q2**2*q3*q4*q5*q6*q7**2 + 3600*p**4*q2**2*q3*q4*q5*q6*q7 - 12600*p**4*q2**2*q3*q4*q5*q6 - 450*p**4*q2*q3**4*q4*q5*q6 - 450*p**4*q2*q3**3*q4**2*q5*q6 - 450*p**4*q2*q3**3*q4*q5**2*q6 - 450*p**4*q2*q3**3*q4*q5*q6**2 - 450*p**4*q2*q3**3*q4*q5*q6*q7 + 3600*p**4*q2*q3**3*q4*q5*q6 - 450*p**4*q2*q3**2*q4**3*q5*q6 - 450*p**4*q2*q3**2*q4**2*q5**2*q6 - 450*p**4*q2*q3**2*q4**2*q5*q6**2 - 450*p**4*q2*q3**2*q4**2*q5*q6*q7 + 3600*p**4*q2*q3**2*q4**2*q5*q6 - 450*p**4*q2*q3**2*q4*q5**3*q6 - 450*p**4*q2*q3**2*q4*q5**2*q6**2 - 450*p**4*q2*q3**2*q4*q5**2*q6*q7 + 3600*p**4*q2*q3**2*q4*q5**2*q6 - 450*p**4*q2*q3**2*q4*q5*q6**3 - 450*p**4*q2*q3**2*q4*q5*q6**2*q7 + 3600*p**4*q2*q3**2*q4*q5*q6**2 - 450*p**4*q2*q3**2*q4*q5*q6*q7**2 + 3600*p**4*q2*q3**2*q4*q5*q6*q7 - 12600*p**4*q2*q3**2*q4*q5*q6 - 450*p**4*q2*q3*q4**4*q5*q6 - 450*p**4*q2*q3*q4**3*q5**2*q6 - 450*p**4*q2*q3*q4**3*q5*q6**2 - 450*p**4*q2*q3*q4**3*q5*q6*q7 + 3600*p**4*q2*q3*q4**3*q5*q6 - 450*p**4*q2*q3*q4**2*q5**3*q6 - 450*p**4*q2*q3*q4**2*q5**2*q6**2 - 450*p**4*q2*q3*q4**2*q5**2*q6*q7 + 3600*p**4*q2*q3*q4**2*q5**2*q6 - 450*p**4*q2*q3*q4**2*q5*q6**3 - 450*p**4*q2*q3*q4**2*q5*q6**2*q7 + 3600*p**4*q2*q3*q4**2*q5*q6**2 - 450*p**4*q2*q3*q4**2*q5*q6*q7**2 + 3600*p**4*q2*q3*q4**2*q5*q6*q7 - 12600*p**4*q2*q3*q4**2*q5*q6 - 450*p**4*q2*q3*q4*q5**4*q6 - 450*p**4*q2*q3*q4*q5**3*q6**2 - 450*p**4*q2*q3*q4*q5**3*q6*q7 + 3600*p**4*q2*q3*q4*q5**3*q6 - 450*p**4*q2*q3*q4*q5**2*q6**3 - 450*p**4*q2*q3*q4*q5**2*q6**2*q7 + 3600*p**4*q2*q3*q4*q5**2*q6**2 - 450*p**4*q2*q3*q4*q5**2*q6*q7**2 + 3600*p**4*q2*q3*q4*q5**2*q6*q7 - 12600*p**4*q2*q3*q4*q5**2*q6 - 450*p**4*q2*q3*q4*q5*q6**4 - 450*p**4*q2*q3*q4*q5*q6**3*q7 + 3600*p**4*q2*q3*q4*q5*q6**3 - 450*p**4*q2*q3*q4*q5*q6**2*q7**2 + 3600*p**4*q2*q3*q4*q5*q6**2*q7 - 12600*p**4*q2*q3*q4*q5*q6**2 - 450*p**4*q2*q3*q4*q5*q6*q7**3 + 3600*p**4*q2*q3*q4*q5*q6*q7**2 - 12600*p**4*q2*q3*q4*q5*q6*q7 + 25200*p**4*q2*q3*q4*q5*q6 + 720*p**4*q3**4*q4*q5*q6 + 720*p**4*q3**3*q4**2*q5*q6 + 720*p**4*q3**3*q4*q5**2*q6 + 720*p**4*q3**3*q4*q5*q6**2 + 720*p**4*q3**3*q4*q5*q6*q7 - 5040*p**4*q3**3*q4*q5*q6 + 720*p**4*q3**2*q4**3*q5*q6 + 720*p**4*q3**2*q4**2*q5**2*q6 + 720*p**4*q3**2*q4**2*q5*q6**2 + 720*p**4*q3**2*q4**2*q5*q6*q7 - 5040*p**4*q3**2*q4**2*q5*q6 + 720*p**4*q3**2*q4*q5**3*q6 + 720*p**4*q3**2*q4*q5**2*q6**2 + 720*p**4*q3**2*q4*q5**2*q6*q7 - 5040*p**4*q3**2*q4*q5**2*q6 + 720*p**4*q3**2*q4*q5*q6**3 + 720*p**4*q3**2*q4*q5*q6**2*q7 - 5040*p**4*q3**2*q4*q5*q6**2 + 720*p**4*q3**2*q4*q5*q6*q7**2 - 5040*p**4*q3**2*q4*q5*q6*q7 + 15120*p**4*q3**2*q4*q5*q6 + 720*p**4*q3*q4**4*q5*q6 + 720*p**4*q3*q4**3*q5**2*q6 + 720*p**4*q3*q4**3*q5*q6**2 + 720*p**4*q3*q4**3*q5*q6*q7 - 5040*p**4*q3*q4**3*q5*q6 + 720*p**4*q3*q4**2*q5**3*q6 + 720*p**4*q3*q4**2*q5**2*q6**2 + 720*p**4*q3*q4**2*q5**2*q6*q7 - 5040*p**4*q3*q4**2*q5**2*q6 + 720*p**4*q3*q4**2*q5*q6**3 + 720*p**4*q3*q4**2*q5*q6**2*q7 - 5040*p**4*q3*q4**2*q5*q6**2 + 720*p**4*q3*q4**2*q5*q6*q7**2 - 5040*p**4*q3*q4**2*q5*q6*q7 + 15120*p**4*q3*q4**2*q5*q6 + 720*p**4*q3*q4*q5**4*q6 + 720*p**4*q3*q4*q5**3*q6**2 + 720*p**4*q3*q4*q5**3*q6*q7 - 5040*p**4*q3*q4*q5**3*q6 + 720*p**4*q3*q4*q5**2*q6**3 + 720*p**4*q3*q4*q5**2*q6**2*q7 - 5040*p**4*q3*q4*q5**2*q6**2 + 720*p**4*q3*q4*q5**2*q6*q7**2 - 5040*p**4*q3*q4*q5**2*q6*q7 + 15120*p**4*q3*q4*q5**2*q6 + 720*p**4*q3*q4*q5*q6**4 + 720*p**4*q3*q4*q5*q6**3*q7 - 5040*p**4*q3*q4*q5*q6**3 + 720*p**4*q3*q4*q5*q6**2*q7**2 - 5040*p**4*q3*q4*q5*q6**2*q7 + 15120*p**4*q3*q4*q5*q6**2 + 720*p**4*q3*q4*q5*q6*q7**3 - 5040*p**4*q3*q4*q5*q6*q7**2 + 15120*p**4*q3*q4*q5*q6*q7 - 25200*p**4*q3*q4*q5*q6 - 630*p**4*q4**4*q5*q6 - 630*p**4*q4**3*q5**2*q6 - 630*p**4*q4**3*q5*q6**2 - 630*p**4*q4**3*q5*q6*q7 + 3780*p**4*q4**3*q5*q6 - 630*p**4*q4**2*q5**3*q6 - 630*p**4*q4**2*q5**2*q6**2 - 630*p**4*q4**2*q5**2*q6*q7 + 3780*p**4*q4**2*q5**2*q6 - 630*p**4*q4**2*q5*q6**3 - 630*p**4*q4**2*q5*q6**2*q7 + 3780*p**4*q4**2*q5*q6**2 - 630*p**4*q4**2*q5*q6*q7**2 + 3780*p**4*q4**2*q5*q6*q7 - 9450*p**4*q4**2*q5*q6 - 630*p**4*q4*q5**4*q6 - 630*p**4*q4*q5**3*q6**2 - 630*p**4*q4*q5**3*q6*q7 + 3780*p**4*q4*q5**3*q6 - 630*p**4*q4*q5**2*q6**3 - 630*p**4*q4*q5**2*q6**2*q7 + 3780*p**4*q4*q5**2*q6**2 - 630*p**4*q4*q5**2*q6*q7**2 + 3780*p**4*q4*q5**2*q6*q7 - 9450*p**4*q4*q5**2*q6 - 630*p**4*q4*q5*q6**4 - 630*p**4*q4*q5*q6**3*q7 + 3780*p**4*q4*q5*q6**3 - 630*p**4*q4*q5*q6**2*q7**2 + 3780*p**4*q4*q5*q6**2*q7 - 9450*p**4*q4*q5*q6**2 - 630*p**4*q4*q5*q6*q7**3 + 3780*p**4*q4*q5*q6*q7**2 - 9450*p**4*q4*q5*q6*q7 + 12600*p**4*q4*q5*q6 + 252*p**4*q5**4*q6 + 252*p**4*q5**3*q6**2 + 252*p**4*q5**3*q6*q7 - 1260*p**4*q5**3*q6 + 252*p**4*q5**2*q6**3 + 252*p**4*q5**2*q6**2*q7 - 1260*p**4*q5**2*q6**2 + 252*p**4*q5**2*q6*q7**2 - 1260*p**4*q5**2*q6*q7 + 2520*p**4*q5**2*q6 + 252*p**4*q5*q6**4 + 252*p**4*q5*q6**3*q7 - 1260*p**4*q5*q6**3 + 252*p**4*q5*q6**2*q7**2 - 1260*p**4*q5*q6**2*q7 + 2520*p**4*q5*q6**2 + 252*p**4*q5*q6*q7**3 - 1260*p**4*q5*q6*q7**2 + 2520*p**4*q5*q6*q7 - 2520*p**4*q5*q6 - 200*p**3*q1**4*q2*q3*q4*q5*q6 - 200*p**3*q1**3*q2**2*q3*q4*q5*q6 - 200*p**3*q1**3*q2*q3**2*q4*q5*q6 - 200*p**3*q1**3*q2*q3*q4**2*q5*q6 - 200*p**3*q1**3*q2*q3*q4*q5**2*q6 - 200*p**3*q1**3*q2*q3*q4*q5*q6**2 - 200*p**3*q1**3*q2*q3*q4*q5*q6*q7 + 1800*p**3*q1**3*q2*q3*q4*q5*q6 - 200*p**3*q1**2*q2**3*q3*q4*q5*q6 - 200*p**3*q1**2*q2**2*q3**2*q4*q5*q6 - 200*p**3*q1**2*q2**2*q3*q4**2*q5*q6 - 200*p**3*q1**2*q2**2*q3*q4*q5**2*q6 - 200*p**3*q1**2*q2**2*q3*q4*q5*q6**2 - 200*p**3*q1**2*q2**2*q3*q4*q5*q6*q7 + 1800*p**3*q1**2*q2**2*q3*q4*q5*q6 - 200*p**3*q1**2*q2*q3**3*q4*q5*q6 - 200*p**3*q1**2*q2*q3**2*q4**2*q5*q6 - 200*p**3*q1**2*q2*q3**2*q4*q5**2*q6 - 200*p**3*q1**2*q2*q3**2*q4*q5*q6**2 - 200*p**3*q1**2*q2*q3**2*q4*q5*q6*q7 + 1800*p**3*q1**2*q2*q3**2*q4*q5*q6 - 200*p**3*q1**2*q2*q3*q4**3*q5*q6 - 200*p**3*q1**2*q2*q3*q4**2*q5**2*q6 - 200*p**3*q1**2*q2*q3*q4**2*q5*q6**2 - 200*p**3*q1**2*q2*q3*q4**2*q5*q6*q7 + 1800*p**3*q1**2*q2*q3*q4**2*q5*q6 - 200*p**3*q1**2*q2*q3*q4*q5**3*q6 - 200*p**3*q1**2*q2*q3*q4*q5**2*q6**2 - 200*p**3*q1**2*q2*q3*q4*q5**2*q6*q7 + 1800*p**3*q1**2*q2*q3*q4*q5**2*q6 - 200*p**3*q1**2*q2*q3*q4*q5*q6**3 - 200*p**3*q1**2*q2*q3*q4*q5*q6**2*q7 + 1800*p**3*q1**2*q2*q3*q4*q5*q6**2 - 200*p**3*q1**2*q2*q3*q4*q5*q6*q7**2 + 1800*p**3*q1**2*q2*q3*q4*q5*q6*q7 - 7200*p**3*q1**2*q2*q3*q4*q5*q6 - 200*p**3*q1*q2**4*q3*q4*q5*q6 - 200*p**3*q1*q2**3*q3**2*q4*q5*q6 - 200*p**3*q1*q2**3*q3*q4**2*q5*q6 - 200*p**3*q1*q2**3*q3*q4*q5**2*q6 - 200*p**3*q1*q2**3*q3*q4*q5*q6**2 - 200*p**3*q1*q2**3*q3*q4*q5*q6*q7 + 1800*p**3*q1*q2**3*q3*q4*q5*q6 - 200*p**3*q1*q2**2*q3**3*q4*q5*q6 - 200*p**3*q1*q2**2*q3**2*q4**2*q5*q6 - 200*p**3*q1*q2**2*q3**2*q4*q5**2*q6 - 200*p**3*q1*q2**2*q3**2*q4*q5*q6**2 - 200*p**3*q1*q2**2*q3**2*q4*q5*q6*q7 + 1800*p**3*q1*q2**2*q3**2*q4*q5*q6 - 200*p**3*q1*q2**2*q3*q4**3*q5*q6 - 200*p**3*q1*q2**2*q3*q4**2*q5**2*q6 - 200*p**3*q1*q2**2*q3*q4**2*q5*q6**2 - 200*p**3*q1*q2**2*q3*q4**2*q5*q6*q7 + 1800*p**3*q1*q2**2*q3*q4**2*q5*q6 - 200*p**3*q1*q2**2*q3*q4*q5**3*q6 - 200*p**3*q1*q2**2*q3*q4*q5**2*q6**2 - 200*p**3*q1*q2**2*q3*q4*q5**2*q6*q7 + 1800*p**3*q1*q2**2*q3*q4*q5**2*q6 - 200*p**3*q1*q2**2*q3*q4*q5*q6**3 - 200*p**3*q1*q2**2*q3*q4*q5*q6**2*q7 + 1800*p**3*q1*q2**2*q3*q4*q5*q6**2 - 200*p**3*q1*q2**2*q3*q4*q5*q6*q7**2 + 1800*p**3*q1*q2**2*q3*q4*q5*q6*q7 - 7200*p**3*q1*q2**2*q3*q4*q5*q6 - 200*p**3*q1*q2*q3**4*q4*q5*q6 - 200*p**3*q1*q2*q3**3*q4**2*q5*q6 - 200*p**3*q1*q2*q3**3*q4*q5**2*q6 - 200*p**3*q1*q2*q3**3*q4*q5*q6**2 - 200*p**3*q1*q2*q3**3*q4*q5*q6*q7 + 1800*p**3*q1*q2*q3**3*q4*q5*q6 - 200*p**3*q1*q2*q3**2*q4**3*q5*q6 - 200*p**3*q1*q2*q3**2*q4**2*q5**2*q6 - 200*p**3*q1*q2*q3**2*q4**2*q5*q6**2 - 200*p**3*q1*q2*q3**2*q4**2*q5*q6*q7 + 1800*p**3*q1*q2*q3**2*q4**2*q5*q6 - 200*p**3*q1*q2*q3**2*q4*q5**3*q6 - 200*p**3*q1*q2*q3**2*q4*q5**2*q6**2 - 200*p**3*q1*q2*q3**2*q4*q5**2*q6*q7 + 1800*p**3*q1*q2*q3**2*q4*q5**2*q6 - 200*p**3*q1*q2*q3**2*q4*q5*q6**3 - 200*p**3*q1*q2*q3**2*q4*q5*q6**2*q7 + 1800*p**3*q1*q2*q3**2*q4*q5*q6**2 - 200*p**3*q1*q2*q3**2*q4*q5*q6*q7**2 + 1800*p**3*q1*q2*q3**2*q4*q5*q6*q7 - 7200*p**3*q1*q2*q3**2*q4*q5*q6 - 200*p**3*q1*q2*q3*q4**4*q5*q6 - 200*p**3*q1*q2*q3*q4**3*q5**2*q6 - 200*p**3*q1*q2*q3*q4**3*q5*q6**2 - 200*p**3*q1*q2*q3*q4**3*q5*q6*q7 + 1800*p**3*q1*q2*q3*q4**3*q5*q6 - 200*p**3*q1*q2*q3*q4**2*q5**3*q6 - 200*p**3*q1*q2*q3*q4**2*q5**2*q6**2 - 200*p**3*q1*q2*q3*q4**2*q5**2*q6*q7 + 1800*p**3*q1*q2*q3*q4**2*q5**2*q6 - 200*p**3*q1*q2*q3*q4**2*q5*q6**3 - 200*p**3*q1*q2*q3*q4**2*q5*q6**2*q7 + 1800*p**3*q1*q2*q3*q4**2*q5*q6**2 - 200*p**3*q1*q2*q3*q4**2*q5*q6*q7**2 + 1800*p**3*q1*q2*q3*q4**2*q5*q6*q7 - 7200*p**3*q1*q2*q3*q4**2*q5*q6 - 200*p**3*q1*q2*q3*q4*q5**4*q6 - 200*p**3*q1*q2*q3*q4*q5**3*q6**2 - 200*p**3*q1*q2*q3*q4*q5**3*q6*q7 + 1800*p**3*q1*q2*q3*q4*q5**3*q6 - 200*p**3*q1*q2*q3*q4*q5**2*q6**3 - 200*p**3*q1*q2*q3*q4*q5**2*q6**2*q7 + 1800*p**3*q1*q2*q3*q4*q5**2*q6**2 - 200*p**3*q1*q2*q3*q4*q5**2*q6*q7**2 + 1800*p**3*q1*q2*q3*q4*q5**2*q6*q7 - 7200*p**3*q1*q2*q3*q4*q5**2*q6 - 200*p**3*q1*q2*q3*q4*q5*q6**4 - 200*p**3*q1*q2*q3*q4*q5*q6**3*q7 + 1800*p**3*q1*q2*q3*q4*q5*q6**3 - 200*p**3*q1*q2*q3*q4*q5*q6**2*q7**2 + 1800*p**3*q1*q2*q3*q4*q5*q6**2*q7 - 7200*p**3*q1*q2*q3*q4*q5*q6**2 - 200*p**3*q1*q2*q3*q4*q5*q6*q7**3 + 1800*p**3*q1*q2*q3*q4*q5*q6*q7**2 - 7200*p**3*q1*q2*q3*q4*q5*q6*q7 + 16800*p**3*q1*q2*q3*q4*q5*q6 + 450*p**3*q2**4*q3*q4*q5*q6 + 450*p**3*q2**3*q3**2*q4*q5*q6 + 450*p**3*q2**3*q3*q4**2*q5*q6 + 450*p**3*q2**3*q3*q4*q5**2*q6 + 450*p**3*q2**3*q3*q4*q5*q6**2 + 450*p**3*q2**3*q3*q4*q5*q6*q7 - 3600*p**3*q2**3*q3*q4*q5*q6 + 450*p**3*q2**2*q3**3*q4*q5*q6 + 450*p**3*q2**2*q3**2*q4**2*q5*q6 + 450*p**3*q2**2*q3**2*q4*q5**2*q6 + 450*p**3*q2**2*q3**2*q4*q5*q6**2 + 450*p**3*q2**2*q3**2*q4*q5*q6*q7 - 3600*p**3*q2**2*q3**2*q4*q5*q6 + 450*p**3*q2**2*q3*q4**3*q5*q6 + 450*p**3*q2**2*q3*q4**2*q5**2*q6 + 450*p**3*q2**2*q3*q4**2*q5*q6**2 + 450*p**3*q2**2*q3*q4**2*q5*q6*q7 - 3600*p**3*q2**2*q3*q4**2*q5*q6 + 450*p**3*q2**2*q3*q4*q5**3*q6 + 450*p**3*q2**2*q3*q4*q5**2*q6**2 + 450*p**3*q2**2*q3*q4*q5**2*q6*q7 - 3600*p**3*q2**2*q3*q4*q5**2*q6 + 450*p**3*q2**2*q3*q4*q5*q6**3 + 450*p**3*q2**2*q3*q4*q5*q6**2*q7 - 3600*p**3*q2**2*q3*q4*q5*q6**2 + 450*p**3*q2**2*q3*q4*q5*q6*q7**2 - 3600*p**3*q2**2*q3*q4*q5*q6*q7 + 12600*p**3*q2**2*q3*q4*q5*q6 + 450*p**3*q2*q3**4*q4*q5*q6 + 450*p**3*q2*q3**3*q4**2*q5*q6 + 450*p**3*q2*q3**3*q4*q5**2*q6 + 450*p**3*q2*q3**3*q4*q5*q6**2 + 450*p**3*q2*q3**3*q4*q5*q6*q7 - 3600*p**3*q2*q3**3*q4*q5*q6 + 450*p**3*q2*q3**2*q4**3*q5*q6 + 450*p**3*q2*q3**2*q4**2*q5**2*q6 + 450*p**3*q2*q3**2*q4**2*q5*q6**2 + 450*p**3*q2*q3**2*q4**2*q5*q6*q7 - 3600*p**3*q2*q3**2*q4**2*q5*q6 + 450*p**3*q2*q3**2*q4*q5**3*q6 + 450*p**3*q2*q3**2*q4*q5**2*q6**2 + 450*p**3*q2*q3**2*q4*q5**2*q6*q7 - 3600*p**3*q2*q3**2*q4*q5**2*q6 + 450*p**3*q2*q3**2*q4*q5*q6**3 + 450*p**3*q2*q3**2*q4*q5*q6**2*q7 - 3600*p**3*q2*q3**2*q4*q5*q6**2 + 450*p**3*q2*q3**2*q4*q5*q6*q7**2 - 3600*p**3*q2*q3**2*q4*q5*q6*q7 + 12600*p**3*q2*q3**2*q4*q5*q6 + 450*p**3*q2*q3*q4**4*q5*q6 + 450*p**3*q2*q3*q4**3*q5**2*q6 + 450*p**3*q2*q3*q4**3*q5*q6**2 + 450*p**3*q2*q3*q4**3*q5*q6*q7 - 3600*p**3*q2*q3*q4**3*q5*q6 + 450*p**3*q2*q3*q4**2*q5**3*q6 + 450*p**3*q2*q3*q4**2*q5**2*q6**2 + 450*p**3*q2*q3*q4**2*q5**2*q6*q7 - 3600*p**3*q2*q3*q4**2*q5**2*q6 + 450*p**3*q2*q3*q4**2*q5*q6**3 + 450*p**3*q2*q3*q4**2*q5*q6**2*q7 - 3600*p**3*q2*q3*q4**2*q5*q6**2 + 450*p**3*q2*q3*q4**2*q5*q6*q7**2 - 3600*p**3*q2*q3*q4**2*q5*q6*q7 + 12600*p**3*q2*q3*q4**2*q5*q6 + 450*p**3*q2*q3*q4*q5**4*q6 + 450*p**3*q2*q3*q4*q5**3*q6**2 + 450*p**3*q2*q3*q4*q5**3*q6*q7 - 3600*p**3*q2*q3*q4*q5**3*q6 + 450*p**3*q2*q3*q4*q5**2*q6**3 + 450*p**3*q2*q3*q4*q5**2*q6**2*q7 - 3600*p**3*q2*q3*q4*q5**2*q6**2 + 450*p**3*q2*q3*q4*q5**2*q6*q7**2 - 3600*p**3*q2*q3*q4*q5**2*q6*q7 + 12600*p**3*q2*q3*q4*q5**2*q6 + 450*p**3*q2*q3*q4*q5*q6**4 + 450*p**3*q2*q3*q4*q5*q6**3*q7 - 3600*p**3*q2*q3*q4*q5*q6**3 + 450*p**3*q2*q3*q4*q5*q6**2*q7**2 - 3600*p**3*q2*q3*q4*q5*q6**2*q7 + 12600*p**3*q2*q3*q4*q5*q6**2 + 450*p**3*q2*q3*q4*q5*q6*q7**3 - 3600*p**3*q2*q3*q4*q5*q6*q7**2 + 12600*p**3*q2*q3*q4*q5*q6*q7 - 25200*p**3*q2*q3*q4*q5*q6 - 480*p**3*q3**4*q4*q5*q6 - 480*p**3*q3**3*q4**2*q5*q6 - 480*p**3*q3**3*q4*q5**2*q6 - 480*p**3*q3**3*q4*q5*q6**2 - 480*p**3*q3**3*q4*q5*q6*q7 + 3360*p**3*q3**3*q4*q5*q6 - 480*p**3*q3**2*q4**3*q5*q6 - 480*p**3*q3**2*q4**2*q5**2*q6 - 480*p**3*q3**2*q4**2*q5*q6**2 - 480*p**3*q3**2*q4**2*q5*q6*q7 + 3360*p**3*q3**2*q4**2*q5*q6 - 480*p**3*q3**2*q4*q5**3*q6 - 480*p**3*q3**2*q4*q5**2*q6**2 - 480*p**3*q3**2*q4*q5**2*q6*q7 + 3360*p**3*q3**2*q4*q5**2*q6 - 480*p**3*q3**2*q4*q5*q6**3 - 480*p**3*q3**2*q4*q5*q6**2*q7 + 3360*p**3*q3**2*q4*q5*q6**2 - 480*p**3*q3**2*q4*q5*q6*q7**2 + 3360*p**3*q3**2*q4*q5*q6*q7 - 10080*p**3*q3**2*q4*q5*q6 - 480*p**3*q3*q4**4*q5*q6 - 480*p**3*q3*q4**3*q5**2*q6 - 480*p**3*q3*q4**3*q5*q6**2 - 480*p**3*q3*q4**3*q5*q6*q7 + 3360*p**3*q3*q4**3*q5*q6 - 480*p**3*q3*q4**2*q5**3*q6 - 480*p**3*q3*q4**2*q5**2*q6**2 - 480*p**3*q3*q4**2*q5**2*q6*q7 + 3360*p**3*q3*q4**2*q5**2*q6 - 480*p**3*q3*q4**2*q5*q6**3 - 480*p**3*q3*q4**2*q5*q6**2*q7 + 3360*p**3*q3*q4**2*q5*q6**2 - 480*p**3*q3*q4**2*q5*q6*q7**2 + 3360*p**3*q3*q4**2*q5*q6*q7 - 10080*p**3*q3*q4**2*q5*q6 - 480*p**3*q3*q4*q5**4*q6 - 480*p**3*q3*q4*q5**3*q6**2 - 480*p**3*q3*q4*q5**3*q6*q7 + 3360*p**3*q3*q4*q5**3*q6 - 480*p**3*q3*q4*q5**2*q6**3 - 480*p**3*q3*q4*q5**2*q6**2*q7 + 3360*p**3*q3*q4*q5**2*q6**2 - 480*p**3*q3*q4*q5**2*q6*q7**2 + 3360*p**3*q3*q4*q5**2*q6*q7 - 10080*p**3*q3*q4*q5**2*q6 - 480*p**3*q3*q4*q5*q6**4 - 480*p**3*q3*q4*q5*q6**3*q7 + 3360*p**3*q3*q4*q5*q6**3 - 480*p**3*q3*q4*q5*q6**2*q7**2 + 3360*p**3*q3*q4*q5*q6**2*q7 - 10080*p**3*q3*q4*q5*q6**2 - 480*p**3*q3*q4*q5*q6*q7**3 + 3360*p**3*q3*q4*q5*q6*q7**2 - 10080*p**3*q3*q4*q5*q6*q7 + 16800*p**3*q3*q4*q5*q6 + 210*p**3*q4**4*q5*q6 + 210*p**3*q4**3*q5**2*q6 + 210*p**3*q4**3*q5*q6**2 + 210*p**3*q4**3*q5*q6*q7 - 1260*p**3*q4**3*q5*q6 + 210*p**3*q4**2*q5**3*q6 + 210*p**3*q4**2*q5**2*q6**2 + 210*p**3*q4**2*q5**2*q6*q7 - 1260*p**3*q4**2*q5**2*q6 + 210*p**3*q4**2*q5*q6**3 + 210*p**3*q4**2*q5*q6**2*q7 - 1260*p**3*q4**2*q5*q6**2 + 210*p**3*q4**2*q5*q6*q7**2 - 1260*p**3*q4**2*q5*q6*q7 + 3150*p**3*q4**2*q5*q6 + 210*p**3*q4*q5**4*q6 + 210*p**3*q4*q5**3*q6**2 + 210*p**3*q4*q5**3*q6*q7 - 1260*p**3*q4*q5**3*q6 + 210*p**3*q4*q5**2*q6**3 + 210*p**3*q4*q5**2*q6**2*q7 - 1260*p**3*q4*q5**2*q6**2 + 210*p**3*q4*q5**2*q6*q7**2 - 1260*p**3*q4*q5**2*q6*q7 + 3150*p**3*q4*q5**2*q6 + 210*p**3*q4*q5*q6**4 + 210*p**3*q4*q5*q6**3*q7 - 1260*p**3*q4*q5*q6**3 + 210*p**3*q4*q5*q6**2*q7**2 - 1260*p**3*q4*q5*q6**2*q7 + 3150*p**3*q4*q5*q6**2 + 210*p**3*q4*q5*q6*q7**3 - 1260*p**3*q4*q5*q6*q7**2 + 3150*p**3*q4*q5*q6*q7 - 4200*p**3*q4*q5*q6 + 150*p**2*q1**4*q2*q3*q4*q5*q6 + 150*p**2*q1**3*q2**2*q3*q4*q5*q6 + 150*p**2*q1**3*q2*q3**2*q4*q5*q6 + 150*p**2*q1**3*q2*q3*q4**2*q5*q6 + 150*p**2*q1**3*q2*q3*q4*q5**2*q6 + 150*p**2*q1**3*q2*q3*q4*q5*q6**2 + 150*p**2*q1**3*q2*q3*q4*q5*q6*q7 - 1350*p**2*q1**3*q2*q3*q4*q5*q6 + 150*p**2*q1**2*q2**3*q3*q4*q5*q6 + 150*p**2*q1**2*q2**2*q3**2*q4*q5*q6 + 150*p**2*q1**2*q2**2*q3*q4**2*q5*q6 + 150*p**2*q1**2*q2**2*q3*q4*q5**2*q6 + 150*p**2*q1**2*q2**2*q3*q4*q5*q6**2 + 150*p**2*q1**2*q2**2*q3*q4*q5*q6*q7 - 1350*p**2*q1**2*q2**2*q3*q4*q5*q6 + 150*p**2*q1**2*q2*q3**3*q4*q5*q6 + 150*p**2*q1**2*q2*q3**2*q4**2*q5*q6 + 150*p**2*q1**2*q2*q3**2*q4*q5**2*q6 + 150*p**2*q1**2*q2*q3**2*q4*q5*q6**2 + 150*p**2*q1**2*q2*q3**2*q4*q5*q6*q7 - 1350*p**2*q1**2*q2*q3**2*q4*q5*q6 + 150*p**2*q1**2*q2*q3*q4**3*q5*q6 + 150*p**2*q1**2*q2*q3*q4**2*q5**2*q6 + 150*p**2*q1**2*q2*q3*q4**2*q5*q6**2 + 150*p**2*q1**2*q2*q3*q4**2*q5*q6*q7 - 1350*p**2*q1**2*q2*q3*q4**2*q5*q6 + 150*p**2*q1**2*q2*q3*q4*q5**3*q6 + 150*p**2*q1**2*q2*q3*q4*q5**2*q6**2 + 150*p**2*q1**2*q2*q3*q4*q5**2*q6*q7 - 1350*p**2*q1**2*q2*q3*q4*q5**2*q6 + 150*p**2*q1**2*q2*q3*q4*q5*q6**3 + 150*p**2*q1**2*q2*q3*q4*q5*q6**2*q7 - 1350*p**2*q1**2*q2*q3*q4*q5*q6**2 + 150*p**2*q1**2*q2*q3*q4*q5*q6*q7**2 - 1350*p**2*q1**2*q2*q3*q4*q5*q6*q7 + 5400*p**2*q1**2*q2*q3*q4*q5*q6 + 150*p**2*q1*q2**4*q3*q4*q5*q6 + 150*p**2*q1*q2**3*q3**2*q4*q5*q6 + 150*p**2*q1*q2**3*q3*q4**2*q5*q6 + 150*p**2*q1*q2**3*q3*q4*q5**2*q6 + 150*p**2*q1*q2**3*q3*q4*q5*q6**2 + 150*p**2*q1*q2**3*q3*q4*q5*q6*q7 - 1350*p**2*q1*q2**3*q3*q4*q5*q6 + 150*p**2*q1*q2**2*q3**3*q4*q5*q6 + 150*p**2*q1*q2**2*q3**2*q4**2*q5*q6 + 150*p**2*q1*q2**2*q3**2*q4*q5**2*q6 + 150*p**2*q1*q2**2*q3**2*q4*q5*q6**2 + 150*p**2*q1*q2**2*q3**2*q4*q5*q6*q7 - 1350*p**2*q1*q2**2*q3**2*q4*q5*q6 + 150*p**2*q1*q2**2*q3*q4**3*q5*q6 + 150*p**2*q1*q2**2*q3*q4**2*q5**2*q6 + 150*p**2*q1*q2**2*q3*q4**2*q5*q6**2 + 150*p**2*q1*q2**2*q3*q4**2*q5*q6*q7 - 1350*p**2*q1*q2**2*q3*q4**2*q5*q6 + 150*p**2*q1*q2**2*q3*q4*q5**3*q6 + 150*p**2*q1*q2**2*q3*q4*q5**2*q6**2 + 150*p**2*q1*q2**2*q3*q4*q5**2*q6*q7 - 1350*p**2*q1*q2**2*q3*q4*q5**2*q6 + 150*p**2*q1*q2**2*q3*q4*q5*q6**3 + 150*p**2*q1*q2**2*q3*q4*q5*q6**2*q7 - 1350*p**2*q1*q2**2*q3*q4*q5*q6**2 + 150*p**2*q1*q2**2*q3*q4*q5*q6*q7**2 - 1350*p**2*q1*q2**2*q3*q4*q5*q6*q7 + 5400*p**2*q1*q2**2*q3*q4*q5*q6 + 150*p**2*q1*q2*q3**4*q4*q5*q6 + 150*p**2*q1*q2*q3**3*q4**2*q5*q6 + 150*p**2*q1*q2*q3**3*q4*q5**2*q6 + 150*p**2*q1*q2*q3**3*q4*q5*q6**2 + 150*p**2*q1*q2*q3**3*q4*q5*q6*q7 - 1350*p**2*q1*q2*q3**3*q4*q5*q6 + 150*p**2*q1*q2*q3**2*q4**3*q5*q6 + 150*p**2*q1*q2*q3**2*q4**2*q5**2*q6 + 150*p**2*q1*q2*q3**2*q4**2*q5*q6**2 + 150*p**2*q1*q2*q3**2*q4**2*q5*q6*q7 - 1350*p**2*q1*q2*q3**2*q4**2*q5*q6 + 150*p**2*q1*q2*q3**2*q4*q5**3*q6 + 150*p**2*q1*q2*q3**2*q4*q5**2*q6**2 + 150*p**2*q1*q2*q3**2*q4*q5**2*q6*q7 - 1350*p**2*q1*q2*q3**2*q4*q5**2*q6 + 150*p**2*q1*q2*q3**2*q4*q5*q6**3 + 150*p**2*q1*q2*q3**2*q4*q5*q6**2*q7 - 1350*p**2*q1*q2*q3**2*q4*q5*q6**2 + 150*p**2*q1*q2*q3**2*q4*q5*q6*q7**2 - 1350*p**2*q1*q2*q3**2*q4*q5*q6*q7 + 5400*p**2*q1*q2*q3**2*q4*q5*q6 + 150*p**2*q1*q2*q3*q4**4*q5*q6 + 150*p**2*q1*q2*q3*q4**3*q5**2*q6 + 150*p**2*q1*q2*q3*q4**3*q5*q6**2 + 150*p**2*q1*q2*q3*q4**3*q5*q6*q7 - 1350*p**2*q1*q2*q3*q4**3*q5*q6 + 150*p**2*q1*q2*q3*q4**2*q5**3*q6 + 150*p**2*q1*q2*q3*q4**2*q5**2*q6**2 + 150*p**2*q1*q2*q3*q4**2*q5**2*q6*q7 - 1350*p**2*q1*q2*q3*q4**2*q5**2*q6 + 150*p**2*q1*q2*q3*q4**2*q5*q6**3 + 150*p**2*q1*q2*q3*q4**2*q5*q6**2*q7 - 1350*p**2*q1*q2*q3*q4**2*q5*q6**2 + 150*p**2*q1*q2*q3*q4**2*q5*q6*q7**2 - 1350*p**2*q1*q2*q3*q4**2*q5*q6*q7 + 5400*p**2*q1*q2*q3*q4**2*q5*q6 + 150*p**2*q1*q2*q3*q4*q5**4*q6 + 150*p**2*q1*q2*q3*q4*q5**3*q6**2 + 150*p**2*q1*q2*q3*q4*q5**3*q6*q7 - 1350*p**2*q1*q2*q3*q4*q5**3*q6 + 150*p**2*q1*q2*q3*q4*q5**2*q6**3 + 150*p**2*q1*q2*q3*q4*q5**2*q6**2*q7 - 1350*p**2*q1*q2*q3*q4*q5**2*q6**2 + 150*p**2*q1*q2*q3*q4*q5**2*q6*q7**2 - 1350*p**2*q1*q2*q3*q4*q5**2*q6*q7 + 5400*p**2*q1*q2*q3*q4*q5**2*q6 + 150*p**2*q1*q2*q3*q4*q5*q6**4 + 150*p**2*q1*q2*q3*q4*q5*q6**3*q7 - 1350*p**2*q1*q2*q3*q4*q5*q6**3 + 150*p**2*q1*q2*q3*q4*q5*q6**2*q7**2 - 1350*p**2*q1*q2*q3*q4*q5*q6**2*q7 + 5400*p**2*q1*q2*q3*q4*q5*q6**2 + 150*p**2*q1*q2*q3*q4*q5*q6*q7**3 - 1350*p**2*q1*q2*q3*q4*q5*q6*q7**2 + 5400*p**2*q1*q2*q3*q4*q5*q6*q7 - 12600*p**2*q1*q2*q3*q4*q5*q6 - 225*p**2*q2**4*q3*q4*q5*q6 - 225*p**2*q2**3*q3**2*q4*q5*q6 - 225*p**2*q2**3*q3*q4**2*q5*q6 - 225*p**2*q2**3*q3*q4*q5**2*q6 - 225*p**2*q2**3*q3*q4*q5*q6**2 - 225*p**2*q2**3*q3*q4*q5*q6*q7 + 1800*p**2*q2**3*q3*q4*q5*q6 - 225*p**2*q2**2*q3**3*q4*q5*q6 - 225*p**2*q2**2*q3**2*q4**2*q5*q6 - 225*p**2*q2**2*q3**2*q4*q5**2*q6 - 225*p**2*q2**2*q3**2*q4*q5*q6**2 - 225*p**2*q2**2*q3**2*q4*q5*q6*q7 + 1800*p**2*q2**2*q3**2*q4*q5*q6 - 225*p**2*q2**2*q3*q4**3*q5*q6 - 225*p**2*q2**2*q3*q4**2*q5**2*q6 - 225*p**2*q2**2*q3*q4**2*q5*q6**2 - 225*p**2*q2**2*q3*q4**2*q5*q6*q7 + 1800*p**2*q2**2*q3*q4**2*q5*q6 - 225*p**2*q2**2*q3*q4*q5**3*q6 - 225*p**2*q2**2*q3*q4*q5**2*q6**2 - 225*p**2*q2**2*q3*q4*q5**2*q6*q7 + 1800*p**2*q2**2*q3*q4*q5**2*q6 - 225*p**2*q2**2*q3*q4*q5*q6**3 - 225*p**2*q2**2*q3*q4*q5*q6**2*q7 + 1800*p**2*q2**2*q3*q4*q5*q6**2 - 225*p**2*q2**2*q3*q4*q5*q6*q7**2 + 1800*p**2*q2**2*q3*q4*q5*q6*q7 - 6300*p**2*q2**2*q3*q4*q5*q6 - 225*p**2*q2*q3**4*q4*q5*q6 - 225*p**2*q2*q3**3*q4**2*q5*q6 - 225*p**2*q2*q3**3*q4*q5**2*q6 - 225*p**2*q2*q3**3*q4*q5*q6**2 - 225*p**2*q2*q3**3*q4*q5*q6*q7 + 1800*p**2*q2*q3**3*q4*q5*q6 - 225*p**2*q2*q3**2*q4**3*q5*q6 - 225*p**2*q2*q3**2*q4**2*q5**2*q6 - 225*p**2*q2*q3**2*q4**2*q5*q6**2 - 225*p**2*q2*q3**2*q4**2*q5*q6*q7 + 1800*p**2*q2*q3**2*q4**2*q5*q6 - 225*p**2*q2*q3**2*q4*q5**3*q6 - 225*p**2*q2*q3**2*q4*q5**2*q6**2 - 225*p**2*q2*q3**2*q4*q5**2*q6*q7 + 1800*p**2*q2*q3**2*q4*q5**2*q6 - 225*p**2*q2*q3**2*q4*q5*q6**3 - 225*p**2*q2*q3**2*q4*q5*q6**2*q7 + 1800*p**2*q2*q3**2*q4*q5*q6**2 - 225*p**2*q2*q3**2*q4*q5*q6*q7**2 + 1800*p**2*q2*q3**2*q4*q5*q6*q7 - 6300*p**2*q2*q3**2*q4*q5*q6 - 225*p**2*q2*q3*q4**4*q5*q6 - 225*p**2*q2*q3*q4**3*q5**2*q6 - 225*p**2*q2*q3*q4**3*q5*q6**2 - 225*p**2*q2*q3*q4**3*q5*q6*q7 + 1800*p**2*q2*q3*q4**3*q5*q6 - 225*p**2*q2*q3*q4**2*q5**3*q6 - 225*p**2*q2*q3*q4**2*q5**2*q6**2 - 225*p**2*q2*q3*q4**2*q5**2*q6*q7 + 1800*p**2*q2*q3*q4**2*q5**2*q6 - 225*p**2*q2*q3*q4**2*q5*q6**3 - 225*p**2*q2*q3*q4**2*q5*q6**2*q7 + 1800*p**2*q2*q3*q4**2*q5*q6**2 - 225*p**2*q2*q3*q4**2*q5*q6*q7**2 + 1800*p**2*q2*q3*q4**2*q5*q6*q7 - 6300*p**2*q2*q3*q4**2*q5*q6 - 225*p**2*q2*q3*q4*q5**4*q6 - 225*p**2*q2*q3*q4*q5**3*q6**2 - 225*p**2*q2*q3*q4*q5**3*q6*q7 + 1800*p**2*q2*q3*q4*q5**3*q6 - 225*p**2*q2*q3*q4*q5**2*q6**3 - 225*p**2*q2*q3*q4*q5**2*q6**2*q7 + 1800*p**2*q2*q3*q4*q5**2*q6**2 - 225*p**2*q2*q3*q4*q5**2*q6*q7**2 + 1800*p**2*q2*q3*q4*q5**2*q6*q7 - 6300*p**2*q2*q3*q4*q5**2*q6 - 225*p**2*q2*q3*q4*q5*q6**4 - 225*p**2*q2*q3*q4*q5*q6**3*q7 + 1800*p**2*q2*q3*q4*q5*q6**3 - 225*p**2*q2*q3*q4*q5*q6**2*q7**2 + 1800*p**2*q2*q3*q4*q5*q6**2*q7 - 6300*p**2*q2*q3*q4*q5*q6**2 - 225*p**2*q2*q3*q4*q5*q6*q7**3 + 1800*p**2*q2*q3*q4*q5*q6*q7**2 - 6300*p**2*q2*q3*q4*q5*q6*q7 + 12600*p**2*q2*q3*q4*q5*q6 + 120*p**2*q3**4*q4*q5*q6 + 120*p**2*q3**3*q4**2*q5*q6 + 120*p**2*q3**3*q4*q5**2*q6 + 120*p**2*q3**3*q4*q5*q6**2 + 120*p**2*q3**3*q4*q5*q6*q7 - 840*p**2*q3**3*q4*q5*q6 + 120*p**2*q3**2*q4**3*q5*q6 + 120*p**2*q3**2*q4**2*q5**2*q6 + 120*p**2*q3**2*q4**2*q5*q6**2 + 120*p**2*q3**2*q4**2*q5*q6*q7 - 840*p**2*q3**2*q4**2*q5*q6 + 120*p**2*q3**2*q4*q5**3*q6 + 120*p**2*q3**2*q4*q5**2*q6**2 + 120*p**2*q3**2*q4*q5**2*q6*q7 - 840*p**2*q3**2*q4*q5**2*q6 + 120*p**2*q3**2*q4*q5*q6**3 + 120*p**2*q3**2*q4*q5*q6**2*q7 - 840*p**2*q3**2*q4*q5*q6**2 + 120*p**2*q3**2*q4*q5*q6*q7**2 - 840*p**2*q3**2*q4*q5*q6*q7 + 2520*p**2*q3**2*q4*q5*q6 + 120*p**2*q3*q4**4*q5*q6 + 120*p**2*q3*q4**3*q5**2*q6 + 120*p**2*q3*q4**3*q5*q6**2 + 120*p**2*q3*q4**3*q5*q6*q7 - 840*p**2*q3*q4**3*q5*q6 + 120*p**2*q3*q4**2*q5**3*q6 + 120*p**2*q3*q4**2*q5**2*q6**2 + 120*p**2*q3*q4**2*q5**2*q6*q7 - 840*p**2*q3*q4**2*q5**2*q6 + 120*p**2*q3*q4**2*q5*q6**3 + 120*p**2*q3*q4**2*q5*q6**2*q7 - 840*p**2*q3*q4**2*q5*q6**2 + 120*p**2*q3*q4**2*q5*q6*q7**2 - 840*p**2*q3*q4**2*q5*q6*q7 + 2520*p**2*q3*q4**2*q5*q6 + 120*p**2*q3*q4*q5**4*q6 + 120*p**2*q3*q4*q5**3*q6**2 + 120*p**2*q3*q4*q5**3*q6*q7 - 840*p**2*q3*q4*q5**3*q6 + 120*p**2*q3*q4*q5**2*q6**3 + 120*p**2*q3*q4*q5**2*q6**2*q7 - 840*p**2*q3*q4*q5**2*q6**2 + 120*p**2*q3*q4*q5**2*q6*q7**2 - 840*p**2*q3*q4*q5**2*q6*q7 + 2520*p**2*q3*q4*q5**2*q6 + 120*p**2*q3*q4*q5*q6**4 + 120*p**2*q3*q4*q5*q6**3*q7 - 840*p**2*q3*q4*q5*q6**3 + 120*p**2*q3*q4*q5*q6**2*q7**2 - 840*p**2*q3*q4*q5*q6**2*q7 + 2520*p**2*q3*q4*q5*q6**2 + 120*p**2*q3*q4*q5*q6*q7**3 - 840*p**2*q3*q4*q5*q6*q7**2 + 2520*p**2*q3*q4*q5*q6*q7 - 4200*p**2*q3*q4*q5*q6 - 60*p*q1**4*q2*q3*q4*q5*q6 - 60*p*q1**3*q2**2*q3*q4*q5*q6 - 60*p*q1**3*q2*q3**2*q4*q5*q6 - 60*p*q1**3*q2*q3*q4**2*q5*q6 - 60*p*q1**3*q2*q3*q4*q5**2*q6 - 60*p*q1**3*q2*q3*q4*q5*q6**2 - 60*p*q1**3*q2*q3*q4*q5*q6*q7 + 540*p*q1**3*q2*q3*q4*q5*q6 - 60*p*q1**2*q2**3*q3*q4*q5*q6 - 60*p*q1**2*q2**2*q3**2*q4*q5*q6 - 60*p*q1**2*q2**2*q3*q4**2*q5*q6 - 60*p*q1**2*q2**2*q3*q4*q5**2*q6 - 60*p*q1**2*q2**2*q3*q4*q5*q6**2 - 60*p*q1**2*q2**2*q3*q4*q5*q6*q7 + 540*p*q1**2*q2**2*q3*q4*q5*q6 - 60*p*q1**2*q2*q3**3*q4*q5*q6 - 60*p*q1**2*q2*q3**2*q4**2*q5*q6 - 60*p*q1**2*q2*q3**2*q4*q5**2*q6 - 60*p*q1**2*q2*q3**2*q4*q5*q6**2 - 60*p*q1**2*q2*q3**2*q4*q5*q6*q7 + 540*p*q1**2*q2*q3**2*q4*q5*q6 - 60*p*q1**2*q2*q3*q4**3*q5*q6 - 60*p*q1**2*q2*q3*q4**2*q5**2*q6 - 60*p*q1**2*q2*q3*q4**2*q5*q6**2 - 60*p*q1**2*q2*q3*q4**2*q5*q6*q7 + 540*p*q1**2*q2*q3*q4**2*q5*q6 - 60*p*q1**2*q2*q3*q4*q5**3*q6 - 60*p*q1**2*q2*q3*q4*q5**2*q6**2 - 60*p*q1**2*q2*q3*q4*q5**2*q6*q7 + 540*p*q1**2*q2*q3*q4*q5**2*q6 - 60*p*q1**2*q2*q3*q4*q5*q6**3 - 60*p*q1**2*q2*q3*q4*q5*q6**2*q7 + 540*p*q1**2*q2*q3*q4*q5*q6**2 - 60*p*q1**2*q2*q3*q4*q5*q6*q7**2 + 540*p*q1**2*q2*q3*q4*q5*q6*q7 - 2160*p*q1**2*q2*q3*q4*q5*q6 - 60*p*q1*q2**4*q3*q4*q5*q6 - 60*p*q1*q2**3*q3**2*q4*q5*q6 - 60*p*q1*q2**3*q3*q4**2*q5*q6 - 60*p*q1*q2**3*q3*q4*q5**2*q6 - 60*p*q1*q2**3*q3*q4*q5*q6**2 - 60*p*q1*q2**3*q3*q4*q5*q6*q7 + 540*p*q1*q2**3*q3*q4*q5*q6 - 60*p*q1*q2**2*q3**3*q4*q5*q6 - 60*p*q1*q2**2*q3**2*q4**2*q5*q6 - 60*p*q1*q2**2*q3**2*q4*q5**2*q6 - 60*p*q1*q2**2*q3**2*q4*q5*q6**2 - 60*p*q1*q2**2*q3**2*q4*q5*q6*q7 + 540*p*q1*q2**2*q3**2*q4*q5*q6 - 60*p*q1*q2**2*q3*q4**3*q5*q6 - 60*p*q1*q2**2*q3*q4**2*q5**2*q6 - 60*p*q1*q2**2*q3*q4**2*q5*q6**2 - 60*p*q1*q2**2*q3*q4**2*q5*q6*q7 + 540*p*q1*q2**2*q3*q4**2*q5*q6 - 60*p*q1*q2**2*q3*q4*q5**3*q6 - 60*p*q1*q2**2*q3*q4*q5**2*q6**2 - 60*p*q1*q2**2*q3*q4*q5**2*q6*q7 + 540*p*q1*q2**2*q3*q4*q5**2*q6 - 60*p*q1*q2**2*q3*q4*q5*q6**3 - 60*p*q1*q2**2*q3*q4*q5*q6**2*q7 + 540*p*q1*q2**2*q3*q4*q5*q6**2 - 60*p*q1*q2**2*q3*q4*q5*q6*q7**2 + 540*p*q1*q2**2*q3*q4*q5*q6*q7 - 2160*p*q1*q2**2*q3*q4*q5*q6 - 60*p*q1*q2*q3**4*q4*q5*q6 - 60*p*q1*q2*q3**3*q4**2*q5*q6 - 60*p*q1*q2*q3**3*q4*q5**2*q6 - 60*p*q1*q2*q3**3*q4*q5*q6**2 - 60*p*q1*q2*q3**3*q4*q5*q6*q7 + 540*p*q1*q2*q3**3*q4*q5*q6 - 60*p*q1*q2*q3**2*q4**3*q5*q6 - 60*p*q1*q2*q3**2*q4**2*q5**2*q6 - 60*p*q1*q2*q3**2*q4**2*q5*q6**2 - 60*p*q1*q2*q3**2*q4**2*q5*q6*q7 + 540*p*q1*q2*q3**2*q4**2*q5*q6 - 60*p*q1*q2*q3**2*q4*q5**3*q6 - 60*p*q1*q2*q3**2*q4*q5**2*q6**2 - 60*p*q1*q2*q3**2*q4*q5**2*q6*q7 + 540*p*q1*q2*q3**2*q4*q5**2*q6 - 60*p*q1*q2*q3**2*q4*q5*q6**3 - 60*p*q1*q2*q3**2*q4*q5*q6**2*q7 + 540*p*q1*q2*q3**2*q4*q5*q6**2 - 60*p*q1*q2*q3**2*q4*q5*q6*q7**2 + 540*p*q1*q2*q3**2*q4*q5*q6*q7 - 2160*p*q1*q2*q3**2*q4*q5*q6 - 60*p*q1*q2*q3*q4**4*q5*q6 - 60*p*q1*q2*q3*q4**3*q5**2*q6 - 60*p*q1*q2*q3*q4**3*q5*q6**2 - 60*p*q1*q2*q3*q4**3*q5*q6*q7 + 540*p*q1*q2*q3*q4**3*q5*q6 - 60*p*q1*q2*q3*q4**2*q5**3*q6 - 60*p*q1*q2*q3*q4**2*q5**2*q6**2 - 60*p*q1*q2*q3*q4**2*q5**2*q6*q7 + 540*p*q1*q2*q3*q4**2*q5**2*q6 - 60*p*q1*q2*q3*q4**2*q5*q6**3 - 60*p*q1*q2*q3*q4**2*q5*q6**2*q7 + 540*p*q1*q2*q3*q4**2*q5*q6**2 - 60*p*q1*q2*q3*q4**2*q5*q6*q7**2 + 540*p*q1*q2*q3*q4**2*q5*q6*q7 - 2160*p*q1*q2*q3*q4**2*q5*q6 - 60*p*q1*q2*q3*q4*q5**4*q6 - 60*p*q1*q2*q3*q4*q5**3*q6**2 - 60*p*q1*q2*q3*q4*q5**3*q6*q7 + 540*p*q1*q2*q3*q4*q5**3*q6 - 60*p*q1*q2*q3*q4*q5**2*q6**3 - 60*p*q1*q2*q3*q4*q5**2*q6**2*q7 + 540*p*q1*q2*q3*q4*q5**2*q6**2 - 60*p*q1*q2*q3*q4*q5**2*q6*q7**2 + 540*p*q1*q2*q3*q4*q5**2*q6*q7 - 2160*p*q1*q2*q3*q4*q5**2*q6 - 60*p*q1*q2*q3*q4*q5*q6**4 - 60*p*q1*q2*q3*q4*q5*q6**3*q7 + 540*p*q1*q2*q3*q4*q5*q6**3 - 60*p*q1*q2*q3*q4*q5*q6**2*q7**2 + 540*p*q1*q2*q3*q4*q5*q6**2*q7 - 2160*p*q1*q2*q3*q4*q5*q6**2 - 60*p*q1*q2*q3*q4*q5*q6*q7**3 + 540*p*q1*q2*q3*q4*q5*q6*q7**2 - 2160*p*q1*q2*q3*q4*q5*q6*q7 + 5040*p*q1*q2*q3*q4*q5*q6 + 45*p*q2**4*q3*q4*q5*q6 + 45*p*q2**3*q3**2*q4*q5*q6 + 45*p*q2**3*q3*q4**2*q5*q6 + 45*p*q2**3*q3*q4*q5**2*q6 + 45*p*q2**3*q3*q4*q5*q6**2 + 45*p*q2**3*q3*q4*q5*q6*q7 - 360*p*q2**3*q3*q4*q5*q6 + 45*p*q2**2*q3**3*q4*q5*q6 + 45*p*q2**2*q3**2*q4**2*q5*q6 + 45*p*q2**2*q3**2*q4*q5**2*q6 + 45*p*q2**2*q3**2*q4*q5*q6**2 + 45*p*q2**2*q3**2*q4*q5*q6*q7 - 360*p*q2**2*q3**2*q4*q5*q6 + 45*p*q2**2*q3*q4**3*q5*q6 + 45*p*q2**2*q3*q4**2*q5**2*q6 + 45*p*q2**2*q3*q4**2*q5*q6**2 + 45*p*q2**2*q3*q4**2*q5*q6*q7 - 360*p*q2**2*q3*q4**2*q5*q6 + 45*p*q2**2*q3*q4*q5**3*q6 + 45*p*q2**2*q3*q4*q5**2*q6**2 + 45*p*q2**2*q3*q4*q5**2*q6*q7 - 360*p*q2**2*q3*q4*q5**2*q6 + 45*p*q2**2*q3*q4*q5*q6**3 + 45*p*q2**2*q3*q4*q5*q6**2*q7 - 360*p*q2**2*q3*q4*q5*q6**2 + 45*p*q2**2*q3*q4*q5*q6*q7**2 - 360*p*q2**2*q3*q4*q5*q6*q7 + 1260*p*q2**2*q3*q4*q5*q6 + 45*p*q2*q3**4*q4*q5*q6 + 45*p*q2*q3**3*q4**2*q5*q6 + 45*p*q2*q3**3*q4*q5**2*q6 + 45*p*q2*q3**3*q4*q5*q6**2 + 45*p*q2*q3**3*q4*q5*q6*q7 - 360*p*q2*q3**3*q4*q5*q6 + 45*p*q2*q3**2*q4**3*q5*q6 + 45*p*q2*q3**2*q4**2*q5**2*q6 + 45*p*q2*q3**2*q4**2*q5*q6**2 + 45*p*q2*q3**2*q4**2*q5*q6*q7 - 360*p*q2*q3**2*q4**2*q5*q6 + 45*p*q2*q3**2*q4*q5**3*q6 + 45*p*q2*q3**2*q4*q5**2*q6**2 + 45*p*q2*q3**2*q4*q5**2*q6*q7 - 360*p*q2*q3**2*q4*q5**2*q6 + 45*p*q2*q3**2*q4*q5*q6**3 + 45*p*q2*q3**2*q4*q5*q6**2*q7 - 360*p*q2*q3**2*q4*q5*q6**2 + 45*p*q2*q3**2*q4*q5*q6*q7**2 - 360*p*q2*q3**2*q4*q5*q6*q7 + 1260*p*q2*q3**2*q4*q5*q6 + 45*p*q2*q3*q4**4*q5*q6 + 45*p*q2*q3*q4**3*q5**2*q6 + 45*p*q2*q3*q4**3*q5*q6**2 + 45*p*q2*q3*q4**3*q5*q6*q7 - 360*p*q2*q3*q4**3*q5*q6 + 45*p*q2*q3*q4**2*q5**3*q6 + 45*p*q2*q3*q4**2*q5**2*q6**2 + 45*p*q2*q3*q4**2*q5**2*q6*q7 - 360*p*q2*q3*q4**2*q5**2*q6 + 45*p*q2*q3*q4**2*q5*q6**3 + 45*p*q2*q3*q4**2*q5*q6**2*q7 - 360*p*q2*q3*q4**2*q5*q6**2 + 45*p*q2*q3*q4**2*q5*q6*q7**2 - 360*p*q2*q3*q4**2*q5*q6*q7 + 1260*p*q2*q3*q4**2*q5*q6 + 45*p*q2*q3*q4*q5**4*q6 + 45*p*q2*q3*q4*q5**3*q6**2 + 45*p*q2*q3*q4*q5**3*q6*q7 - 360*p*q2*q3*q4*q5**3*q6 + 45*p*q2*q3*q4*q5**2*q6**3 + 45*p*q2*q3*q4*q5**2*q6**2*q7 - 360*p*q2*q3*q4*q5**2*q6**2 + 45*p*q2*q3*q4*q5**2*q6*q7**2 - 360*p*q2*q3*q4*q5**2*q6*q7 + 1260*p*q2*q3*q4*q5**2*q6 + 45*p*q2*q3*q4*q5*q6**4 + 45*p*q2*q3*q4*q5*q6**3*q7 - 360*p*q2*q3*q4*q5*q6**3 + 45*p*q2*q3*q4*q5*q6**2*q7**2 - 360*p*q2*q3*q4*q5*q6**2*q7 + 1260*p*q2*q3*q4*q5*q6**2 + 45*p*q2*q3*q4*q5*q6*q7**3 - 360*p*q2*q3*q4*q5*q6*q7**2 + 1260*p*q2*q3*q4*q5*q6*q7 - 2520*p*q2*q3*q4*q5*q6 + 10*q1**4*q2*q3*q4*q5*q6 + 10*q1**3*q2**2*q3*q4*q5*q6 + 10*q1**3*q2*q3**2*q4*q5*q6 + 10*q1**3*q2*q3*q4**2*q5*q6 + 10*q1**3*q2*q3*q4*q5**2*q6 + 10*q1**3*q2*q3*q4*q5*q6**2 + 10*q1**3*q2*q3*q4*q5*q6*q7 - 90*q1**3*q2*q3*q4*q5*q6 + 10*q1**2*q2**3*q3*q4*q5*q6 + 10*q1**2*q2**2*q3**2*q4*q5*q6 + 10*q1**2*q2**2*q3*q4**2*q5*q6 + 10*q1**2*q2**2*q3*q4*q5**2*q6 + 10*q1**2*q2**2*q3*q4*q5*q6**2 + 10*q1**2*q2**2*q3*q4*q5*q6*q7 - 90*q1**2*q2**2*q3*q4*q5*q6 + 10*q1**2*q2*q3**3*q4*q5*q6 + 10*q1**2*q2*q3**2*q4**2*q5*q6 + 10*q1**2*q2*q3**2*q4*q5**2*q6 + 10*q1**2*q2*q3**2*q4*q5*q6**2 + 10*q1**2*q2*q3**2*q4*q5*q6*q7 - 90*q1**2*q2*q3**2*q4*q5*q6 + 10*q1**2*q2*q3*q4**3*q5*q6 + 10*q1**2*q2*q3*q4**2*q5**2*q6 + 10*q1**2*q2*q3*q4**2*q5*q6**2 + 10*q1**2*q2*q3*q4**2*q5*q6*q7 - 90*q1**2*q2*q3*q4**2*q5*q6 + 10*q1**2*q2*q3*q4*q5**3*q6 + 10*q1**2*q2*q3*q4*q5**2*q6**2 + 10*q1**2*q2*q3*q4*q5**2*q6*q7 - 90*q1**2*q2*q3*q4*q5**2*q6 + 10*q1**2*q2*q3*q4*q5*q6**3 + 10*q1**2*q2*q3*q4*q5*q6**2*q7 - 90*q1**2*q2*q3*q4*q5*q6**2 + 10*q1**2*q2*q3*q4*q5*q6*q7**2 - 90*q1**2*q2*q3*q4*q5*q6*q7 + 360*q1**2*q2*q3*q4*q5*q6 + 10*q1*q2**4*q3*q4*q5*q6 + 10*q1*q2**3*q3**2*q4*q5*q6 + 10*q1*q2**3*q3*q4**2*q5*q6 + 10*q1*q2**3*q3*q4*q5**2*q6 + 10*q1*q2**3*q3*q4*q5*q6**2 + 10*q1*q2**3*q3*q4*q5*q6*q7 - 90*q1*q2**3*q3*q4*q5*q6 + 10*q1*q2**2*q3**3*q4*q5*q6 + 10*q1*q2**2*q3**2*q4**2*q5*q6 + 10*q1*q2**2*q3**2*q4*q5**2*q6 + 10*q1*q2**2*q3**2*q4*q5*q6**2 + 10*q1*q2**2*q3**2*q4*q5*q6*q7 - 90*q1*q2**2*q3**2*q4*q5*q6 + 10*q1*q2**2*q3*q4**3*q5*q6 + 10*q1*q2**2*q3*q4**2*q5**2*q6 + 10*q1*q2**2*q3*q4**2*q5*q6**2 + 10*q1*q2**2*q3*q4**2*q5*q6*q7 - 90*q1*q2**2*q3*q4**2*q5*q6 + 10*q1*q2**2*q3*q4*q5**3*q6 + 10*q1*q2**2*q3*q4*q5**2*q6**2 + 10*q1*q2**2*q3*q4*q5**2*q6*q7 - 90*q1*q2**2*q3*q4*q5**2*q6 + 10*q1*q2**2*q3*q4*q5*q6**3 + 10*q1*q2**2*q3*q4*q5*q6**2*q7 - 90*q1*q2**2*q3*q4*q5*q6**2 + 10*q1*q2**2*q3*q4*q5*q6*q7**2 - 90*q1*q2**2*q3*q4*q5*q6*q7 + 360*q1*q2**2*q3*q4*q5*q6 + 10*q1*q2*q3**4*q4*q5*q6 + 10*q1*q2*q3**3*q4**2*q5*q6 + 10*q1*q2*q3**3*q4*q5**2*q6 + 10*q1*q2*q3**3*q4*q5*q6**2 + 10*q1*q2*q3**3*q4*q5*q6*q7 - 90*q1*q2*q3**3*q4*q5*q6 + 10*q1*q2*q3**2*q4**3*q5*q6 + 10*q1*q2*q3**2*q4**2*q5**2*q6 + 10*q1*q2*q3**2*q4**2*q5*q6**2 + 10*q1*q2*q3**2*q4**2*q5*q6*q7 - 90*q1*q2*q3**2*q4**2*q5*q6 + 10*q1*q2*q3**2*q4*q5**3*q6 + 10*q1*q2*q3**2*q4*q5**2*q6**2 + 10*q1*q2*q3**2*q4*q5**2*q6*q7 - 90*q1*q2*q3**2*q4*q5**2*q6 + 10*q1*q2*q3**2*q4*q5*q6**3 + 10*q1*q2*q3**2*q4*q5*q6**2*q7 - 90*q1*q2*q3**2*q4*q5*q6**2 + 10*q1*q2*q3**2*q4*q5*q6*q7**2 - 90*q1*q2*q3**2*q4*q5*q6*q7 + 360*q1*q2*q3**2*q4*q5*q6 + 10*q1*q2*q3*q4**4*q5*q6 + 10*q1*q2*q3*q4**3*q5**2*q6 + 10*q1*q2*q3*q4**3*q5*q6**2 + 10*q1*q2*q3*q4**3*q5*q6*q7 - 90*q1*q2*q3*q4**3*q5*q6 + 10*q1*q2*q3*q4**2*q5**3*q6 + 10*q1*q2*q3*q4**2*q5**2*q6**2 + 10*q1*q2*q3*q4**2*q5**2*q6*q7 - 90*q1*q2*q3*q4**2*q5**2*q6 + 10*q1*q2*q3*q4**2*q5*q6**3 + 10*q1*q2*q3*q4**2*q5*q6**2*q7 - 90*q1*q2*q3*q4**2*q5*q6**2 + 10*q1*q2*q3*q4**2*q5*q6*q7**2 - 90*q1*q2*q3*q4**2*q5*q6*q7 + 360*q1*q2*q3*q4**2*q5*q6 + 10*q1*q2*q3*q4*q5**4*q6 + 10*q1*q2*q3*q4*q5**3*q6**2 + 10*q1*q2*q3*q4*q5**3*q6*q7 - 90*q1*q2*q3*q4*q5**3*q6 + 10*q1*q2*q3*q4*q5**2*q6**3 + 10*q1*q2*q3*q4*q5**2*q6**2*q7 - 90*q1*q2*q3*q4*q5**2*q6**2 + 10*q1*q2*q3*q4*q5**2*q6*q7**2 - 90*q1*q2*q3*q4*q5**2*q6*q7 + 360*q1*q2*q3*q4*q5**2*q6 + 10*q1*q2*q3*q4*q5*q6**4 + 10*q1*q2*q3*q4*q5*q6**3*q7 - 90*q1*q2*q3*q4*q5*q6**3 + 10*q1*q2*q3*q4*q5*q6**2*q7**2 - 90*q1*q2*q3*q4*q5*q6**2*q7 + 360*q1*q2*q3*q4*q5*q6**2 + 10*q1*q2*q3*q4*q5*q6*q7**3 - 90*q1*q2*q3*q4*q5*q6*q7**2 + 360*q1*q2*q3*q4*q5*q6*q7 - 840*q1*q2*q3*q4*q5*q6)'
        f_multiparam[10][8] = '-p*(p - 1)**2*(10*p**7*q1**3*q2*q3*q4*q5*q6*q7 + 10*p**7*q1**2*q2**2*q3*q4*q5*q6*q7 + 10*p**7*q1**2*q2*q3**2*q4*q5*q6*q7 + 10*p**7*q1**2*q2*q3*q4**2*q5*q6*q7 + 10*p**7*q1**2*q2*q3*q4*q5**2*q6*q7 + 10*p**7*q1**2*q2*q3*q4*q5*q6**2*q7 + 10*p**7*q1**2*q2*q3*q4*q5*q6*q7**2 + 10*p**7*q1**2*q2*q3*q4*q5*q6*q7*q8 - 90*p**7*q1**2*q2*q3*q4*q5*q6*q7 + 10*p**7*q1*q2**3*q3*q4*q5*q6*q7 + 10*p**7*q1*q2**2*q3**2*q4*q5*q6*q7 + 10*p**7*q1*q2**2*q3*q4**2*q5*q6*q7 + 10*p**7*q1*q2**2*q3*q4*q5**2*q6*q7 + 10*p**7*q1*q2**2*q3*q4*q5*q6**2*q7 + 10*p**7*q1*q2**2*q3*q4*q5*q6*q7**2 + 10*p**7*q1*q2**2*q3*q4*q5*q6*q7*q8 - 90*p**7*q1*q2**2*q3*q4*q5*q6*q7 + 10*p**7*q1*q2*q3**3*q4*q5*q6*q7 + 10*p**7*q1*q2*q3**2*q4**2*q5*q6*q7 + 10*p**7*q1*q2*q3**2*q4*q5**2*q6*q7 + 10*p**7*q1*q2*q3**2*q4*q5*q6**2*q7 + 10*p**7*q1*q2*q3**2*q4*q5*q6*q7**2 + 10*p**7*q1*q2*q3**2*q4*q5*q6*q7*q8 - 90*p**7*q1*q2*q3**2*q4*q5*q6*q7 + 10*p**7*q1*q2*q3*q4**3*q5*q6*q7 + 10*p**7*q1*q2*q3*q4**2*q5**2*q6*q7 + 10*p**7*q1*q2*q3*q4**2*q5*q6**2*q7 + 10*p**7*q1*q2*q3*q4**2*q5*q6*q7**2 + 10*p**7*q1*q2*q3*q4**2*q5*q6*q7*q8 - 90*p**7*q1*q2*q3*q4**2*q5*q6*q7 + 10*p**7*q1*q2*q3*q4*q5**3*q6*q7 + 10*p**7*q1*q2*q3*q4*q5**2*q6**2*q7 + 10*p**7*q1*q2*q3*q4*q5**2*q6*q7**2 + 10*p**7*q1*q2*q3*q4*q5**2*q6*q7*q8 - 90*p**7*q1*q2*q3*q4*q5**2*q6*q7 + 10*p**7*q1*q2*q3*q4*q5*q6**3*q7 + 10*p**7*q1*q2*q3*q4*q5*q6**2*q7**2 + 10*p**7*q1*q2*q3*q4*q5*q6**2*q7*q8 - 90*p**7*q1*q2*q3*q4*q5*q6**2*q7 + 10*p**7*q1*q2*q3*q4*q5*q6*q7**3 + 10*p**7*q1*q2*q3*q4*q5*q6*q7**2*q8 - 90*p**7*q1*q2*q3*q4*q5*q6*q7**2 + 10*p**7*q1*q2*q3*q4*q5*q6*q7*q8**2 - 90*p**7*q1*q2*q3*q4*q5*q6*q7*q8 + 360*p**7*q1*q2*q3*q4*q5*q6*q7 - 45*p**7*q2**3*q3*q4*q5*q6*q7 - 45*p**7*q2**2*q3**2*q4*q5*q6*q7 - 45*p**7*q2**2*q3*q4**2*q5*q6*q7 - 45*p**7*q2**2*q3*q4*q5**2*q6*q7 - 45*p**7*q2**2*q3*q4*q5*q6**2*q7 - 45*p**7*q2**2*q3*q4*q5*q6*q7**2 - 45*p**7*q2**2*q3*q4*q5*q6*q7*q8 + 360*p**7*q2**2*q3*q4*q5*q6*q7 - 45*p**7*q2*q3**3*q4*q5*q6*q7 - 45*p**7*q2*q3**2*q4**2*q5*q6*q7 - 45*p**7*q2*q3**2*q4*q5**2*q6*q7 - 45*p**7*q2*q3**2*q4*q5*q6**2*q7 - 45*p**7*q2*q3**2*q4*q5*q6*q7**2 - 45*p**7*q2*q3**2*q4*q5*q6*q7*q8 + 360*p**7*q2*q3**2*q4*q5*q6*q7 - 45*p**7*q2*q3*q4**3*q5*q6*q7 - 45*p**7*q2*q3*q4**2*q5**2*q6*q7 - 45*p**7*q2*q3*q4**2*q5*q6**2*q7 - 45*p**7*q2*q3*q4**2*q5*q6*q7**2 - 45*p**7*q2*q3*q4**2*q5*q6*q7*q8 + 360*p**7*q2*q3*q4**2*q5*q6*q7 - 45*p**7*q2*q3*q4*q5**3*q6*q7 - 45*p**7*q2*q3*q4*q5**2*q6**2*q7 - 45*p**7*q2*q3*q4*q5**2*q6*q7**2 - 45*p**7*q2*q3*q4*q5**2*q6*q7*q8 + 360*p**7*q2*q3*q4*q5**2*q6*q7 - 45*p**7*q2*q3*q4*q5*q6**3*q7 - 45*p**7*q2*q3*q4*q5*q6**2*q7**2 - 45*p**7*q2*q3*q4*q5*q6**2*q7*q8 + 360*p**7*q2*q3*q4*q5*q6**2*q7 - 45*p**7*q2*q3*q4*q5*q6*q7**3 - 45*p**7*q2*q3*q4*q5*q6*q7**2*q8 + 360*p**7*q2*q3*q4*q5*q6*q7**2 - 45*p**7*q2*q3*q4*q5*q6*q7*q8**2 + 360*p**7*q2*q3*q4*q5*q6*q7*q8 - 1260*p**7*q2*q3*q4*q5*q6*q7 + 120*p**7*q3**3*q4*q5*q6*q7 + 120*p**7*q3**2*q4**2*q5*q6*q7 + 120*p**7*q3**2*q4*q5**2*q6*q7 + 120*p**7*q3**2*q4*q5*q6**2*q7 + 120*p**7*q3**2*q4*q5*q6*q7**2 + 120*p**7*q3**2*q4*q5*q6*q7*q8 - 840*p**7*q3**2*q4*q5*q6*q7 + 120*p**7*q3*q4**3*q5*q6*q7 + 120*p**7*q3*q4**2*q5**2*q6*q7 + 120*p**7*q3*q4**2*q5*q6**2*q7 + 120*p**7*q3*q4**2*q5*q6*q7**2 + 120*p**7*q3*q4**2*q5*q6*q7*q8 - 840*p**7*q3*q4**2*q5*q6*q7 + 120*p**7*q3*q4*q5**3*q6*q7 + 120*p**7*q3*q4*q5**2*q6**2*q7 + 120*p**7*q3*q4*q5**2*q6*q7**2 + 120*p**7*q3*q4*q5**2*q6*q7*q8 - 840*p**7*q3*q4*q5**2*q6*q7 + 120*p**7*q3*q4*q5*q6**3*q7 + 120*p**7*q3*q4*q5*q6**2*q7**2 + 120*p**7*q3*q4*q5*q6**2*q7*q8 - 840*p**7*q3*q4*q5*q6**2*q7 + 120*p**7*q3*q4*q5*q6*q7**3 + 120*p**7*q3*q4*q5*q6*q7**2*q8 - 840*p**7*q3*q4*q5*q6*q7**2 + 120*p**7*q3*q4*q5*q6*q7*q8**2 - 840*p**7*q3*q4*q5*q6*q7*q8 + 2520*p**7*q3*q4*q5*q6*q7 - 210*p**7*q4**3*q5*q6*q7 - 210*p**7*q4**2*q5**2*q6*q7 - 210*p**7*q4**2*q5*q6**2*q7 - 210*p**7*q4**2*q5*q6*q7**2 - 210*p**7*q4**2*q5*q6*q7*q8 + 1260*p**7*q4**2*q5*q6*q7 - 210*p**7*q4*q5**3*q6*q7 - 210*p**7*q4*q5**2*q6**2*q7 - 210*p**7*q4*q5**2*q6*q7**2 - 210*p**7*q4*q5**2*q6*q7*q8 + 1260*p**7*q4*q5**2*q6*q7 - 210*p**7*q4*q5*q6**3*q7 - 210*p**7*q4*q5*q6**2*q7**2 - 210*p**7*q4*q5*q6**2*q7*q8 + 1260*p**7*q4*q5*q6**2*q7 - 210*p**7*q4*q5*q6*q7**3 - 210*p**7*q4*q5*q6*q7**2*q8 + 1260*p**7*q4*q5*q6*q7**2 - 210*p**7*q4*q5*q6*q7*q8**2 + 1260*p**7*q4*q5*q6*q7*q8 - 3150*p**7*q4*q5*q6*q7 + 252*p**7*q5**3*q6*q7 + 252*p**7*q5**2*q6**2*q7 + 252*p**7*q5**2*q6*q7**2 + 252*p**7*q5**2*q6*q7*q8 - 1260*p**7*q5**2*q6*q7 + 252*p**7*q5*q6**3*q7 + 252*p**7*q5*q6**2*q7**2 + 252*p**7*q5*q6**2*q7*q8 - 1260*p**7*q5*q6**2*q7 + 252*p**7*q5*q6*q7**3 + 252*p**7*q5*q6*q7**2*q8 - 1260*p**7*q5*q6*q7**2 + 252*p**7*q5*q6*q7*q8**2 - 1260*p**7*q5*q6*q7*q8 + 2520*p**7*q5*q6*q7 - 210*p**7*q6**3*q7 - 210*p**7*q6**2*q7**2 - 210*p**7*q6**2*q7*q8 + 840*p**7*q6**2*q7 - 210*p**7*q6*q7**3 - 210*p**7*q6*q7**2*q8 + 840*p**7*q6*q7**2 - 210*p**7*q6*q7*q8**2 + 840*p**7*q6*q7*q8 - 1260*p**7*q6*q7 + 120*p**7*q7**3 + 120*p**7*q7**2*q8 - 360*p**7*q7**2 + 120*p**7*q7*q8**2 - 360*p**7*q7*q8 + 360*p**7*q7 - 45*p**7*q8**2 + 90*p**7*q8 - 45*p**7 - 70*p**6*q1**3*q2*q3*q4*q5*q6*q7 - 70*p**6*q1**2*q2**2*q3*q4*q5*q6*q7 - 70*p**6*q1**2*q2*q3**2*q4*q5*q6*q7 - 70*p**6*q1**2*q2*q3*q4**2*q5*q6*q7 - 70*p**6*q1**2*q2*q3*q4*q5**2*q6*q7 - 70*p**6*q1**2*q2*q3*q4*q5*q6**2*q7 - 70*p**6*q1**2*q2*q3*q4*q5*q6*q7**2 - 70*p**6*q1**2*q2*q3*q4*q5*q6*q7*q8 + 630*p**6*q1**2*q2*q3*q4*q5*q6*q7 - 70*p**6*q1*q2**3*q3*q4*q5*q6*q7 - 70*p**6*q1*q2**2*q3**2*q4*q5*q6*q7 - 70*p**6*q1*q2**2*q3*q4**2*q5*q6*q7 - 70*p**6*q1*q2**2*q3*q4*q5**2*q6*q7 - 70*p**6*q1*q2**2*q3*q4*q5*q6**2*q7 - 70*p**6*q1*q2**2*q3*q4*q5*q6*q7**2 - 70*p**6*q1*q2**2*q3*q4*q5*q6*q7*q8 + 630*p**6*q1*q2**2*q3*q4*q5*q6*q7 - 70*p**6*q1*q2*q3**3*q4*q5*q6*q7 - 70*p**6*q1*q2*q3**2*q4**2*q5*q6*q7 - 70*p**6*q1*q2*q3**2*q4*q5**2*q6*q7 - 70*p**6*q1*q2*q3**2*q4*q5*q6**2*q7 - 70*p**6*q1*q2*q3**2*q4*q5*q6*q7**2 - 70*p**6*q1*q2*q3**2*q4*q5*q6*q7*q8 + 630*p**6*q1*q2*q3**2*q4*q5*q6*q7 - 70*p**6*q1*q2*q3*q4**3*q5*q6*q7 - 70*p**6*q1*q2*q3*q4**2*q5**2*q6*q7 - 70*p**6*q1*q2*q3*q4**2*q5*q6**2*q7 - 70*p**6*q1*q2*q3*q4**2*q5*q6*q7**2 - 70*p**6*q1*q2*q3*q4**2*q5*q6*q7*q8 + 630*p**6*q1*q2*q3*q4**2*q5*q6*q7 - 70*p**6*q1*q2*q3*q4*q5**3*q6*q7 - 70*p**6*q1*q2*q3*q4*q5**2*q6**2*q7 - 70*p**6*q1*q2*q3*q4*q5**2*q6*q7**2 - 70*p**6*q1*q2*q3*q4*q5**2*q6*q7*q8 + 630*p**6*q1*q2*q3*q4*q5**2*q6*q7 - 70*p**6*q1*q2*q3*q4*q5*q6**3*q7 - 70*p**6*q1*q2*q3*q4*q5*q6**2*q7**2 - 70*p**6*q1*q2*q3*q4*q5*q6**2*q7*q8 + 630*p**6*q1*q2*q3*q4*q5*q6**2*q7 - 70*p**6*q1*q2*q3*q4*q5*q6*q7**3 - 70*p**6*q1*q2*q3*q4*q5*q6*q7**2*q8 + 630*p**6*q1*q2*q3*q4*q5*q6*q7**2 - 70*p**6*q1*q2*q3*q4*q5*q6*q7*q8**2 + 630*p**6*q1*q2*q3*q4*q5*q6*q7*q8 - 2520*p**6*q1*q2*q3*q4*q5*q6*q7 + 270*p**6*q2**3*q3*q4*q5*q6*q7 + 270*p**6*q2**2*q3**2*q4*q5*q6*q7 + 270*p**6*q2**2*q3*q4**2*q5*q6*q7 + 270*p**6*q2**2*q3*q4*q5**2*q6*q7 + 270*p**6*q2**2*q3*q4*q5*q6**2*q7 + 270*p**6*q2**2*q3*q4*q5*q6*q7**2 + 270*p**6*q2**2*q3*q4*q5*q6*q7*q8 - 2160*p**6*q2**2*q3*q4*q5*q6*q7 + 270*p**6*q2*q3**3*q4*q5*q6*q7 + 270*p**6*q2*q3**2*q4**2*q5*q6*q7 + 270*p**6*q2*q3**2*q4*q5**2*q6*q7 + 270*p**6*q2*q3**2*q4*q5*q6**2*q7 + 270*p**6*q2*q3**2*q4*q5*q6*q7**2 + 270*p**6*q2*q3**2*q4*q5*q6*q7*q8 - 2160*p**6*q2*q3**2*q4*q5*q6*q7 + 270*p**6*q2*q3*q4**3*q5*q6*q7 + 270*p**6*q2*q3*q4**2*q5**2*q6*q7 + 270*p**6*q2*q3*q4**2*q5*q6**2*q7 + 270*p**6*q2*q3*q4**2*q5*q6*q7**2 + 270*p**6*q2*q3*q4**2*q5*q6*q7*q8 - 2160*p**6*q2*q3*q4**2*q5*q6*q7 + 270*p**6*q2*q3*q4*q5**3*q6*q7 + 270*p**6*q2*q3*q4*q5**2*q6**2*q7 + 270*p**6*q2*q3*q4*q5**2*q6*q7**2 + 270*p**6*q2*q3*q4*q5**2*q6*q7*q8 - 2160*p**6*q2*q3*q4*q5**2*q6*q7 + 270*p**6*q2*q3*q4*q5*q6**3*q7 + 270*p**6*q2*q3*q4*q5*q6**2*q7**2 + 270*p**6*q2*q3*q4*q5*q6**2*q7*q8 - 2160*p**6*q2*q3*q4*q5*q6**2*q7 + 270*p**6*q2*q3*q4*q5*q6*q7**3 + 270*p**6*q2*q3*q4*q5*q6*q7**2*q8 - 2160*p**6*q2*q3*q4*q5*q6*q7**2 + 270*p**6*q2*q3*q4*q5*q6*q7*q8**2 - 2160*p**6*q2*q3*q4*q5*q6*q7*q8 + 7560*p**6*q2*q3*q4*q5*q6*q7 - 600*p**6*q3**3*q4*q5*q6*q7 - 600*p**6*q3**2*q4**2*q5*q6*q7 - 600*p**6*q3**2*q4*q5**2*q6*q7 - 600*p**6*q3**2*q4*q5*q6**2*q7 - 600*p**6*q3**2*q4*q5*q6*q7**2 - 600*p**6*q3**2*q4*q5*q6*q7*q8 + 4200*p**6*q3**2*q4*q5*q6*q7 - 600*p**6*q3*q4**3*q5*q6*q7 - 600*p**6*q3*q4**2*q5**2*q6*q7 - 600*p**6*q3*q4**2*q5*q6**2*q7 - 600*p**6*q3*q4**2*q5*q6*q7**2 - 600*p**6*q3*q4**2*q5*q6*q7*q8 + 4200*p**6*q3*q4**2*q5*q6*q7 - 600*p**6*q3*q4*q5**3*q6*q7 - 600*p**6*q3*q4*q5**2*q6**2*q7 - 600*p**6*q3*q4*q5**2*q6*q7**2 - 600*p**6*q3*q4*q5**2*q6*q7*q8 + 4200*p**6*q3*q4*q5**2*q6*q7 - 600*p**6*q3*q4*q5*q6**3*q7 - 600*p**6*q3*q4*q5*q6**2*q7**2 - 600*p**6*q3*q4*q5*q6**2*q7*q8 + 4200*p**6*q3*q4*q5*q6**2*q7 - 600*p**6*q3*q4*q5*q6*q7**3 - 600*p**6*q3*q4*q5*q6*q7**2*q8 + 4200*p**6*q3*q4*q5*q6*q7**2 - 600*p**6*q3*q4*q5*q6*q7*q8**2 + 4200*p**6*q3*q4*q5*q6*q7*q8 - 12600*p**6*q3*q4*q5*q6*q7 + 840*p**6*q4**3*q5*q6*q7 + 840*p**6*q4**2*q5**2*q6*q7 + 840*p**6*q4**2*q5*q6**2*q7 + 840*p**6*q4**2*q5*q6*q7**2 + 840*p**6*q4**2*q5*q6*q7*q8 - 5040*p**6*q4**2*q5*q6*q7 + 840*p**6*q4*q5**3*q6*q7 + 840*p**6*q4*q5**2*q6**2*q7 + 840*p**6*q4*q5**2*q6*q7**2 + 840*p**6*q4*q5**2*q6*q7*q8 - 5040*p**6*q4*q5**2*q6*q7 + 840*p**6*q4*q5*q6**3*q7 + 840*p**6*q4*q5*q6**2*q7**2 + 840*p**6*q4*q5*q6**2*q7*q8 - 5040*p**6*q4*q5*q6**2*q7 + 840*p**6*q4*q5*q6*q7**3 + 840*p**6*q4*q5*q6*q7**2*q8 - 5040*p**6*q4*q5*q6*q7**2 + 840*p**6*q4*q5*q6*q7*q8**2 - 5040*p**6*q4*q5*q6*q7*q8 + 12600*p**6*q4*q5*q6*q7 - 756*p**6*q5**3*q6*q7 - 756*p**6*q5**2*q6**2*q7 - 756*p**6*q5**2*q6*q7**2 - 756*p**6*q5**2*q6*q7*q8 + 3780*p**6*q5**2*q6*q7 - 756*p**6*q5*q6**3*q7 - 756*p**6*q5*q6**2*q7**2 - 756*p**6*q5*q6**2*q7*q8 + 3780*p**6*q5*q6**2*q7 - 756*p**6*q5*q6*q7**3 - 756*p**6*q5*q6*q7**2*q8 + 3780*p**6*q5*q6*q7**2 - 756*p**6*q5*q6*q7*q8**2 + 3780*p**6*q5*q6*q7*q8 - 7560*p**6*q5*q6*q7 + 420*p**6*q6**3*q7 + 420*p**6*q6**2*q7**2 + 420*p**6*q6**2*q7*q8 - 1680*p**6*q6**2*q7 + 420*p**6*q6*q7**3 + 420*p**6*q6*q7**2*q8 - 1680*p**6*q6*q7**2 + 420*p**6*q6*q7*q8**2 - 1680*p**6*q6*q7*q8 + 2520*p**6*q6*q7 - 120*p**6*q7**3 - 120*p**6*q7**2*q8 + 360*p**6*q7**2 - 120*p**6*q7*q8**2 + 360*p**6*q7*q8 - 360*p**6*q7 + 210*p**5*q1**3*q2*q3*q4*q5*q6*q7 + 210*p**5*q1**2*q2**2*q3*q4*q5*q6*q7 + 210*p**5*q1**2*q2*q3**2*q4*q5*q6*q7 + 210*p**5*q1**2*q2*q3*q4**2*q5*q6*q7 + 210*p**5*q1**2*q2*q3*q4*q5**2*q6*q7 + 210*p**5*q1**2*q2*q3*q4*q5*q6**2*q7 + 210*p**5*q1**2*q2*q3*q4*q5*q6*q7**2 + 210*p**5*q1**2*q2*q3*q4*q5*q6*q7*q8 - 1890*p**5*q1**2*q2*q3*q4*q5*q6*q7 + 210*p**5*q1*q2**3*q3*q4*q5*q6*q7 + 210*p**5*q1*q2**2*q3**2*q4*q5*q6*q7 + 210*p**5*q1*q2**2*q3*q4**2*q5*q6*q7 + 210*p**5*q1*q2**2*q3*q4*q5**2*q6*q7 + 210*p**5*q1*q2**2*q3*q4*q5*q6**2*q7 + 210*p**5*q1*q2**2*q3*q4*q5*q6*q7**2 + 210*p**5*q1*q2**2*q3*q4*q5*q6*q7*q8 - 1890*p**5*q1*q2**2*q3*q4*q5*q6*q7 + 210*p**5*q1*q2*q3**3*q4*q5*q6*q7 + 210*p**5*q1*q2*q3**2*q4**2*q5*q6*q7 + 210*p**5*q1*q2*q3**2*q4*q5**2*q6*q7 + 210*p**5*q1*q2*q3**2*q4*q5*q6**2*q7 + 210*p**5*q1*q2*q3**2*q4*q5*q6*q7**2 + 210*p**5*q1*q2*q3**2*q4*q5*q6*q7*q8 - 1890*p**5*q1*q2*q3**2*q4*q5*q6*q7 + 210*p**5*q1*q2*q3*q4**3*q5*q6*q7 + 210*p**5*q1*q2*q3*q4**2*q5**2*q6*q7 + 210*p**5*q1*q2*q3*q4**2*q5*q6**2*q7 + 210*p**5*q1*q2*q3*q4**2*q5*q6*q7**2 + 210*p**5*q1*q2*q3*q4**2*q5*q6*q7*q8 - 1890*p**5*q1*q2*q3*q4**2*q5*q6*q7 + 210*p**5*q1*q2*q3*q4*q5**3*q6*q7 + 210*p**5*q1*q2*q3*q4*q5**2*q6**2*q7 + 210*p**5*q1*q2*q3*q4*q5**2*q6*q7**2 + 210*p**5*q1*q2*q3*q4*q5**2*q6*q7*q8 - 1890*p**5*q1*q2*q3*q4*q5**2*q6*q7 + 210*p**5*q1*q2*q3*q4*q5*q6**3*q7 + 210*p**5*q1*q2*q3*q4*q5*q6**2*q7**2 + 210*p**5*q1*q2*q3*q4*q5*q6**2*q7*q8 - 1890*p**5*q1*q2*q3*q4*q5*q6**2*q7 + 210*p**5*q1*q2*q3*q4*q5*q6*q7**3 + 210*p**5*q1*q2*q3*q4*q5*q6*q7**2*q8 - 1890*p**5*q1*q2*q3*q4*q5*q6*q7**2 + 210*p**5*q1*q2*q3*q4*q5*q6*q7*q8**2 - 1890*p**5*q1*q2*q3*q4*q5*q6*q7*q8 + 7560*p**5*q1*q2*q3*q4*q5*q6*q7 - 675*p**5*q2**3*q3*q4*q5*q6*q7 - 675*p**5*q2**2*q3**2*q4*q5*q6*q7 - 675*p**5*q2**2*q3*q4**2*q5*q6*q7 - 675*p**5*q2**2*q3*q4*q5**2*q6*q7 - 675*p**5*q2**2*q3*q4*q5*q6**2*q7 - 675*p**5*q2**2*q3*q4*q5*q6*q7**2 - 675*p**5*q2**2*q3*q4*q5*q6*q7*q8 + 5400*p**5*q2**2*q3*q4*q5*q6*q7 - 675*p**5*q2*q3**3*q4*q5*q6*q7 - 675*p**5*q2*q3**2*q4**2*q5*q6*q7 - 675*p**5*q2*q3**2*q4*q5**2*q6*q7 - 675*p**5*q2*q3**2*q4*q5*q6**2*q7 - 675*p**5*q2*q3**2*q4*q5*q6*q7**2 - 675*p**5*q2*q3**2*q4*q5*q6*q7*q8 + 5400*p**5*q2*q3**2*q4*q5*q6*q7 - 675*p**5*q2*q3*q4**3*q5*q6*q7 - 675*p**5*q2*q3*q4**2*q5**2*q6*q7 - 675*p**5*q2*q3*q4**2*q5*q6**2*q7 - 675*p**5*q2*q3*q4**2*q5*q6*q7**2 - 675*p**5*q2*q3*q4**2*q5*q6*q7*q8 + 5400*p**5*q2*q3*q4**2*q5*q6*q7 - 675*p**5*q2*q3*q4*q5**3*q6*q7 - 675*p**5*q2*q3*q4*q5**2*q6**2*q7 - 675*p**5*q2*q3*q4*q5**2*q6*q7**2 - 675*p**5*q2*q3*q4*q5**2*q6*q7*q8 + 5400*p**5*q2*q3*q4*q5**2*q6*q7 - 675*p**5*q2*q3*q4*q5*q6**3*q7 - 675*p**5*q2*q3*q4*q5*q6**2*q7**2 - 675*p**5*q2*q3*q4*q5*q6**2*q7*q8 + 5400*p**5*q2*q3*q4*q5*q6**2*q7 - 675*p**5*q2*q3*q4*q5*q6*q7**3 - 675*p**5*q2*q3*q4*q5*q6*q7**2*q8 + 5400*p**5*q2*q3*q4*q5*q6*q7**2 - 675*p**5*q2*q3*q4*q5*q6*q7*q8**2 + 5400*p**5*q2*q3*q4*q5*q6*q7*q8 - 18900*p**5*q2*q3*q4*q5*q6*q7 + 1200*p**5*q3**3*q4*q5*q6*q7 + 1200*p**5*q3**2*q4**2*q5*q6*q7 + 1200*p**5*q3**2*q4*q5**2*q6*q7 + 1200*p**5*q3**2*q4*q5*q6**2*q7 + 1200*p**5*q3**2*q4*q5*q6*q7**2 + 1200*p**5*q3**2*q4*q5*q6*q7*q8 - 8400*p**5*q3**2*q4*q5*q6*q7 + 1200*p**5*q3*q4**3*q5*q6*q7 + 1200*p**5*q3*q4**2*q5**2*q6*q7 + 1200*p**5*q3*q4**2*q5*q6**2*q7 + 1200*p**5*q3*q4**2*q5*q6*q7**2 + 1200*p**5*q3*q4**2*q5*q6*q7*q8 - 8400*p**5*q3*q4**2*q5*q6*q7 + 1200*p**5*q3*q4*q5**3*q6*q7 + 1200*p**5*q3*q4*q5**2*q6**2*q7 + 1200*p**5*q3*q4*q5**2*q6*q7**2 + 1200*p**5*q3*q4*q5**2*q6*q7*q8 - 8400*p**5*q3*q4*q5**2*q6*q7 + 1200*p**5*q3*q4*q5*q6**3*q7 + 1200*p**5*q3*q4*q5*q6**2*q7**2 + 1200*p**5*q3*q4*q5*q6**2*q7*q8 - 8400*p**5*q3*q4*q5*q6**2*q7 + 1200*p**5*q3*q4*q5*q6*q7**3 + 1200*p**5*q3*q4*q5*q6*q7**2*q8 - 8400*p**5*q3*q4*q5*q6*q7**2 + 1200*p**5*q3*q4*q5*q6*q7*q8**2 - 8400*p**5*q3*q4*q5*q6*q7*q8 + 25200*p**5*q3*q4*q5*q6*q7 - 1260*p**5*q4**3*q5*q6*q7 - 1260*p**5*q4**2*q5**2*q6*q7 - 1260*p**5*q4**2*q5*q6**2*q7 - 1260*p**5*q4**2*q5*q6*q7**2 - 1260*p**5*q4**2*q5*q6*q7*q8 + 7560*p**5*q4**2*q5*q6*q7 - 1260*p**5*q4*q5**3*q6*q7 - 1260*p**5*q4*q5**2*q6**2*q7 - 1260*p**5*q4*q5**2*q6*q7**2 - 1260*p**5*q4*q5**2*q6*q7*q8 + 7560*p**5*q4*q5**2*q6*q7 - 1260*p**5*q4*q5*q6**3*q7 - 1260*p**5*q4*q5*q6**2*q7**2 - 1260*p**5*q4*q5*q6**2*q7*q8 + 7560*p**5*q4*q5*q6**2*q7 - 1260*p**5*q4*q5*q6*q7**3 - 1260*p**5*q4*q5*q6*q7**2*q8 + 7560*p**5*q4*q5*q6*q7**2 - 1260*p**5*q4*q5*q6*q7*q8**2 + 7560*p**5*q4*q5*q6*q7*q8 - 18900*p**5*q4*q5*q6*q7 + 756*p**5*q5**3*q6*q7 + 756*p**5*q5**2*q6**2*q7 + 756*p**5*q5**2*q6*q7**2 + 756*p**5*q5**2*q6*q7*q8 - 3780*p**5*q5**2*q6*q7 + 756*p**5*q5*q6**3*q7 + 756*p**5*q5*q6**2*q7**2 + 756*p**5*q5*q6**2*q7*q8 - 3780*p**5*q5*q6**2*q7 + 756*p**5*q5*q6*q7**3 + 756*p**5*q5*q6*q7**2*q8 - 3780*p**5*q5*q6*q7**2 + 756*p**5*q5*q6*q7*q8**2 - 3780*p**5*q5*q6*q7*q8 + 7560*p**5*q5*q6*q7 - 210*p**5*q6**3*q7 - 210*p**5*q6**2*q7**2 - 210*p**5*q6**2*q7*q8 + 840*p**5*q6**2*q7 - 210*p**5*q6*q7**3 - 210*p**5*q6*q7**2*q8 + 840*p**5*q6*q7**2 - 210*p**5*q6*q7*q8**2 + 840*p**5*q6*q7*q8 - 1260*p**5*q6*q7 - 350*p**4*q1**3*q2*q3*q4*q5*q6*q7 - 350*p**4*q1**2*q2**2*q3*q4*q5*q6*q7 - 350*p**4*q1**2*q2*q3**2*q4*q5*q6*q7 - 350*p**4*q1**2*q2*q3*q4**2*q5*q6*q7 - 350*p**4*q1**2*q2*q3*q4*q5**2*q6*q7 - 350*p**4*q1**2*q2*q3*q4*q5*q6**2*q7 - 350*p**4*q1**2*q2*q3*q4*q5*q6*q7**2 - 350*p**4*q1**2*q2*q3*q4*q5*q6*q7*q8 + 3150*p**4*q1**2*q2*q3*q4*q5*q6*q7 - 350*p**4*q1*q2**3*q3*q4*q5*q6*q7 - 350*p**4*q1*q2**2*q3**2*q4*q5*q6*q7 - 350*p**4*q1*q2**2*q3*q4**2*q5*q6*q7 - 350*p**4*q1*q2**2*q3*q4*q5**2*q6*q7 - 350*p**4*q1*q2**2*q3*q4*q5*q6**2*q7 - 350*p**4*q1*q2**2*q3*q4*q5*q6*q7**2 - 350*p**4*q1*q2**2*q3*q4*q5*q6*q7*q8 + 3150*p**4*q1*q2**2*q3*q4*q5*q6*q7 - 350*p**4*q1*q2*q3**3*q4*q5*q6*q7 - 350*p**4*q1*q2*q3**2*q4**2*q5*q6*q7 - 350*p**4*q1*q2*q3**2*q4*q5**2*q6*q7 - 350*p**4*q1*q2*q3**2*q4*q5*q6**2*q7 - 350*p**4*q1*q2*q3**2*q4*q5*q6*q7**2 - 350*p**4*q1*q2*q3**2*q4*q5*q6*q7*q8 + 3150*p**4*q1*q2*q3**2*q4*q5*q6*q7 - 350*p**4*q1*q2*q3*q4**3*q5*q6*q7 - 350*p**4*q1*q2*q3*q4**2*q5**2*q6*q7 - 350*p**4*q1*q2*q3*q4**2*q5*q6**2*q7 - 350*p**4*q1*q2*q3*q4**2*q5*q6*q7**2 - 350*p**4*q1*q2*q3*q4**2*q5*q6*q7*q8 + 3150*p**4*q1*q2*q3*q4**2*q5*q6*q7 - 350*p**4*q1*q2*q3*q4*q5**3*q6*q7 - 350*p**4*q1*q2*q3*q4*q5**2*q6**2*q7 - 350*p**4*q1*q2*q3*q4*q5**2*q6*q7**2 - 350*p**4*q1*q2*q3*q4*q5**2*q6*q7*q8 + 3150*p**4*q1*q2*q3*q4*q5**2*q6*q7 - 350*p**4*q1*q2*q3*q4*q5*q6**3*q7 - 350*p**4*q1*q2*q3*q4*q5*q6**2*q7**2 - 350*p**4*q1*q2*q3*q4*q5*q6**2*q7*q8 + 3150*p**4*q1*q2*q3*q4*q5*q6**2*q7 - 350*p**4*q1*q2*q3*q4*q5*q6*q7**3 - 350*p**4*q1*q2*q3*q4*q5*q6*q7**2*q8 + 3150*p**4*q1*q2*q3*q4*q5*q6*q7**2 - 350*p**4*q1*q2*q3*q4*q5*q6*q7*q8**2 + 3150*p**4*q1*q2*q3*q4*q5*q6*q7*q8 - 12600*p**4*q1*q2*q3*q4*q5*q6*q7 + 900*p**4*q2**3*q3*q4*q5*q6*q7 + 900*p**4*q2**2*q3**2*q4*q5*q6*q7 + 900*p**4*q2**2*q3*q4**2*q5*q6*q7 + 900*p**4*q2**2*q3*q4*q5**2*q6*q7 + 900*p**4*q2**2*q3*q4*q5*q6**2*q7 + 900*p**4*q2**2*q3*q4*q5*q6*q7**2 + 900*p**4*q2**2*q3*q4*q5*q6*q7*q8 - 7200*p**4*q2**2*q3*q4*q5*q6*q7 + 900*p**4*q2*q3**3*q4*q5*q6*q7 + 900*p**4*q2*q3**2*q4**2*q5*q6*q7 + 900*p**4*q2*q3**2*q4*q5**2*q6*q7 + 900*p**4*q2*q3**2*q4*q5*q6**2*q7 + 900*p**4*q2*q3**2*q4*q5*q6*q7**2 + 900*p**4*q2*q3**2*q4*q5*q6*q7*q8 - 7200*p**4*q2*q3**2*q4*q5*q6*q7 + 900*p**4*q2*q3*q4**3*q5*q6*q7 + 900*p**4*q2*q3*q4**2*q5**2*q6*q7 + 900*p**4*q2*q3*q4**2*q5*q6**2*q7 + 900*p**4*q2*q3*q4**2*q5*q6*q7**2 + 900*p**4*q2*q3*q4**2*q5*q6*q7*q8 - 7200*p**4*q2*q3*q4**2*q5*q6*q7 + 900*p**4*q2*q3*q4*q5**3*q6*q7 + 900*p**4*q2*q3*q4*q5**2*q6**2*q7 + 900*p**4*q2*q3*q4*q5**2*q6*q7**2 + 900*p**4*q2*q3*q4*q5**2*q6*q7*q8 - 7200*p**4*q2*q3*q4*q5**2*q6*q7 + 900*p**4*q2*q3*q4*q5*q6**3*q7 + 900*p**4*q2*q3*q4*q5*q6**2*q7**2 + 900*p**4*q2*q3*q4*q5*q6**2*q7*q8 - 7200*p**4*q2*q3*q4*q5*q6**2*q7 + 900*p**4*q2*q3*q4*q5*q6*q7**3 + 900*p**4*q2*q3*q4*q5*q6*q7**2*q8 - 7200*p**4*q2*q3*q4*q5*q6*q7**2 + 900*p**4*q2*q3*q4*q5*q6*q7*q8**2 - 7200*p**4*q2*q3*q4*q5*q6*q7*q8 + 25200*p**4*q2*q3*q4*q5*q6*q7 - 1200*p**4*q3**3*q4*q5*q6*q7 - 1200*p**4*q3**2*q4**2*q5*q6*q7 - 1200*p**4*q3**2*q4*q5**2*q6*q7 - 1200*p**4*q3**2*q4*q5*q6**2*q7 - 1200*p**4*q3**2*q4*q5*q6*q7**2 - 1200*p**4*q3**2*q4*q5*q6*q7*q8 + 8400*p**4*q3**2*q4*q5*q6*q7 - 1200*p**4*q3*q4**3*q5*q6*q7 - 1200*p**4*q3*q4**2*q5**2*q6*q7 - 1200*p**4*q3*q4**2*q5*q6**2*q7 - 1200*p**4*q3*q4**2*q5*q6*q7**2 - 1200*p**4*q3*q4**2*q5*q6*q7*q8 + 8400*p**4*q3*q4**2*q5*q6*q7 - 1200*p**4*q3*q4*q5**3*q6*q7 - 1200*p**4*q3*q4*q5**2*q6**2*q7 - 1200*p**4*q3*q4*q5**2*q6*q7**2 - 1200*p**4*q3*q4*q5**2*q6*q7*q8 + 8400*p**4*q3*q4*q5**2*q6*q7 - 1200*p**4*q3*q4*q5*q6**3*q7 - 1200*p**4*q3*q4*q5*q6**2*q7**2 - 1200*p**4*q3*q4*q5*q6**2*q7*q8 + 8400*p**4*q3*q4*q5*q6**2*q7 - 1200*p**4*q3*q4*q5*q6*q7**3 - 1200*p**4*q3*q4*q5*q6*q7**2*q8 + 8400*p**4*q3*q4*q5*q6*q7**2 - 1200*p**4*q3*q4*q5*q6*q7*q8**2 + 8400*p**4*q3*q4*q5*q6*q7*q8 - 25200*p**4*q3*q4*q5*q6*q7 + 840*p**4*q4**3*q5*q6*q7 + 840*p**4*q4**2*q5**2*q6*q7 + 840*p**4*q4**2*q5*q6**2*q7 + 840*p**4*q4**2*q5*q6*q7**2 + 840*p**4*q4**2*q5*q6*q7*q8 - 5040*p**4*q4**2*q5*q6*q7 + 840*p**4*q4*q5**3*q6*q7 + 840*p**4*q4*q5**2*q6**2*q7 + 840*p**4*q4*q5**2*q6*q7**2 + 840*p**4*q4*q5**2*q6*q7*q8 - 5040*p**4*q4*q5**2*q6*q7 + 840*p**4*q4*q5*q6**3*q7 + 840*p**4*q4*q5*q6**2*q7**2 + 840*p**4*q4*q5*q6**2*q7*q8 - 5040*p**4*q4*q5*q6**2*q7 + 840*p**4*q4*q5*q6*q7**3 + 840*p**4*q4*q5*q6*q7**2*q8 - 5040*p**4*q4*q5*q6*q7**2 + 840*p**4*q4*q5*q6*q7*q8**2 - 5040*p**4*q4*q5*q6*q7*q8 + 12600*p**4*q4*q5*q6*q7 - 252*p**4*q5**3*q6*q7 - 252*p**4*q5**2*q6**2*q7 - 252*p**4*q5**2*q6*q7**2 - 252*p**4*q5**2*q6*q7*q8 + 1260*p**4*q5**2*q6*q7 - 252*p**4*q5*q6**3*q7 - 252*p**4*q5*q6**2*q7**2 - 252*p**4*q5*q6**2*q7*q8 + 1260*p**4*q5*q6**2*q7 - 252*p**4*q5*q6*q7**3 - 252*p**4*q5*q6*q7**2*q8 + 1260*p**4*q5*q6*q7**2 - 252*p**4*q5*q6*q7*q8**2 + 1260*p**4*q5*q6*q7*q8 - 2520*p**4*q5*q6*q7 + 350*p**3*q1**3*q2*q3*q4*q5*q6*q7 + 350*p**3*q1**2*q2**2*q3*q4*q5*q6*q7 + 350*p**3*q1**2*q2*q3**2*q4*q5*q6*q7 + 350*p**3*q1**2*q2*q3*q4**2*q5*q6*q7 + 350*p**3*q1**2*q2*q3*q4*q5**2*q6*q7 + 350*p**3*q1**2*q2*q3*q4*q5*q6**2*q7 + 350*p**3*q1**2*q2*q3*q4*q5*q6*q7**2 + 350*p**3*q1**2*q2*q3*q4*q5*q6*q7*q8 - 3150*p**3*q1**2*q2*q3*q4*q5*q6*q7 + 350*p**3*q1*q2**3*q3*q4*q5*q6*q7 + 350*p**3*q1*q2**2*q3**2*q4*q5*q6*q7 + 350*p**3*q1*q2**2*q3*q4**2*q5*q6*q7 + 350*p**3*q1*q2**2*q3*q4*q5**2*q6*q7 + 350*p**3*q1*q2**2*q3*q4*q5*q6**2*q7 + 350*p**3*q1*q2**2*q3*q4*q5*q6*q7**2 + 350*p**3*q1*q2**2*q3*q4*q5*q6*q7*q8 - 3150*p**3*q1*q2**2*q3*q4*q5*q6*q7 + 350*p**3*q1*q2*q3**3*q4*q5*q6*q7 + 350*p**3*q1*q2*q3**2*q4**2*q5*q6*q7 + 350*p**3*q1*q2*q3**2*q4*q5**2*q6*q7 + 350*p**3*q1*q2*q3**2*q4*q5*q6**2*q7 + 350*p**3*q1*q2*q3**2*q4*q5*q6*q7**2 + 350*p**3*q1*q2*q3**2*q4*q5*q6*q7*q8 - 3150*p**3*q1*q2*q3**2*q4*q5*q6*q7 + 350*p**3*q1*q2*q3*q4**3*q5*q6*q7 + 350*p**3*q1*q2*q3*q4**2*q5**2*q6*q7 + 350*p**3*q1*q2*q3*q4**2*q5*q6**2*q7 + 350*p**3*q1*q2*q3*q4**2*q5*q6*q7**2 + 350*p**3*q1*q2*q3*q4**2*q5*q6*q7*q8 - 3150*p**3*q1*q2*q3*q4**2*q5*q6*q7 + 350*p**3*q1*q2*q3*q4*q5**3*q6*q7 + 350*p**3*q1*q2*q3*q4*q5**2*q6**2*q7 + 350*p**3*q1*q2*q3*q4*q5**2*q6*q7**2 + 350*p**3*q1*q2*q3*q4*q5**2*q6*q7*q8 - 3150*p**3*q1*q2*q3*q4*q5**2*q6*q7 + 350*p**3*q1*q2*q3*q4*q5*q6**3*q7 + 350*p**3*q1*q2*q3*q4*q5*q6**2*q7**2 + 350*p**3*q1*q2*q3*q4*q5*q6**2*q7*q8 - 3150*p**3*q1*q2*q3*q4*q5*q6**2*q7 + 350*p**3*q1*q2*q3*q4*q5*q6*q7**3 + 350*p**3*q1*q2*q3*q4*q5*q6*q7**2*q8 - 3150*p**3*q1*q2*q3*q4*q5*q6*q7**2 + 350*p**3*q1*q2*q3*q4*q5*q6*q7*q8**2 - 3150*p**3*q1*q2*q3*q4*q5*q6*q7*q8 + 12600*p**3*q1*q2*q3*q4*q5*q6*q7 - 675*p**3*q2**3*q3*q4*q5*q6*q7 - 675*p**3*q2**2*q3**2*q4*q5*q6*q7 - 675*p**3*q2**2*q3*q4**2*q5*q6*q7 - 675*p**3*q2**2*q3*q4*q5**2*q6*q7 - 675*p**3*q2**2*q3*q4*q5*q6**2*q7 - 675*p**3*q2**2*q3*q4*q5*q6*q7**2 - 675*p**3*q2**2*q3*q4*q5*q6*q7*q8 + 5400*p**3*q2**2*q3*q4*q5*q6*q7 - 675*p**3*q2*q3**3*q4*q5*q6*q7 - 675*p**3*q2*q3**2*q4**2*q5*q6*q7 - 675*p**3*q2*q3**2*q4*q5**2*q6*q7 - 675*p**3*q2*q3**2*q4*q5*q6**2*q7 - 675*p**3*q2*q3**2*q4*q5*q6*q7**2 - 675*p**3*q2*q3**2*q4*q5*q6*q7*q8 + 5400*p**3*q2*q3**2*q4*q5*q6*q7 - 675*p**3*q2*q3*q4**3*q5*q6*q7 - 675*p**3*q2*q3*q4**2*q5**2*q6*q7 - 675*p**3*q2*q3*q4**2*q5*q6**2*q7 - 675*p**3*q2*q3*q4**2*q5*q6*q7**2 - 675*p**3*q2*q3*q4**2*q5*q6*q7*q8 + 5400*p**3*q2*q3*q4**2*q5*q6*q7 - 675*p**3*q2*q3*q4*q5**3*q6*q7 - 675*p**3*q2*q3*q4*q5**2*q6**2*q7 - 675*p**3*q2*q3*q4*q5**2*q6*q7**2 - 675*p**3*q2*q3*q4*q5**2*q6*q7*q8 + 5400*p**3*q2*q3*q4*q5**2*q6*q7 - 675*p**3*q2*q3*q4*q5*q6**3*q7 - 675*p**3*q2*q3*q4*q5*q6**2*q7**2 - 675*p**3*q2*q3*q4*q5*q6**2*q7*q8 + 5400*p**3*q2*q3*q4*q5*q6**2*q7 - 675*p**3*q2*q3*q4*q5*q6*q7**3 - 675*p**3*q2*q3*q4*q5*q6*q7**2*q8 + 5400*p**3*q2*q3*q4*q5*q6*q7**2 - 675*p**3*q2*q3*q4*q5*q6*q7*q8**2 + 5400*p**3*q2*q3*q4*q5*q6*q7*q8 - 18900*p**3*q2*q3*q4*q5*q6*q7 + 600*p**3*q3**3*q4*q5*q6*q7 + 600*p**3*q3**2*q4**2*q5*q6*q7 + 600*p**3*q3**2*q4*q5**2*q6*q7 + 600*p**3*q3**2*q4*q5*q6**2*q7 + 600*p**3*q3**2*q4*q5*q6*q7**2 + 600*p**3*q3**2*q4*q5*q6*q7*q8 - 4200*p**3*q3**2*q4*q5*q6*q7 + 600*p**3*q3*q4**3*q5*q6*q7 + 600*p**3*q3*q4**2*q5**2*q6*q7 + 600*p**3*q3*q4**2*q5*q6**2*q7 + 600*p**3*q3*q4**2*q5*q6*q7**2 + 600*p**3*q3*q4**2*q5*q6*q7*q8 - 4200*p**3*q3*q4**2*q5*q6*q7 + 600*p**3*q3*q4*q5**3*q6*q7 + 600*p**3*q3*q4*q5**2*q6**2*q7 + 600*p**3*q3*q4*q5**2*q6*q7**2 + 600*p**3*q3*q4*q5**2*q6*q7*q8 - 4200*p**3*q3*q4*q5**2*q6*q7 + 600*p**3*q3*q4*q5*q6**3*q7 + 600*p**3*q3*q4*q5*q6**2*q7**2 + 600*p**3*q3*q4*q5*q6**2*q7*q8 - 4200*p**3*q3*q4*q5*q6**2*q7 + 600*p**3*q3*q4*q5*q6*q7**3 + 600*p**3*q3*q4*q5*q6*q7**2*q8 - 4200*p**3*q3*q4*q5*q6*q7**2 + 600*p**3*q3*q4*q5*q6*q7*q8**2 - 4200*p**3*q3*q4*q5*q6*q7*q8 + 12600*p**3*q3*q4*q5*q6*q7 - 210*p**3*q4**3*q5*q6*q7 - 210*p**3*q4**2*q5**2*q6*q7 - 210*p**3*q4**2*q5*q6**2*q7 - 210*p**3*q4**2*q5*q6*q7**2 - 210*p**3*q4**2*q5*q6*q7*q8 + 1260*p**3*q4**2*q5*q6*q7 - 210*p**3*q4*q5**3*q6*q7 - 210*p**3*q4*q5**2*q6**2*q7 - 210*p**3*q4*q5**2*q6*q7**2 - 210*p**3*q4*q5**2*q6*q7*q8 + 1260*p**3*q4*q5**2*q6*q7 - 210*p**3*q4*q5*q6**3*q7 - 210*p**3*q4*q5*q6**2*q7**2 - 210*p**3*q4*q5*q6**2*q7*q8 + 1260*p**3*q4*q5*q6**2*q7 - 210*p**3*q4*q5*q6*q7**3 - 210*p**3*q4*q5*q6*q7**2*q8 + 1260*p**3*q4*q5*q6*q7**2 - 210*p**3*q4*q5*q6*q7*q8**2 + 1260*p**3*q4*q5*q6*q7*q8 - 3150*p**3*q4*q5*q6*q7 - 210*p**2*q1**3*q2*q3*q4*q5*q6*q7 - 210*p**2*q1**2*q2**2*q3*q4*q5*q6*q7 - 210*p**2*q1**2*q2*q3**2*q4*q5*q6*q7 - 210*p**2*q1**2*q2*q3*q4**2*q5*q6*q7 - 210*p**2*q1**2*q2*q3*q4*q5**2*q6*q7 - 210*p**2*q1**2*q2*q3*q4*q5*q6**2*q7 - 210*p**2*q1**2*q2*q3*q4*q5*q6*q7**2 - 210*p**2*q1**2*q2*q3*q4*q5*q6*q7*q8 + 1890*p**2*q1**2*q2*q3*q4*q5*q6*q7 - 210*p**2*q1*q2**3*q3*q4*q5*q6*q7 - 210*p**2*q1*q2**2*q3**2*q4*q5*q6*q7 - 210*p**2*q1*q2**2*q3*q4**2*q5*q6*q7 - 210*p**2*q1*q2**2*q3*q4*q5**2*q6*q7 - 210*p**2*q1*q2**2*q3*q4*q5*q6**2*q7 - 210*p**2*q1*q2**2*q3*q4*q5*q6*q7**2 - 210*p**2*q1*q2**2*q3*q4*q5*q6*q7*q8 + 1890*p**2*q1*q2**2*q3*q4*q5*q6*q7 - 210*p**2*q1*q2*q3**3*q4*q5*q6*q7 - 210*p**2*q1*q2*q3**2*q4**2*q5*q6*q7 - 210*p**2*q1*q2*q3**2*q4*q5**2*q6*q7 - 210*p**2*q1*q2*q3**2*q4*q5*q6**2*q7 - 210*p**2*q1*q2*q3**2*q4*q5*q6*q7**2 - 210*p**2*q1*q2*q3**2*q4*q5*q6*q7*q8 + 1890*p**2*q1*q2*q3**2*q4*q5*q6*q7 - 210*p**2*q1*q2*q3*q4**3*q5*q6*q7 - 210*p**2*q1*q2*q3*q4**2*q5**2*q6*q7 - 210*p**2*q1*q2*q3*q4**2*q5*q6**2*q7 - 210*p**2*q1*q2*q3*q4**2*q5*q6*q7**2 - 210*p**2*q1*q2*q3*q4**2*q5*q6*q7*q8 + 1890*p**2*q1*q2*q3*q4**2*q5*q6*q7 - 210*p**2*q1*q2*q3*q4*q5**3*q6*q7 - 210*p**2*q1*q2*q3*q4*q5**2*q6**2*q7 - 210*p**2*q1*q2*q3*q4*q5**2*q6*q7**2 - 210*p**2*q1*q2*q3*q4*q5**2*q6*q7*q8 + 1890*p**2*q1*q2*q3*q4*q5**2*q6*q7 - 210*p**2*q1*q2*q3*q4*q5*q6**3*q7 - 210*p**2*q1*q2*q3*q4*q5*q6**2*q7**2 - 210*p**2*q1*q2*q3*q4*q5*q6**2*q7*q8 + 1890*p**2*q1*q2*q3*q4*q5*q6**2*q7 - 210*p**2*q1*q2*q3*q4*q5*q6*q7**3 - 210*p**2*q1*q2*q3*q4*q5*q6*q7**2*q8 + 1890*p**2*q1*q2*q3*q4*q5*q6*q7**2 - 210*p**2*q1*q2*q3*q4*q5*q6*q7*q8**2 + 1890*p**2*q1*q2*q3*q4*q5*q6*q7*q8 - 7560*p**2*q1*q2*q3*q4*q5*q6*q7 + 270*p**2*q2**3*q3*q4*q5*q6*q7 + 270*p**2*q2**2*q3**2*q4*q5*q6*q7 + 270*p**2*q2**2*q3*q4**2*q5*q6*q7 + 270*p**2*q2**2*q3*q4*q5**2*q6*q7 + 270*p**2*q2**2*q3*q4*q5*q6**2*q7 + 270*p**2*q2**2*q3*q4*q5*q6*q7**2 + 270*p**2*q2**2*q3*q4*q5*q6*q7*q8 - 2160*p**2*q2**2*q3*q4*q5*q6*q7 + 270*p**2*q2*q3**3*q4*q5*q6*q7 + 270*p**2*q2*q3**2*q4**2*q5*q6*q7 + 270*p**2*q2*q3**2*q4*q5**2*q6*q7 + 270*p**2*q2*q3**2*q4*q5*q6**2*q7 + 270*p**2*q2*q3**2*q4*q5*q6*q7**2 + 270*p**2*q2*q3**2*q4*q5*q6*q7*q8 - 2160*p**2*q2*q3**2*q4*q5*q6*q7 + 270*p**2*q2*q3*q4**3*q5*q6*q7 + 270*p**2*q2*q3*q4**2*q5**2*q6*q7 + 270*p**2*q2*q3*q4**2*q5*q6**2*q7 + 270*p**2*q2*q3*q4**2*q5*q6*q7**2 + 270*p**2*q2*q3*q4**2*q5*q6*q7*q8 - 2160*p**2*q2*q3*q4**2*q5*q6*q7 + 270*p**2*q2*q3*q4*q5**3*q6*q7 + 270*p**2*q2*q3*q4*q5**2*q6**2*q7 + 270*p**2*q2*q3*q4*q5**2*q6*q7**2 + 270*p**2*q2*q3*q4*q5**2*q6*q7*q8 - 2160*p**2*q2*q3*q4*q5**2*q6*q7 + 270*p**2*q2*q3*q4*q5*q6**3*q7 + 270*p**2*q2*q3*q4*q5*q6**2*q7**2 + 270*p**2*q2*q3*q4*q5*q6**2*q7*q8 - 2160*p**2*q2*q3*q4*q5*q6**2*q7 + 270*p**2*q2*q3*q4*q5*q6*q7**3 + 270*p**2*q2*q3*q4*q5*q6*q7**2*q8 - 2160*p**2*q2*q3*q4*q5*q6*q7**2 + 270*p**2*q2*q3*q4*q5*q6*q7*q8**2 - 2160*p**2*q2*q3*q4*q5*q6*q7*q8 + 7560*p**2*q2*q3*q4*q5*q6*q7 - 120*p**2*q3**3*q4*q5*q6*q7 - 120*p**2*q3**2*q4**2*q5*q6*q7 - 120*p**2*q3**2*q4*q5**2*q6*q7 - 120*p**2*q3**2*q4*q5*q6**2*q7 - 120*p**2*q3**2*q4*q5*q6*q7**2 - 120*p**2*q3**2*q4*q5*q6*q7*q8 + 840*p**2*q3**2*q4*q5*q6*q7 - 120*p**2*q3*q4**3*q5*q6*q7 - 120*p**2*q3*q4**2*q5**2*q6*q7 - 120*p**2*q3*q4**2*q5*q6**2*q7 - 120*p**2*q3*q4**2*q5*q6*q7**2 - 120*p**2*q3*q4**2*q5*q6*q7*q8 + 840*p**2*q3*q4**2*q5*q6*q7 - 120*p**2*q3*q4*q5**3*q6*q7 - 120*p**2*q3*q4*q5**2*q6**2*q7 - 120*p**2*q3*q4*q5**2*q6*q7**2 - 120*p**2*q3*q4*q5**2*q6*q7*q8 + 840*p**2*q3*q4*q5**2*q6*q7 - 120*p**2*q3*q4*q5*q6**3*q7 - 120*p**2*q3*q4*q5*q6**2*q7**2 - 120*p**2*q3*q4*q5*q6**2*q7*q8 + 840*p**2*q3*q4*q5*q6**2*q7 - 120*p**2*q3*q4*q5*q6*q7**3 - 120*p**2*q3*q4*q5*q6*q7**2*q8 + 840*p**2*q3*q4*q5*q6*q7**2 - 120*p**2*q3*q4*q5*q6*q7*q8**2 + 840*p**2*q3*q4*q5*q6*q7*q8 - 2520*p**2*q3*q4*q5*q6*q7 + 70*p*q1**3*q2*q3*q4*q5*q6*q7 + 70*p*q1**2*q2**2*q3*q4*q5*q6*q7 + 70*p*q1**2*q2*q3**2*q4*q5*q6*q7 + 70*p*q1**2*q2*q3*q4**2*q5*q6*q7 + 70*p*q1**2*q2*q3*q4*q5**2*q6*q7 + 70*p*q1**2*q2*q3*q4*q5*q6**2*q7 + 70*p*q1**2*q2*q3*q4*q5*q6*q7**2 + 70*p*q1**2*q2*q3*q4*q5*q6*q7*q8 - 630*p*q1**2*q2*q3*q4*q5*q6*q7 + 70*p*q1*q2**3*q3*q4*q5*q6*q7 + 70*p*q1*q2**2*q3**2*q4*q5*q6*q7 + 70*p*q1*q2**2*q3*q4**2*q5*q6*q7 + 70*p*q1*q2**2*q3*q4*q5**2*q6*q7 + 70*p*q1*q2**2*q3*q4*q5*q6**2*q7 + 70*p*q1*q2**2*q3*q4*q5*q6*q7**2 + 70*p*q1*q2**2*q3*q4*q5*q6*q7*q8 - 630*p*q1*q2**2*q3*q4*q5*q6*q7 + 70*p*q1*q2*q3**3*q4*q5*q6*q7 + 70*p*q1*q2*q3**2*q4**2*q5*q6*q7 + 70*p*q1*q2*q3**2*q4*q5**2*q6*q7 + 70*p*q1*q2*q3**2*q4*q5*q6**2*q7 + 70*p*q1*q2*q3**2*q4*q5*q6*q7**2 + 70*p*q1*q2*q3**2*q4*q5*q6*q7*q8 - 630*p*q1*q2*q3**2*q4*q5*q6*q7 + 70*p*q1*q2*q3*q4**3*q5*q6*q7 + 70*p*q1*q2*q3*q4**2*q5**2*q6*q7 + 70*p*q1*q2*q3*q4**2*q5*q6**2*q7 + 70*p*q1*q2*q3*q4**2*q5*q6*q7**2 + 70*p*q1*q2*q3*q4**2*q5*q6*q7*q8 - 630*p*q1*q2*q3*q4**2*q5*q6*q7 + 70*p*q1*q2*q3*q4*q5**3*q6*q7 + 70*p*q1*q2*q3*q4*q5**2*q6**2*q7 + 70*p*q1*q2*q3*q4*q5**2*q6*q7**2 + 70*p*q1*q2*q3*q4*q5**2*q6*q7*q8 - 630*p*q1*q2*q3*q4*q5**2*q6*q7 + 70*p*q1*q2*q3*q4*q5*q6**3*q7 + 70*p*q1*q2*q3*q4*q5*q6**2*q7**2 + 70*p*q1*q2*q3*q4*q5*q6**2*q7*q8 - 630*p*q1*q2*q3*q4*q5*q6**2*q7 + 70*p*q1*q2*q3*q4*q5*q6*q7**3 + 70*p*q1*q2*q3*q4*q5*q6*q7**2*q8 - 630*p*q1*q2*q3*q4*q5*q6*q7**2 + 70*p*q1*q2*q3*q4*q5*q6*q7*q8**2 - 630*p*q1*q2*q3*q4*q5*q6*q7*q8 + 2520*p*q1*q2*q3*q4*q5*q6*q7 - 45*p*q2**3*q3*q4*q5*q6*q7 - 45*p*q2**2*q3**2*q4*q5*q6*q7 - 45*p*q2**2*q3*q4**2*q5*q6*q7 - 45*p*q2**2*q3*q4*q5**2*q6*q7 - 45*p*q2**2*q3*q4*q5*q6**2*q7 - 45*p*q2**2*q3*q4*q5*q6*q7**2 - 45*p*q2**2*q3*q4*q5*q6*q7*q8 + 360*p*q2**2*q3*q4*q5*q6*q7 - 45*p*q2*q3**3*q4*q5*q6*q7 - 45*p*q2*q3**2*q4**2*q5*q6*q7 - 45*p*q2*q3**2*q4*q5**2*q6*q7 - 45*p*q2*q3**2*q4*q5*q6**2*q7 - 45*p*q2*q3**2*q4*q5*q6*q7**2 - 45*p*q2*q3**2*q4*q5*q6*q7*q8 + 360*p*q2*q3**2*q4*q5*q6*q7 - 45*p*q2*q3*q4**3*q5*q6*q7 - 45*p*q2*q3*q4**2*q5**2*q6*q7 - 45*p*q2*q3*q4**2*q5*q6**2*q7 - 45*p*q2*q3*q4**2*q5*q6*q7**2 - 45*p*q2*q3*q4**2*q5*q6*q7*q8 + 360*p*q2*q3*q4**2*q5*q6*q7 - 45*p*q2*q3*q4*q5**3*q6*q7 - 45*p*q2*q3*q4*q5**2*q6**2*q7 - 45*p*q2*q3*q4*q5**2*q6*q7**2 - 45*p*q2*q3*q4*q5**2*q6*q7*q8 + 360*p*q2*q3*q4*q5**2*q6*q7 - 45*p*q2*q3*q4*q5*q6**3*q7 - 45*p*q2*q3*q4*q5*q6**2*q7**2 - 45*p*q2*q3*q4*q5*q6**2*q7*q8 + 360*p*q2*q3*q4*q5*q6**2*q7 - 45*p*q2*q3*q4*q5*q6*q7**3 - 45*p*q2*q3*q4*q5*q6*q7**2*q8 + 360*p*q2*q3*q4*q5*q6*q7**2 - 45*p*q2*q3*q4*q5*q6*q7*q8**2 + 360*p*q2*q3*q4*q5*q6*q7*q8 - 1260*p*q2*q3*q4*q5*q6*q7 - 10*q1**3*q2*q3*q4*q5*q6*q7 - 10*q1**2*q2**2*q3*q4*q5*q6*q7 - 10*q1**2*q2*q3**2*q4*q5*q6*q7 - 10*q1**2*q2*q3*q4**2*q5*q6*q7 - 10*q1**2*q2*q3*q4*q5**2*q6*q7 - 10*q1**2*q2*q3*q4*q5*q6**2*q7 - 10*q1**2*q2*q3*q4*q5*q6*q7**2 - 10*q1**2*q2*q3*q4*q5*q6*q7*q8 + 90*q1**2*q2*q3*q4*q5*q6*q7 - 10*q1*q2**3*q3*q4*q5*q6*q7 - 10*q1*q2**2*q3**2*q4*q5*q6*q7 - 10*q1*q2**2*q3*q4**2*q5*q6*q7 - 10*q1*q2**2*q3*q4*q5**2*q6*q7 - 10*q1*q2**2*q3*q4*q5*q6**2*q7 - 10*q1*q2**2*q3*q4*q5*q6*q7**2 - 10*q1*q2**2*q3*q4*q5*q6*q7*q8 + 90*q1*q2**2*q3*q4*q5*q6*q7 - 10*q1*q2*q3**3*q4*q5*q6*q7 - 10*q1*q2*q3**2*q4**2*q5*q6*q7 - 10*q1*q2*q3**2*q4*q5**2*q6*q7 - 10*q1*q2*q3**2*q4*q5*q6**2*q7 - 10*q1*q2*q3**2*q4*q5*q6*q7**2 - 10*q1*q2*q3**2*q4*q5*q6*q7*q8 + 90*q1*q2*q3**2*q4*q5*q6*q7 - 10*q1*q2*q3*q4**3*q5*q6*q7 - 10*q1*q2*q3*q4**2*q5**2*q6*q7 - 10*q1*q2*q3*q4**2*q5*q6**2*q7 - 10*q1*q2*q3*q4**2*q5*q6*q7**2 - 10*q1*q2*q3*q4**2*q5*q6*q7*q8 + 90*q1*q2*q3*q4**2*q5*q6*q7 - 10*q1*q2*q3*q4*q5**3*q6*q7 - 10*q1*q2*q3*q4*q5**2*q6**2*q7 - 10*q1*q2*q3*q4*q5**2*q6*q7**2 - 10*q1*q2*q3*q4*q5**2*q6*q7*q8 + 90*q1*q2*q3*q4*q5**2*q6*q7 - 10*q1*q2*q3*q4*q5*q6**3*q7 - 10*q1*q2*q3*q4*q5*q6**2*q7**2 - 10*q1*q2*q3*q4*q5*q6**2*q7*q8 + 90*q1*q2*q3*q4*q5*q6**2*q7 - 10*q1*q2*q3*q4*q5*q6*q7**3 - 10*q1*q2*q3*q4*q5*q6*q7**2*q8 + 90*q1*q2*q3*q4*q5*q6*q7**2 - 10*q1*q2*q3*q4*q5*q6*q7*q8**2 + 90*q1*q2*q3*q4*q5*q6*q7*q8 - 360*q1*q2*q3*q4*q5*q6*q7)'
        f_multiparam[10][9] = 'p*(p - 1)*(10*p**8*q1**2*q2*q3*q4*q5*q6*q7*q8 + 10*p**8*q1*q2**2*q3*q4*q5*q6*q7*q8 + 10*p**8*q1*q2*q3**2*q4*q5*q6*q7*q8 + 10*p**8*q1*q2*q3*q4**2*q5*q6*q7*q8 + 10*p**8*q1*q2*q3*q4*q5**2*q6*q7*q8 + 10*p**8*q1*q2*q3*q4*q5*q6**2*q7*q8 + 10*p**8*q1*q2*q3*q4*q5*q6*q7**2*q8 + 10*p**8*q1*q2*q3*q4*q5*q6*q7*q8**2 + 10*p**8*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 90*p**8*q1*q2*q3*q4*q5*q6*q7*q8 - 45*p**8*q2**2*q3*q4*q5*q6*q7*q8 - 45*p**8*q2*q3**2*q4*q5*q6*q7*q8 - 45*p**8*q2*q3*q4**2*q5*q6*q7*q8 - 45*p**8*q2*q3*q4*q5**2*q6*q7*q8 - 45*p**8*q2*q3*q4*q5*q6**2*q7*q8 - 45*p**8*q2*q3*q4*q5*q6*q7**2*q8 - 45*p**8*q2*q3*q4*q5*q6*q7*q8**2 - 45*p**8*q2*q3*q4*q5*q6*q7*q8*q9 + 360*p**8*q2*q3*q4*q5*q6*q7*q8 + 120*p**8*q3**2*q4*q5*q6*q7*q8 + 120*p**8*q3*q4**2*q5*q6*q7*q8 + 120*p**8*q3*q4*q5**2*q6*q7*q8 + 120*p**8*q3*q4*q5*q6**2*q7*q8 + 120*p**8*q3*q4*q5*q6*q7**2*q8 + 120*p**8*q3*q4*q5*q6*q7*q8**2 + 120*p**8*q3*q4*q5*q6*q7*q8*q9 - 840*p**8*q3*q4*q5*q6*q7*q8 - 210*p**8*q4**2*q5*q6*q7*q8 - 210*p**8*q4*q5**2*q6*q7*q8 - 210*p**8*q4*q5*q6**2*q7*q8 - 210*p**8*q4*q5*q6*q7**2*q8 - 210*p**8*q4*q5*q6*q7*q8**2 - 210*p**8*q4*q5*q6*q7*q8*q9 + 1260*p**8*q4*q5*q6*q7*q8 + 252*p**8*q5**2*q6*q7*q8 + 252*p**8*q5*q6**2*q7*q8 + 252*p**8*q5*q6*q7**2*q8 + 252*p**8*q5*q6*q7*q8**2 + 252*p**8*q5*q6*q7*q8*q9 - 1260*p**8*q5*q6*q7*q8 - 210*p**8*q6**2*q7*q8 - 210*p**8*q6*q7**2*q8 - 210*p**8*q6*q7*q8**2 - 210*p**8*q6*q7*q8*q9 + 840*p**8*q6*q7*q8 + 120*p**8*q7**2*q8 + 120*p**8*q7*q8**2 + 120*p**8*q7*q8*q9 - 360*p**8*q7*q8 - 45*p**8*q8**2 - 45*p**8*q8*q9 + 90*p**8*q8 + 10*p**8*q9 - 10*p**8 - 80*p**7*q1**2*q2*q3*q4*q5*q6*q7*q8 - 80*p**7*q1*q2**2*q3*q4*q5*q6*q7*q8 - 80*p**7*q1*q2*q3**2*q4*q5*q6*q7*q8 - 80*p**7*q1*q2*q3*q4**2*q5*q6*q7*q8 - 80*p**7*q1*q2*q3*q4*q5**2*q6*q7*q8 - 80*p**7*q1*q2*q3*q4*q5*q6**2*q7*q8 - 80*p**7*q1*q2*q3*q4*q5*q6*q7**2*q8 - 80*p**7*q1*q2*q3*q4*q5*q6*q7*q8**2 - 80*p**7*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 720*p**7*q1*q2*q3*q4*q5*q6*q7*q8 + 315*p**7*q2**2*q3*q4*q5*q6*q7*q8 + 315*p**7*q2*q3**2*q4*q5*q6*q7*q8 + 315*p**7*q2*q3*q4**2*q5*q6*q7*q8 + 315*p**7*q2*q3*q4*q5**2*q6*q7*q8 + 315*p**7*q2*q3*q4*q5*q6**2*q7*q8 + 315*p**7*q2*q3*q4*q5*q6*q7**2*q8 + 315*p**7*q2*q3*q4*q5*q6*q7*q8**2 + 315*p**7*q2*q3*q4*q5*q6*q7*q8*q9 - 2520*p**7*q2*q3*q4*q5*q6*q7*q8 - 720*p**7*q3**2*q4*q5*q6*q7*q8 - 720*p**7*q3*q4**2*q5*q6*q7*q8 - 720*p**7*q3*q4*q5**2*q6*q7*q8 - 720*p**7*q3*q4*q5*q6**2*q7*q8 - 720*p**7*q3*q4*q5*q6*q7**2*q8 - 720*p**7*q3*q4*q5*q6*q7*q8**2 - 720*p**7*q3*q4*q5*q6*q7*q8*q9 + 5040*p**7*q3*q4*q5*q6*q7*q8 + 1050*p**7*q4**2*q5*q6*q7*q8 + 1050*p**7*q4*q5**2*q6*q7*q8 + 1050*p**7*q4*q5*q6**2*q7*q8 + 1050*p**7*q4*q5*q6*q7**2*q8 + 1050*p**7*q4*q5*q6*q7*q8**2 + 1050*p**7*q4*q5*q6*q7*q8*q9 - 6300*p**7*q4*q5*q6*q7*q8 - 1008*p**7*q5**2*q6*q7*q8 - 1008*p**7*q5*q6**2*q7*q8 - 1008*p**7*q5*q6*q7**2*q8 - 1008*p**7*q5*q6*q7*q8**2 - 1008*p**7*q5*q6*q7*q8*q9 + 5040*p**7*q5*q6*q7*q8 + 630*p**7*q6**2*q7*q8 + 630*p**7*q6*q7**2*q8 + 630*p**7*q6*q7*q8**2 + 630*p**7*q6*q7*q8*q9 - 2520*p**7*q6*q7*q8 - 240*p**7*q7**2*q8 - 240*p**7*q7*q8**2 - 240*p**7*q7*q8*q9 + 720*p**7*q7*q8 + 45*p**7*q8**2 + 45*p**7*q8*q9 - 90*p**7*q8 + 280*p**6*q1**2*q2*q3*q4*q5*q6*q7*q8 + 280*p**6*q1*q2**2*q3*q4*q5*q6*q7*q8 + 280*p**6*q1*q2*q3**2*q4*q5*q6*q7*q8 + 280*p**6*q1*q2*q3*q4**2*q5*q6*q7*q8 + 280*p**6*q1*q2*q3*q4*q5**2*q6*q7*q8 + 280*p**6*q1*q2*q3*q4*q5*q6**2*q7*q8 + 280*p**6*q1*q2*q3*q4*q5*q6*q7**2*q8 + 280*p**6*q1*q2*q3*q4*q5*q6*q7*q8**2 + 280*p**6*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 2520*p**6*q1*q2*q3*q4*q5*q6*q7*q8 - 945*p**6*q2**2*q3*q4*q5*q6*q7*q8 - 945*p**6*q2*q3**2*q4*q5*q6*q7*q8 - 945*p**6*q2*q3*q4**2*q5*q6*q7*q8 - 945*p**6*q2*q3*q4*q5**2*q6*q7*q8 - 945*p**6*q2*q3*q4*q5*q6**2*q7*q8 - 945*p**6*q2*q3*q4*q5*q6*q7**2*q8 - 945*p**6*q2*q3*q4*q5*q6*q7*q8**2 - 945*p**6*q2*q3*q4*q5*q6*q7*q8*q9 + 7560*p**6*q2*q3*q4*q5*q6*q7*q8 + 1800*p**6*q3**2*q4*q5*q6*q7*q8 + 1800*p**6*q3*q4**2*q5*q6*q7*q8 + 1800*p**6*q3*q4*q5**2*q6*q7*q8 + 1800*p**6*q3*q4*q5*q6**2*q7*q8 + 1800*p**6*q3*q4*q5*q6*q7**2*q8 + 1800*p**6*q3*q4*q5*q6*q7*q8**2 + 1800*p**6*q3*q4*q5*q6*q7*q8*q9 - 12600*p**6*q3*q4*q5*q6*q7*q8 - 2100*p**6*q4**2*q5*q6*q7*q8 - 2100*p**6*q4*q5**2*q6*q7*q8 - 2100*p**6*q4*q5*q6**2*q7*q8 - 2100*p**6*q4*q5*q6*q7**2*q8 - 2100*p**6*q4*q5*q6*q7*q8**2 - 2100*p**6*q4*q5*q6*q7*q8*q9 + 12600*p**6*q4*q5*q6*q7*q8 + 1512*p**6*q5**2*q6*q7*q8 + 1512*p**6*q5*q6**2*q7*q8 + 1512*p**6*q5*q6*q7**2*q8 + 1512*p**6*q5*q6*q7*q8**2 + 1512*p**6*q5*q6*q7*q8*q9 - 7560*p**6*q5*q6*q7*q8 - 630*p**6*q6**2*q7*q8 - 630*p**6*q6*q7**2*q8 - 630*p**6*q6*q7*q8**2 - 630*p**6*q6*q7*q8*q9 + 2520*p**6*q6*q7*q8 + 120*p**6*q7**2*q8 + 120*p**6*q7*q8**2 + 120*p**6*q7*q8*q9 - 360*p**6*q7*q8 - 560*p**5*q1**2*q2*q3*q4*q5*q6*q7*q8 - 560*p**5*q1*q2**2*q3*q4*q5*q6*q7*q8 - 560*p**5*q1*q2*q3**2*q4*q5*q6*q7*q8 - 560*p**5*q1*q2*q3*q4**2*q5*q6*q7*q8 - 560*p**5*q1*q2*q3*q4*q5**2*q6*q7*q8 - 560*p**5*q1*q2*q3*q4*q5*q6**2*q7*q8 - 560*p**5*q1*q2*q3*q4*q5*q6*q7**2*q8 - 560*p**5*q1*q2*q3*q4*q5*q6*q7*q8**2 - 560*p**5*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 5040*p**5*q1*q2*q3*q4*q5*q6*q7*q8 + 1575*p**5*q2**2*q3*q4*q5*q6*q7*q8 + 1575*p**5*q2*q3**2*q4*q5*q6*q7*q8 + 1575*p**5*q2*q3*q4**2*q5*q6*q7*q8 + 1575*p**5*q2*q3*q4*q5**2*q6*q7*q8 + 1575*p**5*q2*q3*q4*q5*q6**2*q7*q8 + 1575*p**5*q2*q3*q4*q5*q6*q7**2*q8 + 1575*p**5*q2*q3*q4*q5*q6*q7*q8**2 + 1575*p**5*q2*q3*q4*q5*q6*q7*q8*q9 - 12600*p**5*q2*q3*q4*q5*q6*q7*q8 - 2400*p**5*q3**2*q4*q5*q6*q7*q8 - 2400*p**5*q3*q4**2*q5*q6*q7*q8 - 2400*p**5*q3*q4*q5**2*q6*q7*q8 - 2400*p**5*q3*q4*q5*q6**2*q7*q8 - 2400*p**5*q3*q4*q5*q6*q7**2*q8 - 2400*p**5*q3*q4*q5*q6*q7*q8**2 - 2400*p**5*q3*q4*q5*q6*q7*q8*q9 + 16800*p**5*q3*q4*q5*q6*q7*q8 + 2100*p**5*q4**2*q5*q6*q7*q8 + 2100*p**5*q4*q5**2*q6*q7*q8 + 2100*p**5*q4*q5*q6**2*q7*q8 + 2100*p**5*q4*q5*q6*q7**2*q8 + 2100*p**5*q4*q5*q6*q7*q8**2 + 2100*p**5*q4*q5*q6*q7*q8*q9 - 12600*p**5*q4*q5*q6*q7*q8 - 1008*p**5*q5**2*q6*q7*q8 - 1008*p**5*q5*q6**2*q7*q8 - 1008*p**5*q5*q6*q7**2*q8 - 1008*p**5*q5*q6*q7*q8**2 - 1008*p**5*q5*q6*q7*q8*q9 + 5040*p**5*q5*q6*q7*q8 + 210*p**5*q6**2*q7*q8 + 210*p**5*q6*q7**2*q8 + 210*p**5*q6*q7*q8**2 + 210*p**5*q6*q7*q8*q9 - 840*p**5*q6*q7*q8 + 700*p**4*q1**2*q2*q3*q4*q5*q6*q7*q8 + 700*p**4*q1*q2**2*q3*q4*q5*q6*q7*q8 + 700*p**4*q1*q2*q3**2*q4*q5*q6*q7*q8 + 700*p**4*q1*q2*q3*q4**2*q5*q6*q7*q8 + 700*p**4*q1*q2*q3*q4*q5**2*q6*q7*q8 + 700*p**4*q1*q2*q3*q4*q5*q6**2*q7*q8 + 700*p**4*q1*q2*q3*q4*q5*q6*q7**2*q8 + 700*p**4*q1*q2*q3*q4*q5*q6*q7*q8**2 + 700*p**4*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 6300*p**4*q1*q2*q3*q4*q5*q6*q7*q8 - 1575*p**4*q2**2*q3*q4*q5*q6*q7*q8 - 1575*p**4*q2*q3**2*q4*q5*q6*q7*q8 - 1575*p**4*q2*q3*q4**2*q5*q6*q7*q8 - 1575*p**4*q2*q3*q4*q5**2*q6*q7*q8 - 1575*p**4*q2*q3*q4*q5*q6**2*q7*q8 - 1575*p**4*q2*q3*q4*q5*q6*q7**2*q8 - 1575*p**4*q2*q3*q4*q5*q6*q7*q8**2 - 1575*p**4*q2*q3*q4*q5*q6*q7*q8*q9 + 12600*p**4*q2*q3*q4*q5*q6*q7*q8 + 1800*p**4*q3**2*q4*q5*q6*q7*q8 + 1800*p**4*q3*q4**2*q5*q6*q7*q8 + 1800*p**4*q3*q4*q5**2*q6*q7*q8 + 1800*p**4*q3*q4*q5*q6**2*q7*q8 + 1800*p**4*q3*q4*q5*q6*q7**2*q8 + 1800*p**4*q3*q4*q5*q6*q7*q8**2 + 1800*p**4*q3*q4*q5*q6*q7*q8*q9 - 12600*p**4*q3*q4*q5*q6*q7*q8 - 1050*p**4*q4**2*q5*q6*q7*q8 - 1050*p**4*q4*q5**2*q6*q7*q8 - 1050*p**4*q4*q5*q6**2*q7*q8 - 1050*p**4*q4*q5*q6*q7**2*q8 - 1050*p**4*q4*q5*q6*q7*q8**2 - 1050*p**4*q4*q5*q6*q7*q8*q9 + 6300*p**4*q4*q5*q6*q7*q8 + 252*p**4*q5**2*q6*q7*q8 + 252*p**4*q5*q6**2*q7*q8 + 252*p**4*q5*q6*q7**2*q8 + 252*p**4*q5*q6*q7*q8**2 + 252*p**4*q5*q6*q7*q8*q9 - 1260*p**4*q5*q6*q7*q8 - 560*p**3*q1**2*q2*q3*q4*q5*q6*q7*q8 - 560*p**3*q1*q2**2*q3*q4*q5*q6*q7*q8 - 560*p**3*q1*q2*q3**2*q4*q5*q6*q7*q8 - 560*p**3*q1*q2*q3*q4**2*q5*q6*q7*q8 - 560*p**3*q1*q2*q3*q4*q5**2*q6*q7*q8 - 560*p**3*q1*q2*q3*q4*q5*q6**2*q7*q8 - 560*p**3*q1*q2*q3*q4*q5*q6*q7**2*q8 - 560*p**3*q1*q2*q3*q4*q5*q6*q7*q8**2 - 560*p**3*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 5040*p**3*q1*q2*q3*q4*q5*q6*q7*q8 + 945*p**3*q2**2*q3*q4*q5*q6*q7*q8 + 945*p**3*q2*q3**2*q4*q5*q6*q7*q8 + 945*p**3*q2*q3*q4**2*q5*q6*q7*q8 + 945*p**3*q2*q3*q4*q5**2*q6*q7*q8 + 945*p**3*q2*q3*q4*q5*q6**2*q7*q8 + 945*p**3*q2*q3*q4*q5*q6*q7**2*q8 + 945*p**3*q2*q3*q4*q5*q6*q7*q8**2 + 945*p**3*q2*q3*q4*q5*q6*q7*q8*q9 - 7560*p**3*q2*q3*q4*q5*q6*q7*q8 - 720*p**3*q3**2*q4*q5*q6*q7*q8 - 720*p**3*q3*q4**2*q5*q6*q7*q8 - 720*p**3*q3*q4*q5**2*q6*q7*q8 - 720*p**3*q3*q4*q5*q6**2*q7*q8 - 720*p**3*q3*q4*q5*q6*q7**2*q8 - 720*p**3*q3*q4*q5*q6*q7*q8**2 - 720*p**3*q3*q4*q5*q6*q7*q8*q9 + 5040*p**3*q3*q4*q5*q6*q7*q8 + 210*p**3*q4**2*q5*q6*q7*q8 + 210*p**3*q4*q5**2*q6*q7*q8 + 210*p**3*q4*q5*q6**2*q7*q8 + 210*p**3*q4*q5*q6*q7**2*q8 + 210*p**3*q4*q5*q6*q7*q8**2 + 210*p**3*q4*q5*q6*q7*q8*q9 - 1260*p**3*q4*q5*q6*q7*q8 + 280*p**2*q1**2*q2*q3*q4*q5*q6*q7*q8 + 280*p**2*q1*q2**2*q3*q4*q5*q6*q7*q8 + 280*p**2*q1*q2*q3**2*q4*q5*q6*q7*q8 + 280*p**2*q1*q2*q3*q4**2*q5*q6*q7*q8 + 280*p**2*q1*q2*q3*q4*q5**2*q6*q7*q8 + 280*p**2*q1*q2*q3*q4*q5*q6**2*q7*q8 + 280*p**2*q1*q2*q3*q4*q5*q6*q7**2*q8 + 280*p**2*q1*q2*q3*q4*q5*q6*q7*q8**2 + 280*p**2*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 2520*p**2*q1*q2*q3*q4*q5*q6*q7*q8 - 315*p**2*q2**2*q3*q4*q5*q6*q7*q8 - 315*p**2*q2*q3**2*q4*q5*q6*q7*q8 - 315*p**2*q2*q3*q4**2*q5*q6*q7*q8 - 315*p**2*q2*q3*q4*q5**2*q6*q7*q8 - 315*p**2*q2*q3*q4*q5*q6**2*q7*q8 - 315*p**2*q2*q3*q4*q5*q6*q7**2*q8 - 315*p**2*q2*q3*q4*q5*q6*q7*q8**2 - 315*p**2*q2*q3*q4*q5*q6*q7*q8*q9 + 2520*p**2*q2*q3*q4*q5*q6*q7*q8 + 120*p**2*q3**2*q4*q5*q6*q7*q8 + 120*p**2*q3*q4**2*q5*q6*q7*q8 + 120*p**2*q3*q4*q5**2*q6*q7*q8 + 120*p**2*q3*q4*q5*q6**2*q7*q8 + 120*p**2*q3*q4*q5*q6*q7**2*q8 + 120*p**2*q3*q4*q5*q6*q7*q8**2 + 120*p**2*q3*q4*q5*q6*q7*q8*q9 - 840*p**2*q3*q4*q5*q6*q7*q8 - 80*p*q1**2*q2*q3*q4*q5*q6*q7*q8 - 80*p*q1*q2**2*q3*q4*q5*q6*q7*q8 - 80*p*q1*q2*q3**2*q4*q5*q6*q7*q8 - 80*p*q1*q2*q3*q4**2*q5*q6*q7*q8 - 80*p*q1*q2*q3*q4*q5**2*q6*q7*q8 - 80*p*q1*q2*q3*q4*q5*q6**2*q7*q8 - 80*p*q1*q2*q3*q4*q5*q6*q7**2*q8 - 80*p*q1*q2*q3*q4*q5*q6*q7*q8**2 - 80*p*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 720*p*q1*q2*q3*q4*q5*q6*q7*q8 + 45*p*q2**2*q3*q4*q5*q6*q7*q8 + 45*p*q2*q3**2*q4*q5*q6*q7*q8 + 45*p*q2*q3*q4**2*q5*q6*q7*q8 + 45*p*q2*q3*q4*q5**2*q6*q7*q8 + 45*p*q2*q3*q4*q5*q6**2*q7*q8 + 45*p*q2*q3*q4*q5*q6*q7**2*q8 + 45*p*q2*q3*q4*q5*q6*q7*q8**2 + 45*p*q2*q3*q4*q5*q6*q7*q8*q9 - 360*p*q2*q3*q4*q5*q6*q7*q8 + 10*q1**2*q2*q3*q4*q5*q6*q7*q8 + 10*q1*q2**2*q3*q4*q5*q6*q7*q8 + 10*q1*q2*q3**2*q4*q5*q6*q7*q8 + 10*q1*q2*q3*q4**2*q5*q6*q7*q8 + 10*q1*q2*q3*q4*q5**2*q6*q7*q8 + 10*q1*q2*q3*q4*q5*q6**2*q7*q8 + 10*q1*q2*q3*q4*q5*q6*q7**2*q8 + 10*q1*q2*q3*q4*q5*q6*q7*q8**2 + 10*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 90*q1*q2*q3*q4*q5*q6*q7*q8)'
        f_multiparam[10][10] = '-p*(10*p**9*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 45*p**9*q2*q3*q4*q5*q6*q7*q8*q9 + 120*p**9*q3*q4*q5*q6*q7*q8*q9 - 210*p**9*q4*q5*q6*q7*q8*q9 + 252*p**9*q5*q6*q7*q8*q9 - 210*p**9*q6*q7*q8*q9 + 120*p**9*q7*q8*q9 - 45*p**9*q8*q9 + 10*p**9*q9 - p**9 - 90*p**8*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 360*p**8*q2*q3*q4*q5*q6*q7*q8*q9 - 840*p**8*q3*q4*q5*q6*q7*q8*q9 + 1260*p**8*q4*q5*q6*q7*q8*q9 - 1260*p**8*q5*q6*q7*q8*q9 + 840*p**8*q6*q7*q8*q9 - 360*p**8*q7*q8*q9 + 90*p**8*q8*q9 - 10*p**8*q9 + 360*p**7*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 1260*p**7*q2*q3*q4*q5*q6*q7*q8*q9 + 2520*p**7*q3*q4*q5*q6*q7*q8*q9 - 3150*p**7*q4*q5*q6*q7*q8*q9 + 2520*p**7*q5*q6*q7*q8*q9 - 1260*p**7*q6*q7*q8*q9 + 360*p**7*q7*q8*q9 - 45*p**7*q8*q9 - 840*p**6*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 2520*p**6*q2*q3*q4*q5*q6*q7*q8*q9 - 4200*p**6*q3*q4*q5*q6*q7*q8*q9 + 4200*p**6*q4*q5*q6*q7*q8*q9 - 2520*p**6*q5*q6*q7*q8*q9 + 840*p**6*q6*q7*q8*q9 - 120*p**6*q7*q8*q9 + 1260*p**5*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 3150*p**5*q2*q3*q4*q5*q6*q7*q8*q9 + 4200*p**5*q3*q4*q5*q6*q7*q8*q9 - 3150*p**5*q4*q5*q6*q7*q8*q9 + 1260*p**5*q5*q6*q7*q8*q9 - 210*p**5*q6*q7*q8*q9 - 1260*p**4*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 2520*p**4*q2*q3*q4*q5*q6*q7*q8*q9 - 2520*p**4*q3*q4*q5*q6*q7*q8*q9 + 1260*p**4*q4*q5*q6*q7*q8*q9 - 252*p**4*q5*q6*q7*q8*q9 + 840*p**3*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 1260*p**3*q2*q3*q4*q5*q6*q7*q8*q9 + 840*p**3*q3*q4*q5*q6*q7*q8*q9 - 210*p**3*q4*q5*q6*q7*q8*q9 - 360*p**2*q1*q2*q3*q4*q5*q6*q7*q8*q9 + 360*p**2*q2*q3*q4*q5*q6*q7*q8*q9 - 120*p**2*q3*q4*q5*q6*q7*q8*q9 + 90*p*q1*q2*q3*q4*q5*q6*q7*q8*q9 - 45*p*q2*q3*q4*q5*q6*q7*q8*q9 - 10*q1*q2*q3*q4*q5*q6*q7*q8*q9)'

        experiment = [0.33371428571428574, 0.16028571428571428, 0.22114285714285714, 0.17657142857142857, 0.08514285714285715, 0.019428571428571427, 0.0034285714285714284, 0.00028571428571428574, 0.0, 0.0, 0.0]
        n_samples = 3500
        intervals = create_intervals(0.95, n_samples, experiment)
        replaced_f6 = f_multiparam[10][6].replace("p", "0.10400390625").replace("q1", "0.09326171875").replace("q2", "0.1064453125").replace("q3", "0.099609375").replace("q4", "0.072021484375")
        sys.setrecursionlimit(23000)
        start_time = time.time()

        # result6 = check_deeper([(0.0869140625000000, 0.112304687500000), (0, 1)], [replaced_f6], [intervals[6]], 16, 0.01**3*0.5, 0.999, False, 5)

        ## TO RUN THIS TEST UNCOMENT FOLLOWING LINE
        # result6 = check_deeper([(0.0869140625000000, 0.112304687500000), (0, 1)], [replaced_f6], [intervals[6]], 16, 0.01 ** 3 , 0.999, False, 4)

        print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")

        # start_time = time.time()
        # check_deeper([(0, 2)], ["x**2", "x+3"], [Interval(0, 1), Interval(0, 1)], 6, 0.01 ** 2, 0.9, False, 4)
        # print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")


        ## VERY VERY INTERESTING RESULT
        from load import load_all_data
        D = load_all_data("data/data*.csv")
        check_deeper([(0, 1), (0, 1)], f[2], create_intervals(0.95, 1500, D[2]), 14, 0.01 ** 2, 0.997, False, 5)

    def test_sample(self):
        print(colored("Sample test here", 'blue'))
        ## def sample(space, props, intervals, size_q, compress)
        # print(sample(RefinedSpace((0, 1), ["x"]), ["x"], [Interval(0, 1)], 3))
        # print(sample(RefinedSpace((0, 1), ["x"]), ["x"], [Interval(0, 1)], 3, compress=True))

        # print(sample(RefinedSpace((0, 2), ["x"]), ["x"], [Interval(0, 1)], 3))
        # print(sample(RefinedSpace((0, 2), ["x"]), ["x"], [Interval(0, 1)], 3, compress=True))

        # sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y"], [Interval(0, 1)], 3, compress=True)

        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y"], [Interval(0, 1)], 2, compress=True)
        # print(a)
        # refine_into_rectangles(a)

        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], 5,
        #            compress=True, silent=False)
        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 0.9), Interval(0, 1)], 3, compress=True)

        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], 2)
        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y"], [Interval(0, 1)], 2, compress=True)

        # print(a)
        # b = refine_into_rectangles(a, silent=False)

    def test_presampled(self):
        print(colored("Presampled refinement here", 'blue'))
        ## UNCOMMENT THIS TBD
        # check_deeper([(0, 1), (2, 3)], ["x+y"], [Interval(0, 3)], 5, 0, 0.95, silent=False, version=5)

        # check_deeper([(0, 1), (2, 3)], ["x+y"], [Interval(0, 3)], 5, 0, 0.95, silent=True, version=5)
        # check_deeper([(0, 1), (2, 3)], ["x+y"], [Interval(0, 3)], 5, 0, 0.95, silent=True, version=6)

        # check_deeper([(0, 1), (2, 3)], ["x", "y"], [Interval(0.5, 3), Interval(2.5, 3)], 5, 0, 0.95, silent=True, version=5)
        # check_deeper([(0, 1), (2, 3)], ["x", "y"], [Interval(0, 3), Interval(2.5, 3)], 20, 0, 0.95, silent=True, version=5)

        # check_deeper([[0, 1], [2, 2.5]], ["x", "y"], [Interval(0, 3), Interval(2.5, 3)], 20, 0, 0.95, silent=False, version=5)


        check_deeper([(0, 1), (2, 3)], ["x", "y"], [Interval(0, 3), Interval(2.5, 3)], 15, 0, 0.95, silent=False, version=6)

        # check_deeper([(0, 0.5), (0, 0.5)], ["x+y"], [Interval(0, 1)], 5, 0, 0.95, silent=False, version=5)

        # a = sample(RefinedSpace([(0, 1), (0, 1), (0, 1)], ["x", "y", "z"]), ["x+y"], [Interval(0, 1)], 3, compress=True)
        # print(a)
        # b = refine_into_rectangles(a)
        print(colored("Presampled refinement ends here", 'blue'))

    def test_timeout(self):
        print(colored("Timeout test here", 'blue'))
        ## TIMEOUT TEST
        # print("TIMEOUT TEST not finish")
        # print(timeout(check_deeper, ([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 10, 0, 0.95, True, 1), timeout_duration=20,
        #        default=4))

        # print("TIMEOUT TEST2 not finish")
        # print(type(check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 10, 0, 0.95, silent=True, version=1, time_out=2)))

        # print("TIMEOUT TEST finish")
        # timeout(check_deeper, ([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 10, 0, 0.95, True, 1), timeout_duration=20, default=4)

        # print("TIMEOUT TEST2 finish")
        # print(type(check_deeper([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 10, 0, 0.95, silent=True, version=1, time_out=20)))


if __name__ == "__main__":
    unittest.main()
    # check_interval([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=False)
    # check_interval([(0, 3)], ["x"], [Interval(0, 2)], silent=False, called=False)
