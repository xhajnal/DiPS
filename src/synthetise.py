import os
import re
import socket
import sys
import time
from collections import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numpy import prod

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param
from space import RefinedSpace


# space = RefinedSpace([(0,1)])

import configparser

config = configparser.ConfigParser()
# print(os.getcwd())
config.read("../config.ini")
# config.sections()
z3_path = config.get("paths", "z3_path")

if not os.path.exists(z3_path):
    raise OSError("Directory does not exist: " + str(z3_path))

cwd = os.getcwd()
os.chdir(z3_path)
print("z3_path", z3_path)

# sys.path.append("../python")
# sys.path.append("..")
# from z3 import *
# os.chdir(cwd)

try:
    from z3 import *
    # print(os.getcwd())
    # import subprocess
    # subprocess.call(["python", "example.py"])
except:
    raise Exception("could not load z3 from: ", z3_path)
finally:
    os.chdir(cwd)

try:
    p = Real('p')
except:
    import platform

    if '/' in z3_path:
        z3_path_short = '/'.join(z3_path.split("/")[:-1])
    elif '\\' in z3_path:
        z3_path_short = '\\'.join(z3_path.split("\\")[:-1])
    else:
        print("Warning: Could not set path to add to the PATH, please add it manually")

    if z3_path_short not in os.environ["PATH"]:
        if z3_path_short.replace("/", "\\") not in os.environ["PATH"]:
            if "wind" in platform.system().lower():
                os.environ["PATH"] = os.environ["PATH"] + ";" + z3_path_short
            else:
                os.environ["PATH"] = os.environ["PATH"] + ":" + z3_path_short
    os.environ["PYTHONPATH"] = z3_path
    os.environ["Z3_LIBRARY_PATH"] = z3_path
    os.environ["Z3_LIBRARY_DIRS"] = z3_path
    try:
        p = Real('p')
    except:
        raise Exception("z3 not loaded properly")


class Queue:
    # Constructor creates a list
    def __init__(self):
        self.queue = list()

    # Adding elements to queue
    def enqueue(self, data):
        # Checking to avoid duplicate entry (not mandatory)
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return "Queue Empty!"

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue


def check(region, prop, intervals, silent=False, called=False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: array of intervals to constrain properties
    silent: if silent printed output is set to minimum
    called: if called updates the global variables (use when calling it directly)
    """
    ## Initialisation
    if not silent:
        print("checking unsafe", region)

    if called:
        ## Parse parameteres from properties
        globals()["parameters"] = set()
        for polynome in prop:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  print(parameters)
        ##           >> ['p','q']
        for param in globals()["parameters"]:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p) 
        if not len(globals()["parameters"]) == len(region) and not silent:
            print("number of parameters in property ({}) and dimension of the region ({}) is not equal".format(
                len(globals()["parameters"]), len(region)))

    s = Solver()

    ## Adding regional restrictions to solver
    for j in range(len(globals()["parameters"])):
        s.add(globals()[parameters[j]] > region[j][0])
        # print(str(globals()[parameters[j]] > region[j][0]))
        s.add(globals()[parameters[j]] < region[j][1])
        # print(str(globals()[parameters[j]] < region[j][1]))

    ## Adding property in the interval restrictions to solver
    for i in range(0, len(prop)):
        # if intervals[i]<100/n_samples:
        #    continue

        ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        # if  intervals[i]<0.01:
        #    continue

        s.add(eval(prop[i]) > intervals[i].start, eval(prop[i]) < intervals[i].end)
        # print(prop[i],intervals[i])

    if s.check() == sat:
        return s.model()
    else:
        add_space = []
        for interval in region:
            add_space.append(interval[1] - interval[0])
        # print("add_space", add_space)
        globals()["non_white_area"] = globals()["non_white_area"] + prod(add_space)
        # print("area added: ",prod(add_space))
        globals()["hyper_rectangles_unsat"].append(region)
        if len(region) == 2:  # if two-dim param space
            globals()["rectangles_unsat"].append(
                Rectangle((region[0][0], region[1][0]), region[0][1] - region[0][0], region[1][1] - region[1][0], fc='r'))
        if len(region) == 1:
            globals()["rectangles_unsat"].append(
                Rectangle((region[0][0], 0.33), region[0][1] - region[0][0], 0.33, fc='r'))
        # print("red", Rectangle((region[0][0],region[1][0]), region[0][1]-region[0][0], region[1][1]-region[1][0], fc='r'))
        return ("unsafe")


def check_safe(region, prop, intervals, silent=False, called=False):
    """ Check if the given region is safe or not 

    It means whether for all parametrisations in **region** every property(prop) is evaluated within the given
    **interval**, otherwise it is not safe and counterexample is returned.

    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: array of intervals to constrain properties
    silent: if silent printed output is set to minimum
    called: if called updates the global variables (use when calling it directly)
    """
    # initialisation
    if not silent:
        print("checking safe", region)

    if called:
        globals()["parameters"] = set()
        for polynome in prop:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        for param in globals()["parameters"]:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p) 
        if not len(globals()["parameters"]) == len(region) and not silent:
            print("number of parameters in property ({}) and dimension of the region ({}) is not equal".format(
                len(globals()["parameters"]), len(region)))
    s = Solver()

    ## Adding regional restrictions to solver
    for j in range(len(globals()["parameters"])):
        s.add(globals()[parameters[j]] > region[j][0])
        s.add(globals()[parameters[j]] < region[j][1])

    ## Adding property in the interval restrictions to solver

    formula = Or(Not(eval(prop[0]) > intervals[0].start), Not(eval(prop[0]) < intervals[0].end))

    ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
    # if intervals[0]<0.01:
    #    formula = Or(Not(eval(prop[0]) > intervals[0][0])), Not(eval(prop[0]) < intervals[0][1]))) 
    # else:
    #    formula = False

    for i in range(1, len(prop)):
        # if intervals[i]<100/n_samples:
        #    continue       

        ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        # if  intervals[i]<0.01:
        #    continue
        formula = Or(formula, Or(Not(eval(prop[i]) > intervals[i].start), Not(eval(prop[i]) < intervals[i].end)))
    s.add(formula)
    # print(s.check())
    # return s.check()
    if s.check() == unsat:
        add_space = []
        for interval in region:
            add_space.append(interval[1] - interval[0])
        # print("add_space", add_space)
        globals()["non_white_area"] = globals()["non_white_area"] + prod(add_space)
        # print("area added: ",prod(add_space))
        globals()["hyper_rectangles_sat"].append(region)
        if len(region) == 2:  # if two-dim param space
            globals()["rectangles_sat"].append(
                Rectangle((region[0][0], region[1][0]), region[0][1] - region[0][0], region[1][1] - region[1][0],
                          fc='g'))
        if len(region) == 1:
            globals()["rectangles_sat"].append(
                Rectangle((region[0][0], 0.33), region[0][1] - region[0][0], 0.33, fc='g'))

        # print("green", Rectangle((region[0][0],region[1][0]), region[0][1]-region[0][0], region[1][1]-region[1][0], fc='g'))
        return "safe"
    else:
        return s.model()


# class Refinement


def check_deeper(region, prop, intervals, n, epsilon, cov, silent, version):
    """ Refining the parameter space into safe and unsafe regions with respective alg/method
    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computation
    silent: if silent printed output is set to minimum
    version: version of the algorithm to be used
    """

    if not isinstance(region, Iterable):
        region = [region]
    if not isinstance(prop, Iterable):
        prop = [prop]
    if not isinstance(intervals, Iterable):
        intervals = [intervals]

    ## Initialisation
    ## region
    globals()["default_region"] = copy.copy(region)

    globals()["rectangles_sat"] = []
    globals()["rectangles_unsat"] = []
    globals()["rectangles_unsat_added"] = []

    globals()["hyper_rectangles_sat"] = []
    globals()["hyper_rectangles_unsat"] = []
    globals()["hyper_rectangles_white"] = [region]

    globals()["non_white_area"] = 0
    globals()["whole_area"] = []
    for interval in region:
        globals()["whole_area"].append(interval[1] - interval[0])
    globals()["whole_area"] = prod(globals()["whole_area"])

    if not silent:
        print("the area is: ", region)
        print("the volume of the whole area is:", globals()["whole_area"])

    ## params
    globals()["parameters"] = set()
    for polynome in prop:
        globals()["parameters"].update(find_param(polynome))
    globals()["parameters"] = sorted(list(globals()["parameters"]))
    ## EXAMPLE:  parameters >> ['p','q']

    for param in globals()["parameters"]:
        globals()[param] = Real(param)
    ## EXAMPLE: p = Real(p)

    if not len(globals()["parameters"]) == len(region) and not silent:
        print("number of parameters in property ({}) and dimension of the region ({}) is not equal".format(
            len(globals()["parameters"]), len(region)))

    ## Choosing from versions
    start_time = time.time()
    if version == 1:
        print("Using DFS method")
        private_check_deeper(region, prop, intervals, n, epsilon, cov, silent)
    if version == 2:
        print("Using BFS method")
        globals()["que"] = Queue()
        private_check_deeper_queue(region, prop, intervals, n, epsilon, cov, silent)
    if version == 3:
        print("Using BFS method with passing examples")
        globals()["que"] = Queue()
        private_check_deeper_queue_checking(region, prop, intervals, n, epsilon, cov, silent, None)
    if version == 4:
        print("Using BFS method with passing examples and counterexamples")
        globals()["que"] = Queue()
        private_check_deeper_queue_checking_both(region, prop, intervals, n, epsilon, cov, silent, None)
    if version == 5:
        print("Using iterative methods")
        print(check_deeper_iter(region, prop, intervals, n, epsilon, cov, silent))

    ## Visualisation
    if len(region) == 1 or len(region) == 2:
        colored(globals()["default_region"], region)
        # from matplotlib import rcParams
        # rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
        fig = plt.figure()
        pic = fig.add_subplot(111, aspect='equal')
        pic.set_xlabel(globals()["parameters"][0])
        ## set axis ranges
        if region[0][1] - region[0][0] < 0.1:
            region[0] = (region[0][0] - 0.2, region[0][1] + 0.2)
        pic.axis([region[0][0], region[0][1], 0, 1])
        if len(region) == 2:
            pic.set_ylabel(globals()["parameters"][1])
            if region[1][1] - region[1][0] < 0.1:
                region[1] = (region[1][0] - 0.2, region[1][1] + 0.2)
            pic.axis([region[0][0], region[0][1], region[1][0], region[1][1]])
        pic.set_title("red = unsafe region, green = safe region, white = in between \n max_recursion_depth:{},"
                      " \n min_rec_size:{}, achieved_coverage:{}, alg{} \n It took {} {} second(s)".format(
            n, epsilon, globals()["non_white_area"] / globals()["whole_area"], version,
            socket.gethostname(), round(time.time() - start_time, 1)))
        pc = PatchCollection(rectangles_unsat, facecolor='r', alpha=0.5)
        pic.add_collection(pc)
        pc = PatchCollection(rectangles_sat, facecolor='g', alpha=0.5)
        pic.add_collection(pc)
        pc = PatchCollection(rectangles_unsat_added, facecolor='xkcd:grey', alpha=0.5)
        pic.add_collection(pc)
        plt.show()
    print("result coverage is: ", globals()["non_white_area"] / globals()["whole_area"])
    return (globals()["hyper_rectangles_sat"], globals()["hyper_rectangles_unsat"], globals()["hyper_rectangles_white"],
            globals()["non_white_area"] / globals()["whole_area"])


def private_check_deeper_sampling(region, prop, intervals, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions
    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent print
    """

    import numpy as np

    sampling_size = 10

    sampled_true = []
    sampled_false = []

    for interval in len(region):
        for value in np.linspace(region[interval][0], region[interval][1], num=sampling_size):
            globals()[parameters[interval]] = value


def private_check_deeper(region, prop, intervals, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions
    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent printed output is set to minimum
    """

    # checking this:
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    # stop if the given hyperrectangle is to small
    add_space = []
    for interval in region:
        add_space.append(interval[1] - interval[0])
    add_space = prod(add_space)
    if add_space < epsilon:
        if len(region) > 2:
            # if not silent:
            #    print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(region) == 2:
            # if not silent:
            #    print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            # if not silent:
            #    print("interval too small, skipped")
            return "interval too small, skipped"

    # stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"] / globals()["whole_area"] > coverage:
            return "coverage ", globals()["non_white_area"] / globals()["whole_area"], " is above the threshold"

    # HERE MAY ADDING THE MODEL
    if check(region, prop, intervals, silent) == "unsafe":
        result = "unsafe"
    elif check_safe(region, prop, intervals, silent) == "safe":
        result = "safe"
    else:
        result = "unknown"

    # print("result",result)
    if result == "safe" or result == "unsafe":
        globals()["hyper_rectangles_white"].remove(region)
    if n == 0:
        # print("[",p_low,",",p_high ,"],[",q_low,",",q_high ,"]",result)
        if not silent:
            print("maximal recursion reached here with coverage:",
                  globals()["non_white_area"] / globals()["whole_area"])
        return result
    else:
        if not (
                result == "safe" or result == "unsafe"):  # here is necessary to check only 3 of 4, since this line check 1 segment
            # find max interval
            index, maximum = 0, 0
            for i in range(len(region)):
                value = region[i][1] - region[i][0]
                if value > maximum:
                    index = i
                    maximum = value
            low = region[index][0]
            high = region[index][1]
            foo = copy.copy(region)
            foo[index] = (low, low + (high - low) / 2)
            foo2 = copy.copy(region)
            foo2[index] = (low + (high - low) / 2, high)
            globals()["hyper_rectangles_white"].remove(region)
            globals()["hyper_rectangles_white"].append(foo)  # add this region as white
            globals()["hyper_rectangles_white"].append(foo2)  # add this region as white
            # print("white area",globals()["hyper_rectangles_white"])
            if silent:
                private_check_deeper(foo, prop, intervals, n - 1, epsilon, coverage, silent)
                if globals()["whole_area"] > 0:
                    if globals()["non_white_area"] / globals()["whole_area"] > coverage:
                        return "coverage ", globals()["non_white_area"] / globals()[
                            "whole_area"], " is above the threshold"
                private_check_deeper(foo2, prop, intervals, n - 1, epsilon, coverage, silent)
            else:
                print(n, foo, globals()["non_white_area"] / globals()["whole_area"],
                      private_check_deeper(foo, prop, intervals, n - 1, epsilon, coverage, silent))
                if globals()["whole_area"] > 0:
                    if globals()["non_white_area"] / globals()["whole_area"] > coverage:
                        return "coverage ", globals()["non_white_area"] / globals()[
                            "whole_area"], " is above the threshold"
                print(n, foo2, globals()["non_white_area"] / globals()["whole_area"],
                      private_check_deeper(foo2, prop, intervals, n - 1, epsilon, coverage, silent))

    return result


def colored(greater, smaller):
    """ Colors outside of the smaller region in the greater region as previously unsat

    Parameters
    ----------
    greater: region in which the smaller region is located
    smaller: smaller region which is not to be colored
    """
    # rectangles_sat.append(Rectangle((low_x,low_y), width, height, fc='g'))
    # print("greater ",greater)
    # print("smaller ",smaller)
    if greater is None or smaller is None:
        return

    ## if 1 dimensional coloring
    if len(smaller) == 1:
        ## color 2 regions, to the left, to the right
        ## to the left
        globals()["rectangles_unsat_added"].append(
            Rectangle([greater[0][0], 0], smaller[0][0] - greater[0][0], 1, fc='r'))
        ## to the right
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][1], 0], greater[0][1] - smaller[0][1], 1, fc='r'))

    # else 2 dimensional coloring
    elif len(smaller) == 2:
        ## color 4 regions, to the left, to the right, below, and above
        ##
        globals()["rectangles_unsat_added"].append(
            Rectangle([greater[0][0], 0], smaller[0][0] - greater[0][0], 1, fc='r'))
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][1], 0], greater[0][1] - smaller[0][1], 1, fc='r'))
        globals()["rectangles_unsat_added"].append(
            Rectangle([smaller[0][0], 0], smaller[0][1] - smaller[0][0], smaller[1][0], fc='r'))
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


def check_deeper_iter(region, props, intervals, n, epsilon, coverage, silent):
    """ New Refining the parameter space into safe and unsafe regions with iterative method using alg1

    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    props: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent printed output is set to minimum
    """
    new_tresh = copy.copy(region)

    # implement ordering of the props with intervals
    for i in range(len(props) - 1):
        if not silent:
            # print("white: ",globals()["hyper_rectangles_white"])
            print("check_deeper(", new_tresh, [props[i]], [intervals[i]], ")")
        check_deeper(new_tresh, [props[i]], [intervals[i]], n, epsilon, coverage, True, 1)

        new_tresh = []
        for interval_index in range(len(region)):
            minimum = 9001
            maximum = 0
            # iterate though green regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_sat"])):
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0] < minimum:
                    minimum = globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1] > maximum:
                    maximum = globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1]
            # iterate though white regions to find min and max
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


def private_check_deeper_queue(region, prop, intervals, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions
    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent printed output is set to minimum
    """
    # print(region,prop,intervals,n,epsilon,coverage,silent)
    # print("region",region)
    # print("white regions: ", globals()["hyper_rectangles_white"])

    ## checking this:
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    ## stop if the given hyperrectangle is to small
    add_space = []
    for interval in region:
        add_space.append(interval[1] - interval[0])
    add_space = prod(add_space)
    # print("add_space",add_space)
    if add_space < epsilon:
        if len(region) > 2:
            if not silent:
                print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            if not silent:
                print("interval too small, skipped")
            return "interval too small, skipped"

    ## stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"] / globals()["whole_area"] > coverage:
            globals()["que"] = Queue()
            return "coverage ", globals()["non_white_area"] / globals()["whole_area"], " is above the threshold"

    # HERE I CAN APPEND THE VALUE OF EXAMPLE AND COUNTEREXAMPLE
    # print("hello check =",check(region,prop,intervals,silent))
    # print("hello check safe =",check_safe(region,prop,n_samples,silent))
    if check(region, prop, intervals, silent) == "unsafe":
        result = "unsafe"
    elif check_safe(region, prop, intervals, silent) == "safe":
        result = "safe"
    else:
        result = "unknown"

    if result == "safe" or result == "unsafe":
        # print("removing region:", region)
        globals()["hyper_rectangles_white"].remove(region)

    if not silent:
        print(n, region, globals()["non_white_area"] / globals()["whole_area"], result)

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
    foo = copy.copy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.copy(region)
    foo2[index] = (low + (high - low) / 2, high)
    globals()["hyper_rectangles_white"].remove(region)
    globals()["hyper_rectangles_white"].append(foo)
    globals()["hyper_rectangles_white"].append(foo2)

    # ADD CALLS TO QUEUE
    # print("adding",[copy.copy(foo),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo),prop,intervals,n-1,epsilon,coverage,silent]))
    # print("adding",[copy.copy(foo2),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo2),prop,intervals,n-1,epsilon,coverage,silent]))
    globals()["que"].enqueue([copy.copy(foo), prop, intervals, n - 1, epsilon, coverage, silent])
    globals()["que"].enqueue([copy.copy(foo2), prop, intervals, n - 1, epsilon, coverage, silent])

    # CALL QUEUE
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue(*que.dequeue())


def private_check_deeper_queue_checking(region, prop, intervals, n, epsilon, coverage, silent, model=None):
    """ THIS IS OBSOLETE METHOD, HERE JUST TO BE COMPARED WITH THE NEW ONE

    Refining the parameter space into safe and unsafe regions

    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent printed output is set to minimum
    model: (example,counterexample) of the satisfaction in the given region
    """
    # print(region,prop,intervals,n,epsilon,coverage,silent)

    # checking this:
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    ## stop if the given hyperrectangle is to small
    add_space = []
    for interval in region:
        add_space.append(interval[1] - interval[0])
    add_space = prod(add_space)
    # print("add_space",add_space)
    if add_space < epsilon:
        if len(region) > 2:
            if not silent:
                print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            if not silent:
                print("interval too small, skipped")
            return "interval too small, skipped"

    ## stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"] / globals()["whole_area"] > coverage:
            globals()["que"] = Queue()
            return "coverage ", globals()["non_white_area"] / globals()["whole_area"], " is above the threshold"

    if model is None:
        example = check(region, prop, intervals, silent)
        # counterexample = check_safe(region,prop,intervals,silent)
    elif model[0] is None:
        example = check(region, prop, intervals, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]

    ## Resolving the result
    if example == "unsafe":
        globals()["hyper_rectangles_white"].remove(region)
        if not silent:
            print(n, region, globals()["non_white_area"] / globals()["whole_area"], "unsafe")
        return
    elif check_safe(region, prop, intervals, silent) == "safe":
        globals()["hyper_rectangles_white"].remove(region)
        if not silent:
            print(n, region, globals()["non_white_area"] / globals()["whole_area"], "safe")
        return
    else:  ## unknown
        if not silent:
            print(n, region, globals()["non_white_area"] / globals()["whole_area"], example)

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
    foo = copy.copy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.copy(region)
    foo2[index] = (low + (high - low) / 2, high)
    globals()["hyper_rectangles_white"].remove(region)
    globals()["hyper_rectangles_white"].append(foo)
    globals()["hyper_rectangles_white"].append(foo2)

    model_low = [9, 9]
    model_high = [9, 9]
    if float(eval(example_points[index])) > low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = example
    else:
        model_low[0] = example
        model_high[0] = None
    # overwrite if equal
    if float(eval(example_points[index])) == low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = None

    globals()["que"].enqueue(
        [copy.copy(foo), prop, intervals, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue(
        [copy.copy(foo2), prop, intervals, n - 1, epsilon, coverage, silent, model_high])

    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking(*que.dequeue())


def private_check_deeper_queue_checking_both(region, prop, intervals, n, epsilon, coverage, silent,
                                             model=None):
    """ Refining the parameter space into safe and unsafe regions
    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    prop: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent printed output is set to minimum
    model: (example,counterexample) of the satisfaction in the given region
    """
    # print(region,prop,intervals,n,epsilon,coverage,silent)

    # checking this:
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    # stop if the given hyperrectangle is to small
    add_space = []
    for interval in region:
        add_space.append(interval[1] - interval[0])
    add_space = prod(add_space)
    # print("add_space",add_space)
    if add_space < epsilon:
        if len(region) > 2:
            if not silent:
                print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            if not silent:
                print("interval too small, skipped")
            return "interval too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"] / globals()["whole_area"] > coverage:
            globals()["que"] = Queue()
            return "coverage ", globals()["non_white_area"] / globals()["whole_area"], " is above the threshold"

    ## Resolving if the region safe/unsafe/unknown
    if model is None:
        example = check(region, prop, intervals, silent)
        counterexample = check_safe(region, prop, intervals, silent)
    elif model[0] is None:
        example = check(region, prop, intervals, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]
    if model is not None:
        if model[1] is None:
            counterexample = check_safe(region, prop, intervals, silent)
        else:
            if not silent:
                print("skipping check_safe at", region, "since counterexample", model[1])
            counterexample = model[1]

    ## Resolving the result
    if example == "unsafe":
        globals()["hyper_rectangles_white"].remove(region)
        if not silent:
            print(n, region, globals()["non_white_area"] / globals()["whole_area"], "unsafe")
        return
    elif counterexample == "safe":
        globals()["hyper_rectangles_white"].remove(region)
        if not silent:
            print(n, region, globals()["non_white_area"] / globals()["whole_area"], "safe")
        return
    else:  ## unknown
        if not silent:
            print(n, region, globals()["non_white_area"] / globals()["whole_area"], (example, counterexample))

    if n == 0:
        return

    example_points = re.findall(r'[0-9/]+', str(example))
    counterexample_points = re.findall(r'[0-9/]+', str(counterexample))
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
    foo = copy.copy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.copy(region)
    foo2[index] = (low + (high - low) / 2, high)
    globals()["hyper_rectangles_white"].remove(region)
    globals()["hyper_rectangles_white"].append(foo)
    globals()["hyper_rectangles_white"].append(foo2)

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
    # overwrite if equal
    if float(eval(example_points[index])) == low + (high - low) / 2:
        model_low[0] = None
        model_high[0] = None
    if float(eval(counterexample_points[index])) == low + (high - low) / 2:
        model_low[1] = None
        model_high[1] = None

    globals()["que"].enqueue([copy.copy(foo), prop, intervals, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue([copy.copy(foo2), prop, intervals, n - 1, epsilon, coverage, silent, model_high])

    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking_both(*que.dequeue())
