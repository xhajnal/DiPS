import os
import re
import socket
import sys
import time
from matplotlib.patches import Rectangle

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param
from space import RefinedSpace
from space import get_rectangle_volume

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


def check(region, props, intervals, silent=False, called=False):
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
        print("checking unsafe", region)

    # p = Real('p')
    # print(p)
    # print(type(p))

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        parameters = sorted(list(parameters))
        ## EXAMPLE:  parameters >> ['p','q']
        for param in parameters:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.copy(region), parameters, [], [])
    else:
        space = globals()["space"]

    s = Solver()

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        s.add(globals()[param] > region[j][0])
        s.add(globals()[param] < region[j][1])
        j = j + 1

    ## Adding property in the interval restrictions to solver
    for i in range(0, len(props)):
        # if intervals[i]<100/n_samples:
        #    continue

        ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        # if  intervals[i]<0.01:
        #    continue

        s.add(eval(props[i]) > intervals[i].start, eval(props[i]) < intervals[i].end)
        # print(prop[i],intervals[i])

    if s.check() == sat:
        return s.model()
    else:
        space.add_red(region)
        return ("unsafe")


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
        print("checking safe", region)

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        parameters = sorted(list(parameters))
        ## EXAMPLE:  parameters >> ['p','q']

        for param in parameters:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.copy(region), parameters, [], [])
    else:
        space = globals()["space"]

    s = Solver()

    ## Adding regional restrictions to solver
    j = 0
    for param in globals()["parameters"]:
        s.add(globals()[param] > region[j][0])
        s.add(globals()[param] < region[j][1])
        j = j + 1

    ## Adding property in the interval restrictions to solver

    formula = Or(Not(eval(props[0]) > intervals[0].start), Not(eval(props[0]) < intervals[0].end))

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
        formula = Or(formula, Or(Not(eval(props[i]) > intervals[i].start), Not(eval(props[i]) < intervals[i].end)))
    s.add(formula)
    # print(s.check())
    # return s.check()
    if s.check() == unsat:
        space.add_green(region)
        return "safe"
    else:
        return s.model()


def check_deeper(region, props, intervals, n, epsilon, coverage, silent, version):
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
    version: (Bool): version of the algorithm to be used
    """
    ## Initialisation
    ## Params
    globals()["parameters"] = set()
    for polynome in props:
        globals()["parameters"].update(find_param(polynome))
    globals()["parameters"] = sorted(list(globals()["parameters"]))
    parameters = globals()["parameters"]

    globals()["space"] = RefinedSpace(copy.copy(region), parameters, [], [])
    space = globals()["space"]

    globals()["default_region"] = copy.copy(region)

    if not silent:
        print("the area is: ", space.region)
        print("the volume of the whole area is:", space.get_volume())

    ## Choosing from versions
    start_time = time.time()
    if version <= 4:
        for param in parameters:
            globals()[param] = Real(param)
    else:
        print()
        # i = 0
        # for param in parameters:
        #     globals()[param] = mpi(region[i][0], region[i][1])

    if version == 1:
        print("Using DFS method")
        private_check_deeper(region, props, intervals, n, epsilon, coverage, silent)
    if version == 2:
        print("Using BFS method")
        globals()["que"] = Queue()
        private_check_deeper_queue(region, props, intervals, n, epsilon, coverage, silent)
    if version == 3:
        print("Using BFS method with passing examples")
        globals()["que"] = Queue()
        private_check_deeper_queue_checking(region, props, intervals, n, epsilon, coverage, silent, None)
    if version == 4:
        print("Using BFS method with passing examples and counterexamples")
        globals()["que"] = Queue()
        private_check_deeper_queue_checking_both(region, props, intervals, n, epsilon, coverage, silent, None)
    if version == 5:
        print("Using iterative method")
        print(check_deeper_iter(region, props, intervals, n, epsilon, coverage, silent))

    ## Visualisation
    space.show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version} \n It took {socket.gethostname()} {round(time.time() - start_time)} second(s)")
    print("result coverage is: ", space.get_coverage())
    return space


def private_check_deeper_sampling(region, props, intervals, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions
    Args
    ----------
    region: (list of intervals) array of pairs, low and high bound, defining the parameter space to be refined
    props:  (list of strings): array of polynomes
    intervals: (list of sympy.Interval): array of intervals to constrain properties
    n: (Int): max number of recursions to do
    epsilon: (Float): minimal size of rectangle to be checked
    coverage: (Float): coverage threshold to stop computation
    silent: (Bool): if silent print
    """

    import numpy as np

    sampling_size = 10

    sampled_true = []
    sampled_false = []

    for interval in len(region):
        for value in np.linspace(region[interval][0], region[interval][1], num=sampling_size):
            globals()[parameters[interval]] = value


def private_check_deeper(region, props, intervals, n, epsilon, coverage, silent):
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
    """

    ## TBD check consitency
    # print(region,prop,intervals,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
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

    ## Stop if the the current coverage is above the given thresholds
    if space.get_coverage() > coverage:
        return "coverage ", space.get_coverage(), " is above the threshold"

    # HERE MAY ADDING THE MODEL
    if check(region, props, intervals, silent) == "unsafe":
        result = "unsafe"
    elif check_safe(region, props, intervals, silent) == "safe":
        result = "safe"
    else:
        result = "unknown"

    # print("result",result)
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
            foo = copy.copy(region)
            foo[index] = (low, low + (high - low) / 2)
            foo2 = copy.copy(region)
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


def colored(greater, smaller):
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
        ## color 4 regions, to the left, to the right, below, and above
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


def check_deeper_iter(region, props, intervals, n, epsilon, coverage, silent):
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
    """
    new_tresh = copy.copy(region)

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


def private_check_deeper_queue(region, props, intervals, n, epsilon, coverage, silent):
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
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    ## HERE I CAN APPEND THE VALUE OF EXAMPLE AND COUNTEREXAMPLE
    # print("hello check =",check(region,prop,intervals,silent))
    # print("hello check safe =",check_safe(region,prop,n_samples,silent))
    if check(region, props, intervals, silent) == "unsafe":
        result = "unsafe"
    elif check_safe(region, props, intervals, silent) == "safe":
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
    foo = copy.copy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.copy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Add calls to the Queue
    # print("adding",[copy.copy(foo),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo),prop,intervals,n-1,epsilon,coverage,silent]))
    # print("adding",[copy.copy(foo2),prop,intervals,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo2),prop,intervals,n-1,epsilon,coverage,silent]))
    globals()["que"].enqueue([copy.copy(foo), props, intervals, n - 1, epsilon, coverage, silent])
    globals()["que"].enqueue([copy.copy(foo2), props, intervals, n - 1, epsilon, coverage, silent])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue(*que.dequeue())


def private_check_deeper_queue_checking(region, props, intervals, n, epsilon, coverage, silent, model=None):
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
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    if model is None:
        example = check(region, props, intervals, silent)
        # counterexample = check_safe(region,prop,intervals,silent)
    elif model[0] is None:
        example = check(region, props, intervals, silent)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]

    ## Resolving the result
    if example == "unsafe":
        space.remove_white(region)
        if not silent:
            print(n, region, space.get_coverage(), "unsafe")
        return
    elif check_safe(region, props, intervals, silent) == "safe":
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
    foo = copy.copy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.copy(region)
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
        [copy.copy(foo), props, intervals, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue(
        [copy.copy(foo2), props, intervals, n - 1, epsilon, coverage, silent, model_high])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking(*que.dequeue())


def private_check_deeper_queue_checking_both(region, props, intervals, n, epsilon, coverage, silent,
                                             model=None):
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
    if space.get_coverage() > coverage:
        globals()["que"] = Queue()
        return "coverage ", space.get_coverage(), " is above the threshold"

    ## Resolving if the region safe/unsafe/unknown
    if model is None:
        example = check(region, props, intervals, silent)
        counterexample = check_safe(region, props, intervals, silent)
    elif model[0] is None:
        example = check(region, props, intervals, silent)
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
    if example == "unsafe":
        space.remove_white(region)
        if not silent:
            print(n, region, space.get_coverage(), "unsafe")
        return
    elif counterexample == "safe":
        space.remove_white(region)
        if not silent:
            print(n, region, space.get_coverage(), "safe")
        return
    else:  ## unknown
        if not silent:
            print(n, region, space.get_coverage(), (example, counterexample))

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
    globals()["que"].enqueue([copy.copy(foo), props, intervals, n - 1, epsilon, coverage, silent, model_low])
    globals()["que"].enqueue([copy.copy(foo2), props, intervals, n - 1, epsilon, coverage, silent, model_high])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking_both(*que.dequeue())
