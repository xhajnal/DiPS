import socket
import sys
import os
import re
import copy
from collections import Iterable
import time
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpmath import mpi
from sympy import Interval
from numpy import prod

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param


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

def check_interval_in(region, props, intervals, silent=False, called=True):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    props: array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: array of intervals to constrain properties
    silent: if silent printed output is set to minimum
    called: if called updates the global variables (use when calling it directly)
    """

    if called:
        globals()["rectangles_sat"] = []
        globals()["parameters"] = set()
        for polynome in props:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        if not len(globals()["parameters"]) == len(region) and not silent:
            print("number of parameters in property ({}) and dimension of the region ({}) is not equal".format(
                len(globals()["parameters"]), len(region)))

    ## assign each parameter its interval
    i = 0
    for param in globals()["parameters"]:
        globals()[param] = mpi(region[i][0], region[i][1])
        i = i + 1

    ## check that all prop are in its interval
    i = 0
    for prop in props:
        # print(eval(prop))
        # print(intervals[i])
        # print((intervals[i].start, intervals[i].end))
        # print(mpi(0,1) in mpi(0,2))
        if not eval(prop) in mpi(float(intervals[i].start), float(intervals[i].end)):
            return False
        i = i + 1

    #print("region ",region, "sat, adding it to sat")
    if len(region) == 2:  # if two-dim param space
        globals()["rectangles_sat"].append(
            Rectangle((region[0][0], region[1][0]), region[0][1] - region[0][0], region[1][1] - region[1][0], fc='g'))
    if len(region) == 1:  # if one-dim param space
        globals()["rectangles_sat"].append(
            Rectangle((region[0][0], 0.33), region[0][1] - region[0][0], 0.33, fc='g'))
    #print(globals()["rectangles_sat"])
    return True


def check_interval_out(region, props, intervals, silent=False, called=True):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    props: array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals: array of intervals to constrain properties
    silent: if silent printed output is set to minimum
    called: if called updates the global variables (use when calling it directly)
    """

    if called:
        globals()["rectangles_unsat"] = []
        globals()["parameters"] = set()
        for polynome in props:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        if not len(globals()["parameters"]) == len(region) and not silent:
            print("number of parameters in property ({}) and dimension of the region ({}) is not equal".format(
                len(globals()["parameters"]), len(region)))

    ## assign each parameter its interval
    i = 0
    for param in globals()["parameters"]:
        globals()[param] = mpi(region[i][0], region[i][1])
        i = i + 1

    ## check that all prop are in its interval
    i = 0
    for prop in props:
        # print(eval(prop))
        # print(intervals[i])
        # print((intervals[i].start, intervals[i].end))
        # print(mpi(0,1) in mpi(0,2))
        prop_eval = eval(prop)
        interval = mpi(float(intervals[i].start), float(intervals[i].end))
        ## if there exists an intersection (neither of these interval is greater in all points)
        if not (prop_eval > interval or prop_eval < interval):
            return False
        i = i + 1
    #print("region ", region, "unsat, adding it to unsat")
    if len(region) == 2:  # if two-dim param space
        globals()["rectangles_unsat"].append(
            Rectangle((region[0][0], region[1][0]), region[0][1] - region[0][0], region[1][1] - region[1][0], fc='r'))
    if len(region) == 1:  # if one-dim param space
        globals()["rectangles_unsat"].append(
            Rectangle((region[0][0], 0.33), region[0][1] - region[0][0], 0.33, fc='r'))
    #print(globals()["rectangles_unsat"])
    return True

def check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent, version):
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


    if not len(globals()["parameters"]) == len(region) and not silent:
        print("number of parameters in property ({}) and dimension of the region ({}) is not equal".format(
            len(globals()["parameters"]), len(region)))

    start_time = time.time()
    print("Using interval method")
    globals()["que"] = Queue()
    private_check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent)

    #print("computed with results:")
    #print(globals()["rectangles_sat"])
    #print(globals()["rectangles_unsat"])


    ## Visualisation
    if len(region) == 1 or len(region) == 2:

        ##### UNCOMMENT THIS
        # colored(globals()["default_region"], region)

        # from matplotlib import rcParams
        # rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
        fig = plt.figure()
        pic = fig.add_subplot(111, aspect='equal')
        pic.set_xlabel(globals()["parameters"][0])
        pic.axis([region[0][0], region[0][1], 0, 1])
        if len(region) == 2:
            pic.set_ylabel(globals()["parameters"][1])
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

def private_check_deeper_interval(region, props, intervals, n, epsilon, coverage, silent, model=None):
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

    if check_interval_out(region, props, intervals, silent, called=False):
        result = "unsafe"
    elif check_interval_in(region, props, intervals, silent, called=False):
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
    globals()["que"].enqueue([copy.copy(foo), props, intervals, n - 1, epsilon, coverage, silent])
    globals()["que"].enqueue([copy.copy(foo2), props, intervals, n - 1, epsilon, coverage, silent])

    # CALL QUEUE
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_interval(*que.dequeue())


import unittest


class TestLoad(unittest.TestCase):
    def test_check_interval_single(self):
        # IS IN
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=True), True)
        self.assertEqual(check_interval_in([(1, 1)], ["x"], [Interval(0, 2)], silent=False, called=True), True)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0.5, 3)], silent=False, called=True), False)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(2, 3)], silent=False, called=True), False)
        self.assertEqual(check_interval_in([(1, 4)], ["x"], [Interval(2, 3)], silent=False, called=True), False)

        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True, called=True),
                         True)
        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], silent=True, called=True),
                         False)
        ## IS OUT
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(2, 3)], silent=False, called=True), True)
        self.assertEqual(check_interval_out([(1, 1)], ["x"], [Interval(2, 3)], silent=False, called=True), True)
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(1, 3)], silent=False, called=True), False)
        self.assertEqual(check_interval_out([(0, 3)], ["x"], [Interval(2, 3)], silent=False, called=True), False)
        self.assertEqual(check_interval_out([(1, 4)], ["x"], [Interval(2, 3)], silent=False, called=True), False)

        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=False, called=True),
                         False)
        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(4, 5)], silent=False, called=True),
                         True)

    def test_check_interval_multiple(self):
        ## IS IN
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 2)], silent=False, called=True),
            True)
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False, called=True),
            False)
        ## TRICKY
        self.assertEqual(check_interval_in([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=False, called=True), False)

        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False,
                                           called=True), True)
        self.assertEqual(
            check_interval_in([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False,
                              called=True), False)
        ## IS OUT
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(2, 3), Interval(3, 4)], silent=False, called=True),
            True)
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False, called=True),
            False)
        ## TRICKY
        self.assertEqual(check_interval_out([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=False, called=True),
                         False)

        self.assertEqual(
            check_interval_out([(0, 1), (0, 1)], ["x", "y"], [Interval(2, 3), Interval(2, 3)], silent=False,
                               called=True), True)
        self.assertEqual(
            check_interval_out([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False,
                               called=True),
            False)

    def test_check_interval_deeper(self):
        ## check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent, version)
        check_deeper_interval([(0, 4)], ["x"], [Interval(0, 3)], 5, 0, 0.95, silent=False, version=1)
        #print(globals()["rectangles_unsat"])
        #print(globals()["rectangles_sat"])
        #self.assertEqual(check_deeper_interval([(0, 1)], ["x"], [Interval(0, 1)], 0, 0, 0.95, silent=False, version=1), True)



if __name__ == "__main__":

    unittest.main()
    # check_interval([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=False)
    # check_interval([(0, 3)], ["x"], [Interval(0, 2)], silent=False, called=False)
