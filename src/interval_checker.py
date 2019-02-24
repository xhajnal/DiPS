import socket
import sys
import os
import copy
import time
from matplotlib.patches import Rectangle
from mpmath import mpi
from sympy import Interval

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param
from space import RefinedSpace
from space import get_rectangle_volume


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
        print("CALLED")
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        parameters = sorted(list(parameters))
        space = RefinedSpace(copy.copy(region), parameters, [], [])
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
        # print((intervals[i].start, intervals[i].end))
        # print(mpi(0,1) in mpi(0,2))
        if not eval(prop) in mpi(float(intervals[i].start), float(intervals[i].end)):
            return False
        i = i + 1

    space.add_green(region)
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
        print("CALLED")
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynome in props:
            parameters.update(find_param(polynome))
        parameters = sorted(list(parameters))
        space = RefinedSpace(copy.copy(region), parameters, [], [])
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
        # print((intervals[i].start, intervals[i].end))
        # print(mpi(0,1) in mpi(0,2))
        prop_eval = eval(prop)
        interval = mpi(float(intervals[i].start), float(intervals[i].end))
        ## If there exists an intersection (neither of these interval is greater in all points)
        if not (prop_eval > interval or prop_eval < interval):
            return False
        i = i + 1
    # print("region ", region, "unsat, adding it to unsat")
    space.add_red(region)
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

    globals()["parameters"] = set()
    for polynome in prop:
        globals()["parameters"].update(find_param(polynome))
    globals()["parameters"] = sorted(list(globals()["parameters"]))
    parameters = globals()["parameters"]

    globals()["space"] = RefinedSpace(copy.copy(region), parameters, [], [])
    space = globals()["space"]

    globals()["default_region"] = copy.copy(region)

    if not silent:
        print("the area is: ", space.region)
        print("the volume of the whole area is:", space.get_volume())

    start_time = time.time()
    version = "-interval"
    print("Using interval method")
    globals()["que"] = Queue()
    private_check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent)

    # print("computed with results:")
    # print(globals()["rectangles_sat"])
    # print(globals()["rectangles_unsat"])

    ## Visualisation
    space.show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version} \n It took {socket.gethostname()} {round(time.time() - start_time)} second(s)")

    # print("result coverage is: ", space.get_coverage())
    print("result coverage is: ", space.get_coverage())
    # return (globals()["hyper_rectangles_sat"], globals()["hyper_rectangles_unsat"], globals()["hyper_rectangles_white"],
    #        space.get_coverage())
    # return (space.sat, space.unsat, space.unknown, space.get_coverage())
    return space


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

    ## Resolve the result
    if check_interval_out(region, props, intervals, silent, called=False):
        result = "unsafe"
    elif check_interval_in(region, props, intervals, silent, called=False):
        result = "safe"
    else:
        result = "unknown"

    if result == "safe" or result == "unsafe":
        # print("removing region:", region)
        space.remove_white(region)

    if not silent:
        print(n, region, space.get_coverage(), result)

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

    ## Execute the queue
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
        # check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent, version)
        check_deeper_interval([(0, 4)], ["x"], [Interval(0, 3)], 5, 0, 0.95, silent=False, version=1)
        # print(globals()["rectangles_unsat"])
        # print(globals()["rectangles_sat"])
        # self.assertEqual(check_deeper_interval([(0, 1)], ["x"], [Interval(0, 1)], 0, 0, 0.95, silent=False, version=1), True)


if __name__ == "__main__":
    unittest.main()
    # check_interval([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=False)
    # check_interval([(0, 3)], ["x"], [Interval(0, 2)], silent=False, called=False)
