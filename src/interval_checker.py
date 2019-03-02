import socket
import sys
import os
import copy
import time
from matplotlib.patches import Rectangle
from mpmath import mpi
from numpy import linspace, newaxis, zeros, nditer, asarray, ones, array, place, indices
from sympy import Interval
import unittest

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param
from space import RefinedSpace
from space import get_rectangle_volume
from sample_n_visualise import cartesian_product


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


def create_matrix(size_q, dim):
    """ Return **dim** dimensional array of length **size_q** in each dimension

       Args
       -------
       size_q (int): number of samples in dimension
       dim (int): number of dimensions

    """
    return private_create_matrix(size_q, dim, dim)


def private_create_matrix(size_q, dim, n_param):
    """ Return **dim** dimensional array of length **size_q** in each dimension

       Args
       -------
       size_q (int): number of samples in dimension
       dim (int):number of dimensions
       n_param (int): dummy parameter

       @author: xtrojak, xhajnal
    """
    if dim == 0:
        point = []
        for i in range(n_param):
            point.append(0)
        return [point, 9]
    return [private_create_matrix(size_q, dim-1, n_param) for _ in range(size_q)]


def sample(space, props, intervals, size_q, compress=False, silent=True):
    """ Samples the space in **size_q** samples in each dimension and saves if the point is in respective interval

    Args
    -------
    space (space.RefinedSpace): space
    props (list of strings): array of functions (polynomes or general rational functions in the case of Markov Chains)
    intervals (list of sympy.Interval): array of intervals to constrain properties
    size_q (int): number of samples in dimension

    Returns
    --------
    A map from point to list of Bool whether f(point) in interval[index]
    """

    parameter_values = []
    parameter_indices = []
    for param in range(len(space.params)):
        parameter_values.append(linspace(space.region[param][0], space.region[param][1], size_q, endpoint=True))
        parameter_indices.append(asarray(range(0, size_q)))

    sampling = array(create_matrix(size_q, len(space.params)))
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
        #sampling[0][0] = [[0.], [True]]

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
    sampled_space (space.RefinedSpace): space

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
        print(sampled_space.shape)
        print("dimensions:", dimensions)
    # find_max_rectangle(sampled_space, [0, 0])

    if dimensions == 2:
        parameter_indices = []
        for param in range(dimensions):
            parameter_indices.append(asarray(range(0, size_q)))
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
    sampled_space (space.RefinedSpace): space
    starting_point (list of floats): a point in the space to start search in

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
            print("dealing with 2D space at starting point", starting_point, "and start value", start_value)
        if start_value == 2:
            if not silent:
                print(starting_point, "already added, skipping")
            return
        if index_x >= size_q-1 or index_y >= size_q-1:
            if not silent:
                print(starting_point, "is at the border, skipping")
            sampled_space[index_x][index_y][1] = 2
            return
        ## print(start_value)

        ## While other value is found
        while True:
            ## print(index_x+length)
            ## print(sampled_space[index_x:index_x+length, index_y:index_y+length])
            values = list(map(lambda x: [y[1] for y in x], sampled_space[index_x:index_x+length, index_y:index_y+length]))
            ## print(values)
            foo = []
            for x in values:
                for y in x:
                    foo.append(y)
            values = foo
            if not silent:
                print(values)
            if (not start_value) in values:
                length = length - 1
                if not silent:
                    print(f"rectangle [[{index_x},{index_y}],[{index_x+length},{index_y+length}]] does not satisfy all sat not all unsat")
                sampled_space[index_x][index_y][1] = 2
                break
            elif index_x+length > size_q or index_y+length > size_q:
                if not silent:
                    print(f"rectangle [[{index_x},{index_y}],[{index_x+length},{index_y+length}]] is out of box, using lower value")
                sampled_space[index_x][index_y][1] = 2
                length = length - 1
                break
            else:
                length = length + 1
        #sampled_space[index_x][index_y][1] = 2
        length = length - 1
        if length == 0:
            if not silent:
                print("Only single point found, skipping")
            return
        ## print((sampled_space[index_x, index_y], sampled_space[index_x+length-2, index_y+length-2]))

        # print(type(sampled_space))
        # place(sampled_space, sampled_space==False, 2)
        # print("new sampled_space: \n", sampled_space)

        ## Mark as seen
        # print("the space to be marked: \n", sampled_space[index_x:(index_x + length - 1), index_y:(index_y + length - 1)])
        if not silent:
            print("length", length)

        ## old result
        # result = (sampled_space[index_x, index_y], sampled_space[index_x + length - 1, index_y + length - 1])
        ## new result
        result = ([[sampled_space[index_x, index_y][0][0], sampled_space[index_x + length, index_y][0][0]],
                   [sampled_space[index_x, index_y][0][1], sampled_space[index_x, index_y + length][0][1]]])
        print(f"adding rectangle [[{index_x},{index_y}],[{index_x+length},{index_y+length}]] with value [{sampled_space[index_x, index_y][0]},{sampled_space[index_x + length, index_y + length][0]}]")

        ## OLD setting seen
        #place(sampled_space[index_x:(index_x + length), index_y:(index_y + length)],
        #      sampled_space[index_x:(index_x + length), index_y:(index_y + length)] == False, 2)
        #place(sampled_space[index_x:(index_x + length), index_y:(index_y + length)],
        #      sampled_space[index_x:(index_x + length), index_y:(index_y + length)] == True, 2)

        print("new sampled_space: \n", sampled_space)
        ## globals()["que"].enqueue([[index_x, index_x+length-2],[index_y, index_y+length-2]],start_value)
        return result
    else:
        print(f"Sorry, {dimensions} dimensions TBD")


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
    if version == 1:
        version = "-interval"
        print("Using interval method")
        globals()["que"] = Queue()
        private_check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent)

    ## TBD check the rest of the space
    elif version == "presampled":
        print("Using presampled interval method")
        #globals()["space"] = RefinedSpace(copy.copy(region), parameters, [], [])

        to_be_searched = sample(space, prop, intervals, 5, compress=True, silent=False)
        #to_be_searched = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], , compress=True, silent=False)
        to_be_searched = refine_into_rectangles(to_be_searched, silent=False)
        print("to_be_searched: ", to_be_searched)
        globals()["que"] = Queue()

        for rectangle in to_be_searched:
            print(rectangle)
            #print("sat", space.sat)
            print("unsat", space.unsat)
            space.add_white(rectangle)
            private_check_deeper_interval(rectangle, prop, intervals, 0, epsilon, cov, silent)
            check_interval_out(rectangle, prop, intervals, called=False)
            # check_interval_in(rectangle, prop, intervals, called=False)
            #print("sat", space.sat)
            ## globals()["que"].enqueue([[(0, 0.5), (0, 0.5)], prop, intervals, 0, epsilon, cov, silent])

        # private_check_deeper_interval(*que.dequeue())

    # print("computed with results:")
    # print(globals()["rectangles_sat"])
    # print(globals()["rectangles_unsat"])

    ## Visualisation
    #print("sat here", space.sat)
    print("unsat here", space.unsat)
    # print("sat here", globals()["space"].sat)
    globals()["space"].show(f"max_recursion_depth:{n},\n min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())},"
               f" alg{version} \n It took {socket.gethostname()} {round(time.time() - start_time)} second(s)")

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


def private_check_deeper_interval(region, props, intervals, n, epsilon, coverage, silent, model=None, presampled=False):
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
    #if presampled:
    #    while globals()["que"].size() > 0:
    #        private_check_deeper_interval(*que.dequeue())
    #    return

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


class TestLoad(unittest.TestCase):
    def test_check_interval_single(self):
        # IS IN
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0, 1)], silent=False), True)
        self.assertEqual(check_interval_in([(1, 1)], ["x"], [Interval(0, 2)], silent=False), True)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0.5, 3)], silent=False), False)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(2, 3)], silent=False), False)
        self.assertEqual(check_interval_in([(1, 4)], ["x"], [Interval(2, 3)], silent=False), False)

        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=True),
                         True)
        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], silent=True),
                         False)
        ## IS OUT
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(2, 3)], silent=False), True)
        self.assertEqual(check_interval_out([(1, 1)], ["x"], [Interval(2, 3)], silent=False), True)
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(1, 3)], silent=False), False)
        self.assertEqual(check_interval_out([(0, 3)], ["x"], [Interval(2, 3)], silent=False), False)
        self.assertEqual(check_interval_out([(1, 4)], ["x"], [Interval(2, 3)], silent=False), False)

        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=False),
                         False)
        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(4, 5)], silent=False),
                         True)

    def test_check_interval_multiple(self):
        ## IS IN
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 2)], silent=False),
            True)
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False),
            False)
        ## TRICKY
        self.assertEqual(check_interval_in([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=False), False)

        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False,
                                           called=True), True)
        self.assertEqual(
            check_interval_in([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False,
                              called=True), False)
        ## IS OUT
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(2, 3), Interval(3, 4)], silent=False),
            True)
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False),
            False)
        ## TRICKY
        self.assertEqual(check_interval_out([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=False),
                         False)

        self.assertEqual(
            check_interval_out([(0, 1), (0, 1)], ["x", "y"], [Interval(2, 3), Interval(2, 3)], silent=False,
                               called=True), True)
        self.assertEqual(
            check_interval_out([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False,
                               called=True),
            False)

    def test_check_interval_deeper(self):
        print()
        ## check_deeper_interval(region, prop, intervals, n, epsilon, cov, silent, version)

        ## UNCOMENT THIS TBA
        #check_deeper_interval([(0, 4)], ["x"], [Interval(0, 3)], 5, 0, 0.95, silent=False, version=1)



        # print(globals()["rectangles_unsat"])
        # print(globals()["rectangles_sat"])
        # self.assertEqual(check_deeper_interval([(0, 1)], ["x"], [Interval(0, 1)], 0, 0, 0.95, silent=False, version=1), True)

    def test_sample(self):
        # print("hello")
        print("sample")
        print(0.0 in mpi(0, 0))
        ## def sample(space, props, intervals, size_q)
        # print(sample(RefinedSpace((0, 1), ["x"]), ["x"], [Interval(0, 1)], 3))
        # print(sample(RefinedSpace((0, 1), ["x"]), ["x"], [Interval(0, 1)], 3, compress=True))

        # print(sample(RefinedSpace((0, 2), ["x"]), ["x"], [Interval(0, 1)], 3))
        # print(sample(RefinedSpace((0, 2), ["x"]), ["x"], [Interval(0, 1)], 3, compress=True))

        #sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y"], [Interval(0, 1)], 3, compress=True)

        #a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y"], [Interval(0, 1)], 2, compress=True)
        #print(a)
        #refine_into_rectangles(a)

        #a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], 5,
        #            compress=True, silent=False)
        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 0.9), Interval(0, 1)], 3, compress=True)


        #a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y", "0"], [Interval(0, 1), Interval(0, 1)], 2)
        # a = sample(RefinedSpace([(0, 1), (0, 1)], ["x", "y"]), ["x+y"], [Interval(0, 1)], 2, compress=True)

        #print("result")
        #print(a)

        #b = refine_into_rectangles(a, silent=False)

        ## UNCOMENT THIS
        #check_deeper_interval([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 12, 0, 0.95, silent=False, version=1)
        check_deeper_interval([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], 5, 0, 0.95, silent=False, version="presampled")

        #check_deeper_interval([(0, 0.5), (0, 0.5)], ["x+y"], [Interval(0, 1)], 5, 0, 0.95, silent=False, version=1)


        # b = refine_into_rectangles(a)
        # print(b)

        # a = sample(RefinedSpace([(0, 1), (0, 1), (0, 1)], ["x", "y", "z"]), ["x+y"], [Interval(0, 1)], 3, compress=True)
        # print(a)
        # b = refine_into_rectangles(a)

if __name__ == "__main__":
    unittest.main()
    # check_interval([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=False)
    # check_interval([(0, 3)], ["x"], [Interval(0, 2)], silent=False, called=False)
