import sys
import os
from mpmath import mpi
from sympy import Interval

workspace = os.path.dirname(__file__)
sys.path.append(workspace)
from load import find_param


def private_check_deeper_interval(region, props, intervals, n, epsilon, coverage, silent):
    """ Refining the parameter space into safe and unsafe regions
    Parameters
    ----------
    region: array of pairs, low and high bound, defining the parameter space to be refined
    props: array of polynomes
    intervals: array of intervals to constrain properties
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    coverage: coverage threshold to stop computation
    silent: if silent print
    """

    for prop in props:
        mpi(intervals[0].start, intervals[0].end) in (eval(f_multiparam[10][0]))


def check_interval_in(region, props, intervals, silent=False, called=False):
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
    called = True
    if called:
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
    return True


def check_interval_out(region, props, intervals, silent=False, called=False):
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
    called = True
    if called:
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
    return True

import unittest


class TestLoad(unittest.TestCase):
    def test_check_interval_single(self):
        #IS IN
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=False), True)
        self.assertEqual(check_interval_in([(1, 1)], ["x"], [Interval(0, 2)], silent=False, called=False), True)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(0.5, 3)], silent=False, called=False), False)
        self.assertEqual(check_interval_in([(0, 1)], ["x"], [Interval(2, 3)], silent=False, called=False), False)
        self.assertEqual(check_interval_in([(1, 4)], ["x"], [Interval(2, 3)], silent=False, called=False), False)

        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=False, called=False), True)
        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x+y"], [Interval(0, 1)], silent=False, called=False), False)
        ## IS OUT
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(2, 3)], silent=False, called=False), True)
        self.assertEqual(check_interval_out([(1, 1)], ["x"], [Interval(2, 3)], silent=False, called=False), True)
        self.assertEqual(check_interval_out([(0, 1)], ["x"], [Interval(1, 3)], silent=False, called=False), False)
        self.assertEqual(check_interval_out([(0, 3)], ["x"], [Interval(2, 3)], silent=False, called=False), False)
        self.assertEqual(check_interval_out([(1, 4)], ["x"], [Interval(2, 3)], silent=False, called=False), False)

        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(0, 2)], silent=False, called=False), False)
        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x+y"], [Interval(4, 5)], silent=False, called=False), True)

    def test_check_interval_multiple(self):
        ## IS IN
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 2)], silent=False, called=False), True)
        self.assertEqual(
            check_interval_in([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False, called=False), False)
        ## TRICKY
        self.assertEqual(check_interval_in([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=False, called=False), False)

        self.assertEqual(check_interval_in([(0, 1), (0, 1)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False, called=False), True)
        self.assertEqual(
            check_interval_in([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False, called=False),
            False)
        ## IS OUT
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(2, 3), Interval(3, 4)], silent=False, called=False), True)
        self.assertEqual(
            check_interval_out([(0, 1)], ["x", "2*x"], [Interval(0, 1), Interval(0, 1)], silent=False, called=False), False)
        ## TRICKY
        self.assertEqual(check_interval_out([(0, 2)], ["x", "2*x"], [Interval(0, 1)], silent=False, called=False), False)

        self.assertEqual(check_interval_out([(0, 1), (0, 1)], ["x", "y"], [Interval(2, 3), Interval(2, 3)], silent=False, called=False), True)
        self.assertEqual(
            check_interval_out([(0, 1), (0, 2)], ["x", "y"], [Interval(0, 1), Interval(0, 1)], silent=False, called=False),
            False)

if __name__ == "__main__":
    unittest.main()
    # check_interval([(0, 1)], ["x"], [Interval(0, 1)], silent=False, called=False)
    # check_interval([(0, 3)], ["x"], [Interval(0, 2)], silent=False, called=False)
