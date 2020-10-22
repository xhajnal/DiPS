import unittest

from common.files import pickle_load
from src.optimize import *
import os
from termcolor import colored
from load import get_all_f

curr_dir = os.path.dirname(__file__)
cwd = os.getcwd()
cwd = os.getcwd()
test = cwd
model_dir = os.path.join(cwd, "models")
data_dir = os.path.join(cwd, "data")


class MyTestCase(unittest.TestCase):
    def test_optimize_two_param(self):
        print(colored("optimize test with two params three functions", 'blue'))
        functions = get_all_f(os.path.join(cwd, "results/prism_results/asynchronous_2.txt"), "prism", True)
        functions = functions[2]
        print(functions)
        d = pickle_load(os.path.join(data_dir, "data.p"))
        print("data_point", d)
        result = optimize(functions, ["p", "q"], [[0, 1], [0, 1]], d)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])

    def test_optimize_three_functions(self):
        print(colored("optimize test with two params four functions", 'blue'))
        ## RefinedSpace(region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, true_point=False, title=False):
        functions = get_all_f(os.path.join(cwd, "results/prism_results/asynchronous_3.txt"), "prism", True)
        functions = functions[3]
        print(functions)
        d = [0.2, 0.3, 0.4, 0.1]
        print(d)
        result = optimize(functions, ["p", "q"], [[0, 1], [0, 1]], d)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])

    def test_optimize_multi_param(self):
        print(colored("optimize test with three params four functions", 'blue'))
        functions = get_all_f(os.path.join(cwd, "results/prism_results/multiparam_synchronous_3.txt"), "prism", True)
        functions = functions[3]
        print(functions)
        d = [0.2, 0.3, 0.4, 0.1]
        print(d)
        result = optimize(functions, ["p", "q1", "q2"], [[0, 1], [0, 1], [0, 1]], d)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])


if __name__ == "__main__":
    unittest.main()
