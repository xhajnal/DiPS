import unittest
import warnings

from common.files import pickle_load
from common.mathematics import weight_list
import optimize
import os
from termcolor import colored
from load import get_all_f

curr_dir = os.path.dirname(__file__)
cwd = os.getcwd()
test = cwd
model_dir = os.path.join(cwd, "models")
data_dir = os.path.join(cwd, "data")


class MyTestCase(unittest.TestCase):
    def test_dist(self):
        print(colored("test distance", 'blue'))

        optimize.params = ["x"]
        optimize.functions = ["x*2"]
        optimize.data_point = [10]
        ## abs(10 - 2*2) is 6
        self.assertEqual(optimize.dist([2]), [6])

        optimize.functions = ["x*2", "x*5"]
        optimize.data_point = [10, 5]
        ## abs(10 - 2*2) is 6
        ## abs(5 - 2*5) is 5
        self.assertEqual(list(optimize.dist([2])), [6, 5])

        optimize.params = ["x", "y"]
        optimize.functions = ["x*2 + y", "x*5 + y"]
        optimize.data_point = [10, 5]
        ## abs(10 - (2*2 +1)) is abs(10 - (5)) which is 5
        ## abs(5 - (2*5 + 1) is abs(5 - (11)) which is 6
        self.assertEqual(list(optimize.dist([2, 1])), [5, 6])

    def test_dist_case_study(self):
        print(colored("test distance of case study", 'blue'))
        cs.params = ["x"]
        point = 0.5
        cs.functions = ["x+9"]
        cs.data_point = [0.9]

        self.assertEqual(cs.dist_l1([point]), abs(9.5-0.9))
        self.assertEqual(cs.dist_l2([point]), abs(9.5 - 0.9))

        weight = 8
        self.assertEqual(cs.weighted_dist_l1([point], [weight]), abs(9.5 - 0.9)*weight)
        self.assertEqual(cs.weighted_dist_l2([point], [weight]), abs(9.5 - 0.9)*weight)

    def test_dist_multi_case_study(self):
        print(colored("test distance of case study", 'blue'))
        cs.params = ["x"]
        point = 0.5
        cs.functions = ["x+9", "x*2"]
        cs.data_point = [0.9, 9]

        self.assertEqual(cs.dist_l1([point]), abs(9.5 - 0.9) + abs(1 - 9))
        self.assertEqual(cs.dist_l2([point]), (abs(9.5 - 0.9)**2 + abs(1 - 9)**2)**(1/2))

        weight1 = 8
        weight2 = 2
        self.assertEqual(cs.weighted_dist_l1([point], [weight1, weight2]), (abs(9.5 - 0.9)*weight1) + abs(1 - 9)*weight2)
        self.assertEqual(cs.weighted_dist_l2([point], [weight1, weight2]), ((abs(9.5 - 0.9)*weight1)**2 + (abs(1 - 9)*weight2)**2)**(1/2))

    def test_dist_multi_multi_case_study(self):
        print(colored("test distance of case study", 'blue'))
        cs.params = ["x", "y"]
        point = [0.5, 2]
        cs.functions = ["x+9", "x + y*2"]
        cs.data_point = [0.9, 9]

        self.assertEqual(cs.dist_l1(point), abs(9.5 - 0.9) + abs(0.5 + 4 - 9))
        self.assertEqual(cs.dist_l2(point), (abs(9.5 - 0.9)**2 + abs(0.5 + 4 - 9)**2)**(1/2))

        weight1 = 8
        weight2 = 2
        self.assertEqual(cs.weighted_dist_l1(point, [weight1, weight2]), (abs(9.5 - 0.9)*weight1) + abs(0.5 + 4 - 9)*weight2)
        self.assertEqual(cs.weighted_dist_l2(point, [weight1, weight2]), ((abs(9.5 - 0.9)*weight1)**2 + (abs(0.5 + 4 - 9)*weight2)**2)**(1/2))

    def test_weighted_distance(self):
        print(colored("test weighted distance", 'blue'))
        weights = [0.5, 3]

        ## Second weight not used
        optimize.params = ["x"]
        optimize.functions = ["x*2"]
        optimize.data_point = [10]
        ## abs(10 - 2*2)*0.5 is 3
        self.assertEqual(weight_list(optimize.dist([2]), weights), [3])

        optimize.functions = ["x*2", "x*5"]
        optimize.data_point = [10, 5]
        ## abs(10 - 2*2)*0.5 is 3
        ## abs(5 - 2*5)*3 is 15
        self.assertEqual(list(weight_list(optimize.dist([2]), weights)), [3, 15])

        optimize.params = ["x", "y"]
        optimize.functions = ["x*2 + y", "x*5 + y"]
        optimize.data_point = [10, 5]
        ## abs(10 - (2*2 +1))*0.5 is abs(10 - (5))*0.5 which is 5*0.5 = 2.5
        ## abs(5 - (2*5 + 1)*3 is abs(5 - (11))*3 which is 6*3 = 18
        self.assertEqual(list(weight_list(optimize.dist([2, 1]), weights)), [2.5, 18])

        weights = [0.5]
        ## Second weight missing, used 1 instead
        optimize.params = ["x", "y"]
        optimize.functions = ["x*2 + y", "x*5 + y"]
        optimize.data_point = [10, 5]
        ## abs(10 - (2*2 +1))*0.5 is abs(10 - (5))*0.5 which is 5*0.5 = 2.5
        ## abs(5 - (2*5 + 1)*3 is abs(5 - (11))*1 which is 6*1 = 18
        self.assertEqual(list(weight_list(optimize.dist([2, 1]), weights)), [2.5, 6])

    def test_optimize_two_param(self):
        print(colored("optimize test with two params three functions", 'blue'))
        warnings.warn("This test contains nondeterministic code, please check the results manually!!", RuntimeWarning)
        functions = get_all_f(os.path.join(cwd, "results/prism_results/asynchronous_2.txt"), "prism", True)
        functions = functions[2]
        print("functions", functions)
        d = pickle_load(os.path.join(data_dir, "data.p"))
        print("data_point", d)
        result = optimize.optimize(functions, ["p", "q"], [[0, 1], [0, 1]], d)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])

    def test_optimize_three_functions(self):
        print(colored("optimize test with two params four functions", 'blue'))
        warnings.warn("This test contains nondeterministic code, please check the results manually!!", RuntimeWarning)
        ## RefinedSpace(region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, true_point=False, title=False):
        functions = get_all_f(os.path.join(cwd, "results/prism_results/asynchronous_3.txt"), "prism", True)
        functions = functions[3]
        print("functions", functions)
        d = [0.2, 0.3, 0.4, 0.1]
        print(d)
        result = optimize.optimize(functions, ["p", "q"], [[0, 1], [0, 1]], d)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])

    def test_optimize_multi_param(self):
        print(colored("optimize test with three params four functions", 'blue'))
        warnings.warn("This test contains nondeterministic code, please check the results manually!!", RuntimeWarning)
        functions = get_all_f(os.path.join(cwd, "results/prism_results/multiparam_synchronous_3.txt"), "prism", True)
        functions = functions[3]
        print(functions)
        d = [0.2, 0.3, 0.4, 0.1]
        print(d)
        result = optimize.optimize(functions, ["p", "q1", "q2"], [[0, 1], [0, 1], [0, 1]], d)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])

    def test_optimize_case_study_multi_param(self):
        print(colored("optimize test with three params four functions", 'blue'))
        warnings.warn("This test contains nondeterministic code, please check the results manually!!", RuntimeWarning)
        data = [6]
        result = cs.optimize_case_study(["z+y+1"], ["z", "y"], [[0, 1], [2, 3]], data, weights=False, sort=False, debug=True)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])
        ## function + distance is data in absolute value
        self.assertEqual(abs(result[1][0]) + result[2], abs(data[0]))

    def test_optimize_case_study_multi_param_with_weights(self):
        print(colored("optimize test with three params four functions", 'blue'))
        warnings.warn("This test contains nondeterministic code, please check the results manually!!", RuntimeWarning)
        data = 6
        result = cs.optimize_case_study(["z+y+1"], ["z", "y"], [[0, 1], [2, 3]], [data], weights=[0.5], sort=False, debug=True)
        print("parameter point", result[0])
        print("function values", result[1])
        print("distance", result[2])
        ## function + distance is data in absolute value
        # self.assertEqual(abs(result[1]) + result[2], abs(data))


if __name__ == "__main__":
    unittest.main()
