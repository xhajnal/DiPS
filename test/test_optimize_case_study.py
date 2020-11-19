import unittest
import warnings

import optimize_case_study as cs
import os
from termcolor import colored

curr_dir = os.path.dirname(__file__)
cwd = os.getcwd()
test = cwd
model_dir = os.path.join(cwd, "models")
data_dir = os.path.join(cwd, "data")


class MyTestCase(unittest.TestCase):
    def test_dist_case_study(self):
        print(colored("test distance of case study", 'blue'))
        cs.params = ["x"]
        point = 0.5
        cs.functions = ["x+9"]
        cs.data_point = [0.9]

        self.assertEqual(cs.dist_l1([point]), abs(9.5 - 0.9))
        self.assertEqual(cs.dist_l2([point]), abs(9.5 - 0.9))

        weight = 8
        self.assertEqual(cs.weighted_dist_l1([point], [weight]), abs(9.5 - 0.9)*weight)
        self.assertEqual(cs.weighted_dist_l2([point], [weight]), (abs(9.5 - 0.9)**2 * weight)**(1/2))

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
        self.assertEqual(cs.weighted_dist_l2([point], [weight1, weight2]), ((abs(9.5 - 0.9))**2 * weight1 + (abs(1 - 9))**2 * weight2)**(1/2))

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
        self.assertEqual(cs.weighted_dist_l2(point, [weight1, weight2]), ((abs(9.5 - 0.9))**2 * weight1 + (abs(0.5 + 4 - 9))**2 * weight2)**(1/2))

    def test_optimize_case_study_multi_param(self):
        print(colored("optimize test with three params four functions", 'blue'))
        warnings.warn("This test contains nondeterministic code, please check the results manually!!", RuntimeWarning)
        data = [6]
        result = cs.optimize_case_study(["z+y+1"], ["z", "y"], [[0, 1], [2, 3]], data, weights=[], sort=False, debug=True)
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
