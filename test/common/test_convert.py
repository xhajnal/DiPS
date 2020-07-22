import unittest
from termcolor import colored
from src.common.convert import *


class MyTestCase(unittest.TestCase):
    def test_ineq_to_constraints(self):
        print(colored("Checking conversion from a list of inequalities to list of properties", 'blue'))
        self.assertEqual(ineq_to_constraints(["x+3"], [[0, 1]]), ["x+3>=0", "x+3<=1"])
        self.assertEqual(ineq_to_constraints(["x", "2*x"], [Interval(0, 1), Interval(0, 2)]), ['x>=0', 'x<=1', '2*x>=0', '2*x<=2'])

        self.assertEqual(ineq_to_constraints(["x+3"], []), False)

        ## Bad intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints(["x"], [Interval(3, 2)])
        self.assertTrue('Some intervals are incorrect' in str(context.exception))

        with self.assertRaises(Exception) as context:
            ineq_to_constraints(["x"], [Interval(2, 2)])
        self.assertTrue('Some intervals are incorrect' in str(context.exception))

    def test_constraints_to_ineq(self):
        print(colored("Checking conversion from a list properties to a list of inequalities", 'blue'))
        self.assertEqual(constraints_to_ineq(["x+3>=0", "x+3<=1"]), (["x+3"], [Interval(0, 1)]))
        self.assertEqual(constraints_to_ineq(['x>=0', 'x<=1', '2*x>=0', '2*x<=2']), (["x", "2*x"], [Interval(0, 1), Interval(0, 2)]))

        ## Properties not in a form of inequalities
        # self.assertEqual(constraints_to_ineq(["x+3>=0", "x+4<=1"]), False)
        with self.assertRaises(Exception) as context:
            constraints_to_ineq(["x+3"])
            self.assertTrue("Number of properties is not even, some interval will be invalid" in context.exception)
        with self.assertRaises(Exception) as context:
            constraints_to_ineq(["x+3>=0", "x+4<=1"])
            self.assertTrue("does not have proper number of boundaries" in context.exception)


    def test_to_interval(self):
        print(colored("Checking transformation of a set of points into a set of intervals here", 'blue'))
        self.assertEqual(to_interval([[0, 5]]), [[0, 0], [5, 5]])
        self.assertEqual(to_interval([(0, 2), (1, 3)]), [[0, 1], [2, 3]])
        self.assertEqual(to_interval([(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0), (0, 2, 3), (0, 0, 3), (1, 0, 3), (1, 2, 3)]), [[0, 1], [0, 2], [0, 3]])


if __name__ == '__main__':
    unittest.main()
