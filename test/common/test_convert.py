import unittest

import sympy

from src.common.convert import *


class MyTestCase(unittest.TestCase):
    def test_parse_numbers(self):
        print(colored("Checking parsers of numbers from string", 'blue'))
        self.assertEqual(parse_numbers("h3110 23 cat 444.4 rabbit 11 2 dog"), [3110, 23, 444.4, 11, 2])
        self.assertEqual(parse_numbers('hello 42 I\'m a 32 string 30'), [42, 32, 30])
        self.assertEqual(parse_numbers('hello 8e-7 holario'), [8e-7])
        self.assertEqual(parse_numbers('hello8e-7holario'), [8e-7])
        self.assertEqual(parse_numbers("Time for model construction: 0.262 seconds."), [0.262])
        self.assertEqual(parse_numbers('Time for model input parsing: 0.006s.'), [0.006])
        self.assertEqual(parse_numbers('Time for model input parsing: 0.006s .9'), [0.006, 0.9])
        self.assertEqual(parse_numbers('Time for model input parsing: 0.006s.9'), [0.006, 9])

    def test_to_sympy_intervals(self):
        print(colored("Checking conversion from a list of pairs/lists to list of Intervals", 'blue'))
        self.assertEqual(to_sympy_intervals([[2, 3]]), [Interval(2, 3)])
        self.assertEqual(to_sympy_intervals([[2, 3]]), [Interval(2, 3)])
        self.assertEqual(to_sympy_intervals([(2, 3)]), [Interval(2, 3)])
        self.assertEqual(to_sympy_intervals([[2, 3], (4, 5)]), [Interval(2, 3), Interval(4, 5)])
        self.assertEqual(to_sympy_intervals([(2, 3), [4, 5]]), [Interval(2, 3), Interval(4, 5)])
        self.assertEqual(to_sympy_intervals([(2, 3), (4, 5)]), [Interval(2, 3), Interval(4, 5)])

    def test_ineq_to_constraints(self):
        print(colored("Checking conversion from a list of inequalities to list of properties", 'blue'))
        self.assertEqual(ineq_to_constraints(["x+3"], [[0, 1]], decoupled=True), ["x+3 >= 0", "x+3 <= 1"])
        self.assertEqual(ineq_to_constraints(["x", "2*x"], [Interval(0, 1), Interval(0, 2)], decoupled=True), ['x >= 0', 'x <= 1', '2*x >= 0', '2*x <= 2'])

        ## No intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints(["x+3"], [], decoupled=True)
            self.assertTrue("Constraints cannot be computed" in str(context.exception))

        self.assertEqual(ineq_to_constraints(["x+3"], [[0, 1]], decoupled=False), ["0 <= x+3 <= 1"], )
        self.assertEqual(ineq_to_constraints(["x", "2*x"], [Interval(0, 1), Interval(0, 2)], decoupled=False), ['0 <= x <= 1', '0 <= 2*x <= 2'])

        ## No intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints(["x+3"], [], decoupled=False)
            self.assertTrue("Constraints cannot be computed" in str(context.exception))

        ## Bad intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints(["x"], [Interval(3, 2)])
            print(context.exception)
        self.assertTrue('Some intervals are incorrect' in str(context.exception))

        with self.assertRaises(Exception) as context:
            ineq_to_constraints(["x"], [Interval(2, 2)])
        self.assertTrue('Some intervals are incorrect' in str(context.exception))

    def test_ineq_to_constraints_with_sympy(self):
        print(colored("Checking conversion from a list of inequalities to list of properties with sympy", 'blue'))
        self.assertEqual(ineq_to_constraints([sympy.factor("x+3")], [[0, 1]], decoupled=True), [sympy.factor("x+3 >= 0"), sympy.factor("x+3 <= 1")])
        self.assertEqual(ineq_to_constraints([sympy.factor("x"), sympy.factor("2*x")], [Interval(0, 1), Interval(0, 2)], decoupled=True), [sympy.factor('x >= 0'), sympy.factor('x <= 1'), sympy.factor('2*x >= 0'), sympy.factor('2*x <= 2')])

        ## No intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints([sympy.factor("x+3")], [], decoupled=True)
        self.assertTrue("Constraints cannot be computed" in str(context.exception))
        self.assertTrue("does not correspond" in str(context.exception))


        self.assertEqual(ineq_to_constraints([sympy.factor("x+3")], [[0, 1]], decoupled=False), [sympy.factor("x+3 >= 0"), sympy.factor("x+3 <= 1")])

        self.assertEqual(ineq_to_constraints([sympy.factor("x"), sympy.factor("2*x")], [Interval(0, 1), Interval(0, 2)],
                                             decoupled=False), [sympy.factor('x >= 0'), sympy.factor('x <= 1'), sympy.factor('2*x >= 0'), sympy.factor('2*x <= 2')])

        ## No intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints([sympy.factor("x+3")], [], decoupled=False)
        self.assertTrue("Constraints cannot be computed" in str(context.exception))
        self.assertTrue("does not correspond" in str(context.exception))

        ## Bad intervals
        with self.assertRaises(Exception) as context:
            ineq_to_constraints([sympy.factor("x")], [Interval(3, 2)])
            print(context.exception)
        self.assertTrue('Some intervals are incorrect' in str(context.exception))

        with self.assertRaises(Exception) as context:
            ineq_to_constraints([sympy.factor("x")], [Interval(2, 2)])
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

    def test_decouple_constraint(self):
        print(colored("Checking decoupling of constraints", 'blue'))
        with self.assertRaises(Exception) as context:
            decouple_constraint("0.706726790611575 - (r_0 - r_1)**3 + 0.893273209388426")
        self.assertTrue('No' in str(context.exception))

        self.assertEqual(decouple_constraint("0.706726790611575 <= -(r_0 - r_1)**3"), ["0.706726790611575 <= -(r_0 - r_1)**3"])
        self.assertEqual(decouple_constraint("0.706726790611575 <= -(r_0 - r_1)**3 <= 0.893273209388426"), ['0.706726790611575 <= -(r_0 - r_1)**3', '-(r_0 - r_1)**3 <= 0.893273209388426'])
        with self.assertRaises(Exception) as context:
            decouple_constraint("0.706726790611575 <= -(r_0 - r_1)**3 <= 0.893273 = 209388426")
        self.assertTrue('More than two' in str(context.exception))

    def test_decouple_constraints(self):
        print(colored("Checking decoupling of constraints", 'blue'))
        with self.assertRaises(Exception) as context:
            decouple_constraints(["0.706726790611575 - (r_0 - r_1)**3 + 0.893273209388426"])
        self.assertTrue('No' in str(context.exception))

        self.assertEqual(decouple_constraints(["0.706726790611575 <= -(r_0 - r_1)**3"]), ["0.706726790611575 <= -(r_0 - r_1)**3"])
        self.assertEqual(decouple_constraints(["0.706726790611575 <= -(r_0 - r_1)**3 <= 0.893273209388426"]), ['0.706726790611575 <= -(r_0 - r_1)**3', '-(r_0 - r_1)**3 <= 0.893273209388426'])
        with self.assertRaises(Exception) as context:
            decouple_constraints(["0.706726790611575 <= -(r_0 - r_1)**3 <= 0.893273 = 209388426"])
        self.assertTrue('More than two' in str(context.exception))

    def test_add_white_spaces(self):
        print(colored("Checking adding space in between <, = and <=", 'blue'))
        self.assertEqual(add_white_spaces("0.270794145078059 >= p*q >=  0.129205854921941"), "0.270794145078059 >= p*q >= 0.129205854921941")
        self.assertEqual(add_white_spaces("0.270794145078059>=p*q>=0.129205854921941"), "0.270794145078059 >= p*q >= 0.129205854921941")
        self.assertEqual(add_white_spaces("p*q>=0.129205854921941"), "p*q >= 0.129205854921941")
        self.assertEqual(add_white_spaces("0.270794145078059>=p*q"), "0.270794145078059 >= p*q")
        ## TODO add more tests

    def test_normalise_constraint(self):
        print(colored("Checking normalising constraints", 'blue'))
        self.assertEqual(normalise_constraint("0.2 >= p+q/8  >= 0.1"),  "0.1 <= p+q/8 <= 0.2")
        self.assertEqual(normalise_constraint("0.2 >= p+q/8"), "p+q/8 <= 0.2")
        self.assertEqual(normalise_constraint("0.2 >= p+q/8  > 0.1"), "0.1 < p+q/8 <= 0.2")
        self.assertEqual(normalise_constraint("0.2 > p+q/8 >= 0.1"), "0.1 <= p+q/8 < 0.2")
        self.assertEqual(normalise_constraint("0.2 > p+q/8 > 0.1"), "0.1 < p+q/8 < 0.2")
        self.assertEqual(normalise_constraint("0.2 >= p+q/8"), "p+q/8 <= 0.2")
        self.assertEqual(normalise_constraint("0.2 = p+q/8"), "p+q/8 = 0.2")
        ## TODO add more tests

    def test_split_constraints(self):
        print(colored("Checking splitting constraints", 'blue'))

        self.assertEqual(split_constraints(["0.7 <= p+q < 0.8"]), [["0.7", "p+q", "0.8"]])

        self.assertEqual(split_constraints(["0.7 < p+q < 0.8"]), [["0.7", "p+q", "0.8"]])
        self.assertEqual(split_constraints(["0.7 < p+q"]), [["0.7", "p+q", None]])

        self.assertEqual(split_constraints(["0.7 > p+q < 0.8"]), [["0.7", "p+q", "0.8"]])
        self.assertEqual(split_constraints(["0.7 > p+q"]), [["0.7", "p+q", None]])

        self.assertEqual(split_constraints(["0.7 <= p+q < 0.8"]), [["0.7", "p+q", "0.8"]])
        self.assertEqual(split_constraints(["0.7 <= p+q"]), [["0.7", "p+q", None]])

        self.assertEqual(split_constraints(["0.7 >= p+q < 0.8"]), [["0.7", "p+q", "0.8"]])
        self.assertEqual(split_constraints(["0.7 >= p+q"]), [["0.7", "p+q", None]])

        self.assertEqual(split_constraints(["0.7 = p+q < 0.8"]), [["0.7", "p+q", "0.8"]])
        self.assertEqual(split_constraints(["0.7 = p+q"]), [["0.7", "p+q", None]])

    def test_split_constraint(self):
        print(colored("Checking splitting single constraint", 'blue'))
        self.assertEqual(split_constraint("0.7 < p+q < 0.8"), ["0.7", "p+q", "0.8"])
        self.assertEqual(split_constraint("0.7 < p+q"), ["0.7", "p+q", None])
        self.assertEqual(split_constraint("p+q < 0.7"), [None, "p+q", "0.7"])

        self.assertEqual(split_constraint("0.7 > p+q < 0.8"), ["0.7", "p+q", "0.8"])
        self.assertEqual(split_constraint("0.7 > p+q"), ["0.7", "p+q", None])
        self.assertEqual(split_constraint("p+q > 0.7"), [None, "p+q", "0.7"])

        self.assertEqual(split_constraint("0.7 <= p+q < 0.8"), ["0.7", "p+q", "0.8"])
        self.assertEqual(split_constraint("0.7 <= p+q"), ["0.7", "p+q", None])
        self.assertEqual(split_constraint("p+q <= 0.7"), [None, "p+q", "0.7"])

        self.assertEqual(split_constraint("0.7 >= p+q < 0.8"), ["0.7", "p+q", "0.8"])
        self.assertEqual(split_constraint("0.7 >= p+q"), ["0.7", "p+q", None])
        self.assertEqual(split_constraint("p+q >= 0.7"), [None, "p+q", "0.7"])

        self.assertEqual(split_constraint("0.7 = p+q < 0.8"), ["0.7", "p+q", "0.8"])
        self.assertEqual(split_constraint("0.7 = p+q"), ["0.7", "p+q", None])
        self.assertEqual(split_constraint("p+q = 0.7"), [None, "p+q", "0.7"])

        self.assertEqual(split_constraint("p < q"), [None, "p", "q"])
        self.assertEqual(split_constraint("l > m"), [None, "l", "m"])

    def test_parse_interval_bounds(self):
        print(colored("Checking parsing of interval bounds", 'blue'))
        self.assertEqual(parse_interval_bounds("0.7<p"), [[0.7, None]])
        self.assertEqual(parse_interval_bounds("p<0.8"), [[None, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8"), [[0.7, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<p<=0.8"), [[0.7, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<=p<0.8"), [[0.7, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<=p<=0.8"), [[0.7, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8,"), [[0.7, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8;"), [[0.7, 0.8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8, 7<q<8"), [[0.7, 0.8], [7, 8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8,7<q<8"), [[0.7, 0.8], [7, 8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8; 7<q<8"), [[0.7, 0.8], [7, 8]])
        self.assertEqual(parse_interval_bounds("0.7<p<0.8;7<q<8"), [[0.7, 0.8], [7, 8]])

    def test_to_interval(self):
        print(colored("Checking transformation of a set of points into a set of intervals here", 'blue'))
        self.assertEqual(to_interval([[0, 5]]), [[0, 0], [5, 5]])
        self.assertEqual(to_interval([(0, 2), (1, 3)]), [[0, 1], [2, 3]])
        self.assertEqual(to_interval([(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 0), (0, 2, 3), (0, 0, 3), (1, 0, 3), (1, 2, 3)]), [[0, 1], [0, 2], [0, 3]])
        self.assertEqual(to_interval([(0, 2), (1, 3), (4, 5)]), [[0, 4], [2, 5]])
        self.assertEqual(to_interval([(0, 2, 9), (1, 5, 0), (4, 3, 6)]), [[0, 4], [2, 5], [0, 9]])


if __name__ == '__main__':
    unittest.main()
