import unittest
from src.load import *


class MyTestCase(unittest.TestCase):
    def parse_params_from_model(self):
        model_path = Path(config.get("paths", "models"))
        if os.path.isfile(os.path.join(model_path, "asynchronous_2.pm")):
            self.assertEqual(parse_params_from_model(os.path.join(model_path, "asynchronous_2.pm")), ["p", "q"])

    def test_find_param(self):
        self.assertEqual(find_param("56*4+4**6 +   0.1"), set())
        self.assertEqual(find_param("x+0.1"), {'x'})
        self.assertEqual(find_param("p**2-2*p+1"), {'p'})
        self.assertEqual(find_param("p ** 2 - 2 * p + 1"), {'p'})
        self.assertEqual(find_param("(-2)*q1*p**2+2*q1*p+2*p"), {'p', 'q1'})
        self.assertEqual(find_param('-p*(2*p*If(Or(low<1,1<=high),qmin,qmax)-p-2*If(Or(low<1,1<=high),qmin,qmax))'), {'qmin', 'p', 'low', 'qmax', 'high'})
        self.assertEqual(find_param('10*p*(p - 1)**9*( If ( Or( low < 1 , 1 <= high), qmin, qmax) - 1)**9'), {'p', 'low', "high", "qmin", "qmax"})
        for polynome in ['p**2-2*p+1>=0.255810035160231', 'p**2-2*p+1<=0.453332821982626', '2*q*p**2-2*p**2-2*q*p+2*p>=0.339105082511199', '2*q*p**2-2*p**2-2*q*p+2*p<=0.543752060345944', '(-2)*q*p**2+p**2+2*q*>=0.120019530949760', '(-2)*q*p**2+p**2+2*q*<=0.287980469050240']:
            print(find_param(polynome))

    def test_load_expressions(self):
        ## THIS WILL PASS ONLY AFTER CREATING THE STORM RESULTS
        agents_quantities = [2]
        f_storm = get_f("./asyn*[0-9]_moments.txt", "storm", True, agents_quantities)
        # print(f_storm)
        self.assertFalse(f_storm[2])
        rewards_storm = get_rewards("./asyn*[0-9]_moments.txt", "storm", True, agents_quantities)
        # print(rewards_storm)
        self.assertTrue(rewards_storm[2])

    def test_margins(self):
        ## TODO
        pass

    def test_create_interval(self):
        ## TODO
        pass

    def test_create_intervals(self):
        ## TODO
        pass

    def test_catch_data_error(self):
        ## TODO
        pass

    def test_load_all_data(self):
        ## TODO
        pass

    def test_load_data(self):
        ## TODO
        pass

    def test_to_variance(self):
        ## TODO
        pass

    def test_load_all_functions(self):
        ## TODO
        pass

    def test_mpmath_intervals(self):
        ## Check more here https://docs.sympy.org/0.6.7/modules/mpmath/basics.html
        ## Sanity check test
        from mpmath import mpi  ## Real intervals
        my_interval = mpi(0, 5)
        self.assertEqual(my_interval.a, 0)
        self.assertEqual(my_interval.b, 5)
        self.assertEqual(my_interval.mid, (5+0)/2)
        self.assertEqual(my_interval.delta, abs(0-5))

    def test_sympy_intervals(self):
        ## Check more here https://docs.sympy.org/latest/modules/sets.html
        ## Sanity check test
        my_interval = Interval(0, 5)
        self.assertEqual(my_interval.inf, 0)
        self.assertEqual(my_interval.sup, 5)
        self.assertEqual(my_interval.boundary, {0, 5})
        self.assertEqual(my_interval.contains(2), True)
        self.assertEqual(my_interval.contains(6), False)
        self.assertEqual(my_interval.intersect(Interval(1, 7)), Interval(1, 5))

        self.assertEqual(my_interval.is_disjoint(Interval(1, 2)), False)
        self.assertEqual(my_interval.is_disjoint(Interval(6, 7)), True)

        self.assertEqual(my_interval.is_subset(Interval(0, 1)), False)
        self.assertEqual(my_interval.is_subset(Interval(0, 10)), True)


if __name__ == '__main__':
    unittest.main()
