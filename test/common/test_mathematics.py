import unittest
from termcolor import colored
from src.common.mathematics import *


class MyTestCase(unittest.TestCase):
    def test_sympy_intervals(self):
        print(colored('Sympy Interval sanity check', 'blue'))
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

    def test_Interval(self):
        print(colored("Interval test here", 'blue'))
        self.assertEqual(1.0 in mpi(1, 1), True)
        self.assertEqual(1.0 in mpi(1, 2), True)
        self.assertEqual(1.0 in mpi(0, 1), True)
        self.assertEqual(1.0 in mpi(0, 2), True)

        self.assertEqual(5.0 not in mpi(1, 1), True)
        self.assertEqual(5.0 not in mpi(1, 2), True)
        self.assertEqual(5.0 not in mpi(0, 1), True)
        self.assertEqual(5.0 not in mpi(0, 2), True)

        self.assertEqual(mpi(1, 1) in mpi(1, 1), True)
        self.assertEqual(mpi(1, 1) in mpi(1, 2), True)
        self.assertEqual(mpi(1, 1) in mpi(0, 1), True)
        self.assertEqual(mpi(1, 1) in mpi(0, 2), True)

        self.assertEqual(mpi(1, 2) in mpi(1, 3), True)
        self.assertEqual(mpi(1, 2) in mpi(0, 2), True)
        self.assertEqual(mpi(1, 2) in mpi(0, 3), True)
        self.assertEqual(mpi(1, 2) not in mpi(0, 1), True)
        self.assertEqual(mpi(1, 2) not in mpi(2, 3), True)
        self.assertEqual(mpi(1, 2) not in mpi(1.5, 2), True)

    def test_nCr(self):
        for i in range(10):
            self.assertEqual(nCr(i, 0), 1)
            self.assertEqual(nCr(i, 1), i)
            self.assertEqual(nCr(i, i), 1)
            if not i == 0:
                self.assertEqual(nCr(i, i - 1), i)
        self.assertEqual(nCr(5, 3), 10)
        self.assertEqual(nCr(5, 2), 10)

    def test_catch_data_error(self):
        print(colored('Catching subzero values in data', 'blue'))
        ## upper
        a = {2: [0, 3, 5]}
        catch_data_error(a, 0, 4)
        self.assertEqual(a, {2: [0, 3, 4]})

        ## upper
        a = [0, 3, 5]
        catch_data_error(a, 0, 4)
        self.assertEqual(a, [0, 3, 4])

        ## lower
        a = [-2, 3, 4]
        catch_data_error(a, 0, 4)
        self.assertEqual(a, [0, 3, 4])

        ## both
        a = [-2, 3, 5]
        catch_data_error(a, 0, 4)
        self.assertEqual(a, [0, 3, 4])

        ## none
        a = [0, 3, 4]
        catch_data_error(a, 0, 4)
        self.assertEqual(a, [0, 3, 4])

    def test_create_interval(self):
        print(colored('Single interval computing', 'blue'))
        self.assertEqual(round(create_interval(0.95, 60, 0.5).start, 15), round(Interval(0.365151535478501, 0.634848464521499).start, 15))
        self.assertEqual(round(create_interval(0.95, 60, 0.5).end, 15), round(Interval(0.365151535478501, 0.634848464521499).end, 15))

    def test_create_interval_NEW(self):
        print(colored('Single interval computing NEW', 'blue'))
        self.assertEqual(round(create_interval_NEW(
           samples=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], confidence=0.9).start, 15),
                         round(Interval(0.0826658796286292, 0.317334120371371).start, 15))
        self.assertEqual(round(create_interval_NEW(
           samples=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], confidence=0.9).end, 15),
                         round(Interval(0.0826658796286292, 0.317334120371371).end, 15))

        print(colored('Single interval computing normal', 'blue'))
        self.assertEqual(round(create_interval_NEW(
           samples=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], confidence=0.9, is_normal=True).start, 15),
                         round(Interval(0.0826658796286292, 0.317334120371371).start, 15))
        self.assertEqual(round(create_interval_NEW(
           samples=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], confidence=0.9, is_normal=True).end, 15),
                         round(Interval(0.0826658796286292, 0.317334120371371).end, 15))
        ## TODO the other tests

    def test_create_intervals(self):
        print(colored('Multiple intervals computing', 'blue'))
        self.assertEqual(round(create_intervals(0.95, 60, [0.5])[0].start, 15),
                         round(Interval(0.365151535478501, 0.634848464521499).start, 15))
        self.assertEqual(round(create_intervals(0.95, 60, [0.5])[0].end, 15),
                         round(Interval(0.365151535478501, 0.634848464521499).end, 15))
        # TODO more data points

    def test_margin(self):
        print(colored('Margin/delta computing', 'blue'))
        self.assertEqual(round(margin(0.95, 60, 0.5), 15), round(0.5 - 0.365151535478501, 15))
        # TODO add more examples

    def test_margin_experimental(self):
        print(colored('Margin/delta computing in HSB computing', 'blue'))
        self.assertEqual(round(margin_experimental(0.95, 60, 0.5), 15), round(margin(0.95, 60, 0.5) + 0.005, 15))
        # TODO add more examples

    def test_cartesian_product(self):
        print(colored("Checking cartesian product", 'blue'))
        self.assertEqual(cartesian_product().shape, (0,))
        self.assertEqual(cartesian_product(*[np.array([5])]).shape, (1, 1))
        self.assertEqual(cartesian_product(*[np.array([5]), np.array([50])]).shape, (1, 2))
        self.assertEqual(cartesian_product(*[np.array([5]), np.array([50]), np.array([9])]).shape, (1, 3))

    def test_is_in(self):
        print(colored("Checking if the first region is within the second one here", 'blue'))
        self.assertEqual(is_in([(1, 4)], [(1, 4)]), True)
        self.assertEqual(is_in([(1, 4)], [(0, 5)]), True)
        self.assertEqual(is_in([(1, 4)], [(0, 3)]), False)
        self.assertEqual(is_in([(1, 4)], [(2, 5)]), False)

        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 2), (1, 3)]), True)
        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 3), (1, 4)]), True)
        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 1), (1, 4)]), False)
        self.assertEqual(is_in([(0, 2), (1, 3)], [(0, 2), (1, 2)]), False)

    def test_get_rectangle_volume(self):
        print(colored("get_rectangle_volume tests", 'blue'))
        self.assertEqual(round(get_rectangle_volume([[0.0, 0]]), 1), 0)
        self.assertEqual(round(get_rectangle_volume([[0.0, 0.5]]), 1), 0.5)
        self.assertEqual(round(get_rectangle_volume([[0.0, 0.2], [0, 0.2]]), 2), 0.04)


if __name__ == '__main__':
    unittest.main()
