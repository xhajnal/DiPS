import unittest
from termcolor import colored
from src.common.math import *


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
        # TODO
        pass

    def test_catch_data_error(self):
        # TODO
        pass

    def test_create_intervals(self):
        # TODO
        pass

    def test_create_interval(self):
        # TODO
        pass

    def test_margin(self):
        # TODO
        pass

    def test_margin_experimental(self):
        # TODO
        pass

    def test_cartesian_product(self):
        # TODO
        pass

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


if __name__ == '__main__':
    unittest.main()
