import unittest
from termcolor import colored

from common.space_stuff import *


class MyTestCase(unittest.TestCase):
    def test_rectangular_hull(self):
        print(colored("Checking rectangular hull", 'blue'))
        self.assertEqual(rectangular_hull([[1, 9], [2, 4], [7, 3], [4, 5], [2, 7]]), [[1, 7], [3, 9]])

    def test_points_to_rectangle(self):
        print(colored("Checking converting points to hyperectangle", 'blue'))
        self.assertEqual(points_to_rectangle([[1, 2], [4, 5]]), [[1, 4], [2, 5]])

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
        self.assertEqual(get_rectangle_volume([[0.0, 0]]), 0)
        self.assertEqual(get_rectangle_volume([[0.0, 0.5]]), Fraction(1, 2))
        self.assertEqual(get_rectangle_volume([[0.0, 0.2], [0, 0.2]]), Fraction(4, 100))
        self.assertEqual(get_rectangle_volume([[0.7, 0.8], [8.7, 8.8]]), Fraction(1, 100))

    def test_expand_rectangle(self):
        print(colored("Checking expanding rectangle withing sampling", 'blue'))
        self.assertEqual(expand_rectangle([[0.75, 0.75], [1.0, 1.0]], [[0.5, 0.75], [0.5, 1]], [2, 2]), [[0.625, 0.75], [0.75, 1.0]])

    def test_split_by_longest_dimension(self):
        print(colored("Checking rectangle splicing into two by the longest dimension", 'blue'))
        self.assertEqual(split_by_longest_dimension([[3, 4]])[:2], [[[3, 3.5]], [[3.5, 4]]])
        self.assertEqual(split_by_longest_dimension([[3, 4], [5, 6]])[:2], [[[3, 3.5], [5, 6]], [[3.5, 4], [5, 6]]])
        self.assertEqual(split_by_longest_dimension([[3, 4], [5, 6], [7, 8]])[:2], [[[3, 3.5], [5, 6], [7, 8]], [[3.5, 4], [5, 6], [7, 8]]])

        self.assertEqual(split_by_longest_dimension([[3, 4]]), [[[3, 3.5]], [[3.5, 4]], 0, 3.5])
        self.assertEqual(split_by_longest_dimension([[3, 4], [5, 6]]), [[[3, 3.5], [5, 6]], [[3.5, 4], [5, 6]], 0, 3.5])
        self.assertEqual(split_by_longest_dimension([[3, 4], [5, 6], [7, 8]]), [[[3, 3.5], [5, 6], [7, 8]], [[3.5, 4], [5, 6], [7, 8]], 0, 3.5])

        self.assertEqual(split_by_longest_dimension([[3, 4]]), [[[3, 3.5]], [[3.5, 4]], 0, 3.5])
        self.assertEqual(split_by_longest_dimension([[3, 4], [5, 7]]), [[[3, 4], [5, 6]], [[3, 4], [6, 7]], 1, 6])
        self.assertEqual(split_by_longest_dimension([[3, 4], [5, 6], [7, 9]]), [[[3, 4], [5, 6], [7, 8]], [[3, 4], [5, 6], [8, 9]], 2, 8])

    def test_split_by_all_dimensions(self):
        print(colored("Checking rectangle splicing in each dimension", 'blue'))
        self.assertEqual(split_by_all_dimensions([[3, 4]]), [[[3, 3.5]], [[3.5, 4]]])
        self.assertEqual(split_by_all_dimensions([[3, 4], [5, 6]]), [[[3, 3.5], [5, 5.5]], [[3, 3.5], [5.5, 6]], [[3.5, 4], [5, 5.5]], [[3.5, 4], [5.5, 6]]])
        self.assertEqual(split_by_all_dimensions([[3, 4], [5, 6], [7, 8]]), [[[3, 3.5], [5, 5.5], [7, 7.5]], [[3, 3.5], [5, 5.5], [7.5, 8]], [[3, 3.5], [5.5, 6], [7, 7.5]], [[3, 3.5], [5.5, 6], [7.5, 8]], [[3.5, 4], [5, 5.5], [7, 7.5]], [[3.5, 4], [5, 5.5], [7.5, 8]], [[3.5, 4], [5.5, 6], [7, 7.5]], [[3.5, 4], [5.5, 6], [7.5, 8]]])

    def test_refine_by(self):
        print(colored("Checking spliced hyperrectangle by a second one ", 'blue'))
        # TODO

    def test_refine_into_rectangles(self):
        print(colored("Checking Refining of the sampled space into hyperrectangles such that rectangle is all sat or all unsat", 'blue'))
        # TODO

    def test_find_max_rectangle(self):
        print(colored("Checking Finding of the largest hyperrectangles such that rectangle is all sat or all unsat from starting point in positive direction", 'blue'))
        # TODO


if __name__ == '__main__':
    unittest.main()
