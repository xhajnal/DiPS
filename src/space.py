from collections import Iterable
from numpy import prod
import unittest


class RefinedSpace:
    """ Class to represent space refinement into sat(green), unsat(red), and unknown(white) regions

    Args
    ------
    region: list of intervals -- whole space
    rectangles_sat:  list of intervals -- sat (green) space
    rectangles_unsat:  list of intervals -- unsat (red) space
    rectangles_unknown:  list of intervals -- unknown (white) space
    """

    def __init__(self, region, rectangles_sat=[], rectangles_unsat=[], rectangles_unknown=None):
        if not isinstance(region, Iterable):
            raise Exception("Given region is not iterable")
        if isinstance(region, tuple):
            self.region = [region]
        else:
            self.region = region

        if not isinstance(rectangles_sat, Iterable):
            raise Exception("Given rectangles_sat is not iterable")
        if isinstance(rectangles_sat, tuple):
            self.rectangles_sat = [rectangles_sat]
        else:
            self.sat = rectangles_sat

        if not isinstance(rectangles_unsat, Iterable):
            raise Exception("Given rectangles_unsat is not iterable")
        if isinstance(rectangles_unsat, tuple):
            self.rectangles_sat = [rectangles_sat]
        else:
            self.unsat = rectangles_unsat

        if rectangles_unknown is None:
            self.unknown = region
        else:
            self.unknown = rectangles_unknown

    def get_volume(self):
        add_space = []
        for interval in self.region:
            add_space.append(interval[1] - interval[0])
        return prod(add_space)

    def add_green(self, green):
        self.sat.append(green)

    def add_red(self, red):
        self.unsat.append(red)

    def get_green(self):
        add_space = []
        for interval in self.sat:
            add_space.append(interval[1] - interval[0])
        if add_space:
            return prod(add_space)
        else:
            return 0.0

    def get_red(self):
        add_space = []
        for interval in self.unsat:
            add_space.append(interval[1] - interval[0])
        if add_space:
            return prod(add_space)
        else:
            return 0.0

    def get_nonwhite(self):
        return self.get_green() + self.get_red()

    def get_coverage(self):
        return self.get_nonwhite() / self.get_volume()


class TestLoad(unittest.TestCase):
    def test_space(self):

        with self.assertRaises(Exception):
            RefinedSpace(5)
        with self.assertRaises(Exception):
            RefinedSpace([],5)
        with self.assertRaises(Exception):
            RefinedSpace([],[],5)

        space = RefinedSpace((0, 1))
        print(space.get_volume())
        self.assertEqual(space.get_volume(), 1)
        self.assertEqual(space.get_green(), 0)
        self.assertEqual(space.get_red(), 0)
        self.assertEqual(space.get_nonwhite(), 0)
        self.assertEqual(space.get_coverage(), 0)

        space = RefinedSpace([(0, 1)])
        self.assertEqual(space.get_volume(), 1)
        self.assertEqual(space.get_green(), 0)
        self.assertEqual(space.get_red(), 0)
        self.assertEqual(space.get_nonwhite(), 0)
        self.assertEqual(space.get_coverage(), 0)


if __name__ == "__main__":
    unittest.main()
