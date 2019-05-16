from collections import Iterable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from numpy import prod
import copy
import unittest
## colored output
from termcolor import colored


def get_rectangle_volume(rectangle):
    """Computes the volume of the given (hyper)rectangle

    Args
    ------
    rectangle:  (list of intervals) defining the (hyper)rectangle
    """
    intervals = []
    ## If there is empty rectangle
    if not rectangle:
        raise Exception("empty rectangle has no volume")
    for interval in rectangle:
        intervals.append(interval[1] - interval[0])
    return prod(intervals)


class RefinedSpace:
    """ Class to represent space refinement into sat(green), unsat(red), and unknown(white) regions

    Args
    ------
    region: (list of intervals): whole space
    rectangles_sat: (list of intervals): sat (green) space
    rectangles_unsat: (list of intervals): unsat (red) space
    rectangles_unknown: (list of intervals): unknown (white) space
    """

    def __init__(self, region, params, rectangles_sat=[], rectangles_unsat=[], rectangles_unknown=None):
        if not isinstance(region, Iterable):
            raise Exception("Given region is not iterable")
        if isinstance(region, tuple):
            self.region = [region]
        else:
            self.region = region

        self.params = params
        if not len(self.params) == len(self.region):
            print(colored(f"number of parameters ({len(params)}) and dimension of the region ({len(region)}) is not equal", 'red'))
            # print("region", self.region)

        # print("rectangles_sat", rectangles_sat)
        if not isinstance(rectangles_sat, Iterable):
            raise Exception("Given rectangles_sat is not iterable")
        if isinstance(rectangles_sat, tuple):
            self.rectangles_sat = [rectangles_sat]
        else:
            self.sat = rectangles_sat

        # print("rectangles_unsat", rectangles_unsat)
        if not isinstance(rectangles_unsat, Iterable):
            raise Exception("Given rectangles_unsat is not iterable")
        if isinstance(rectangles_unsat, tuple):
            self.rectangles_sat = [rectangles_sat]
        else:
            self.unsat = rectangles_unsat

        # print("rectangles_unknown", rectangles_unknown)
        if rectangles_unknown is None:
            self.unknown = [region]
        elif not isinstance(rectangles_unknown, Iterable):
            raise Exception("Given rectangles_unknown is not iterable")
        elif isinstance(rectangles_unknown, tuple):
            self.unknown = [rectangles_unknown]
        else:
            self.unknown = rectangles_unknown

    def show(self, title):
        if len(self.region) == 1 or len(self.region) == 2:
            # colored(globals()["default_region"], self.region)

            # from matplotlib import rcParams
            # rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
            fig = plt.figure()
            pic = fig.add_subplot(111, aspect='equal')
            pic.set_xlabel(self.params[0])

            ## Set axis ranges
            region = copy.deepcopy(self.region)
            if region[0][1] - region[0][0] < 0.1:
                region[0] = (region[0][0] - 0.2, region[0][1] + 0.2)
            pic.axis([region[0][0], region[0][1], 0, 1])

            if len(self.region) == 2:
                pic.set_ylabel(self.params[1])
                if region[1][1] - region[1][0] < 0.1:
                    region[1] = (region[1][0] - 0.2, region[1][1] + 0.2)
                pic.axis([region[0][0], region[0][1], region[1][0], region[1][1]])
            pic.set_title("red = unsafe region, green = safe region, white = in between \n " + title)

            pic.add_collection(self.show_green())
            pic.add_collection(self.show_red())
            plt.show()
            del region

    def get_volume(self):
        intervals = []
        for interval in self.region:
            intervals.append(interval[1] - interval[0])
        return prod(intervals)

    def add_green(self, green):
        self.sat.append(green)

    def add_red(self, red):
        self.unsat.append(red)

    def add_white(self, white):
        self.unknown.append(white)

    def remove_green(self, green):
        self.sat.remove(green)

    def remove_red(self, red):
        self.unsat.remove(red)

    def remove_white(self, white):
        try:
            self.unknown.remove(white)
        except:
            print("Could not remove white area ", white)
            return False
        return True

    def get_green(self):
        return self.sat

    def get_red(self):
        return self.unsat

    def get_white(self):
        return self.unknown

    def get_green_volume(self):
        cumulative_volume = 0

        ## If there is no hyperrectangle in the sat space
        if not self.sat:
            return 0.0

        for rectangle in self.sat:
            cumulative_volume = cumulative_volume + get_rectangle_volume(rectangle)
        return cumulative_volume

    def get_red_volume(self):
        cumulative_volume = 0

        ## If there is no hyperrectangle in the unsat space
        if not self.unsat:
            return 0.0

        for rectangle in self.unsat:
            cumulative_volume = cumulative_volume + get_rectangle_volume(rectangle)
        return cumulative_volume

    def get_nonwhite_volume(self):
        return self.get_green_volume() + self.get_red_volume()

    def get_coverage(self):
        return self.get_nonwhite_volume() / self.get_volume()

    ## TBD generalise so that the code is not copied
    def show_green(self):
        rectangles_sat = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return
        elif len(self.region) == 2:
            for rectangle in self.sat:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_sat.append(Rectangle((rectangle[0][0], rectangle[1][0]), rectangle[0][1] - rectangle[0][0],
                                                rectangle[1][1] - rectangle[1][0], fc='g'))
        elif len(self.region) == 1:
            for rectangle in self.sat:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_sat.append(
                    Rectangle((rectangle[0][0], 0.33), rectangle[0][1] - rectangle[0][0], 0.33, fc='g'))
        return PatchCollection(rectangles_sat, facecolor='g', alpha=0.5)

    def show_red(self):
        rectangles_unsat = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return
        elif len(self.region) == 2:
            for rectangle in self.unsat:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_unsat.append(Rectangle((rectangle[0][0], rectangle[1][0]), rectangle[0][1] - rectangle[0][0],
                                                  rectangle[1][1] - rectangle[1][0], fc='r'))
        elif len(self.region) == 1:
            for rectangle in self.unsat:
                ## (Rectangle((low_x,low_y), width, height, fc= color)
                rectangles_unsat.append(
                    Rectangle((rectangle[0][0], 0.33), rectangle[0][1] - rectangle[0][0], 0.33, fc='r'))
        return PatchCollection(rectangles_unsat, facecolor='r', alpha=0.5)

    def __repr__(self):
        return str([self.sat, self.unsat, self.unknown])

    def __str__(self):
        return [self.sat, self.unsat, self.unknown]


class TestLoad(unittest.TestCase):
    def test_space(self):
        print("Test Refined space here")
        ## def __init__(region, params, rectangles_sat=[], rectangles_unsat=[], rectangles_unknown=None):
        with self.assertRaises(Exception):
            RefinedSpace(5, )
        with self.assertRaises(Exception):
            RefinedSpace([], 5)
        with self.assertRaises(Exception):
            RefinedSpace([], [], 5)

        space = RefinedSpace((0, 1), ["x"])
        # print(space.get_volume())
        self.assertEqual(space.get_volume(), 1)
        self.assertEqual(space.get_green_volume(), 0)
        self.assertEqual(space.get_red_volume(), 0)
        self.assertEqual(space.get_nonwhite_volume(), 0)
        self.assertEqual(space.get_coverage(), 0)

        space = RefinedSpace([(0, 1)], ["x"])
        self.assertEqual(space.get_volume(), 1)
        self.assertEqual(space.get_green_volume(), 0)
        self.assertEqual(space.get_red_volume(), 0)
        self.assertEqual(space.get_nonwhite_volume(), 0)
        self.assertEqual(space.get_coverage(), 0)
        # print(space.show_green())
        # print(space.show_red())

        self.assertEqual(round(get_rectangle_volume([[0, 0.5]]), 1), 0.5)
        self.assertEqual(round(get_rectangle_volume([[0, 0.2], [0, 0.2]]), 2), 0.04)

        # space.show( "max_recursion_depth:{},\n min_rec_size:{}, achieved_coverage:{}, alg{} \n It took {} {} second(s)".format(
        #        n, epsilon, self.get_coverage(), version, socket.gethostname(), round(time.time() - start_time, 1)))

        space.show(f"No green, \n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 1), 0.0)
        self.assertEqual(round(space.get_red_volume(), 1), 0.0)
        self.assertEqual(round(space.get_nonwhite_volume(), 1), 0.0)

        space.add_green([[0, 0.5]])
        space.show(f"First half green added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 1), 0.5)
        self.assertEqual(round(space.get_red_volume(), 1), 0.0)
        self.assertEqual(round(space.get_nonwhite_volume(), 1), 0.5)

        ## def __init__(region, params, rectangles_sat=[], rectangles_unsat=[], rectangles_unknown=None):
        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], [[[0, 0.5], [0, 0.5]]], [])
        space.show(f"Left bottom quarter green added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 2), 0.25)
        self.assertEqual(round(space.get_red_volume(), 1), 0.0)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.25)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], [], [[[0, 0.5], [0, 0.5]]])
        space.show(f"Left bottom quarter red added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 1), 0.0)
        self.assertEqual(round(space.get_red_volume(), 2), 0.25)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.25)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], [[[0, 0.2], [0, 0.2]]], [[[0.5, 0.7], [0.1, 0.3]]])
        space.show(f"One green and one red region added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 2), 0.04)
        self.assertEqual(round(space.get_red_volume(), 2), 0.04)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.08)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], [[[0, 0.2], [0, 0.2]], [[0.4, 0.6], [0.6, 0.8]]],
                             [[[0.5, 0.7], [0.1, 0.3]], [[0.6, 0.8], [0.8, 1]]])
        space.show(f"Two green and two red regions added,\n  achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 2), 0.08)
        self.assertEqual(round(space.get_red_volume(), 2), 0.08)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.16)


if __name__ == "__main__":
    unittest.main()
