from collections import Iterable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from numpy import prod
import copy
import unittest
## Colored output
from termcolor import colored

## ONLY FOR SAVING FILES
import os
import sys
workspace = os.path.dirname(__file__)
sys.path.append(workspace)
import configparser

config = configparser.ConfigParser()
workspace = os.path.dirname(__file__)
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)

config.read("../config.ini")
refinement_results = config.get("paths", "refinement_results")
if not os.path.exists(refinement_results):
    os.makedirs(refinement_results)

os.chdir(cwd)


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

    Attributes
    ------
    region: (list of intervals): whole space
    params: (list of strings): parameter names
    types: (list of string): parameter types (Real, Int, Bool, ...)
    rectangles_sat: (list of intervals): sat (green) space
    rectangles_unsat: (list of intervals): unsat (red) space
    rectangles_unknown: (list of intervals): unknown (white) space
    sat_samples: (list of points): satisfying points
    unsat_samples: (list of points): unsatisfying points

    """

    def __init__(self, region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None):
        """
        Args
        ------
        region: (list of intervals): whole space
        params: (list of strings): parameter names
        types: (list of string): parameter types (Real, Int, Bool, ...)
        rectangles_sat: (list of intervals): sat (green) space
        rectangles_unsat: (list of intervals): unsat (red) space
        rectangles_unknown: (list of intervals): unknown (white) space
        sat_samples: (list of points): satisfying points
        unsat_samples: (list of points): unsatisfying points
        """

        ## REGION
        if not isinstance(region, Iterable):
            raise Exception("Given region is not iterable")
        if isinstance(region, tuple):
            self.region = [region]
        else:
            ### Taking care of unchangable tuples
            for interval_index in range(len(region)):
                region[interval_index] = [region[interval_index][0], region[interval_index][1]]
            self.region = region

        ## PARAMS
        self.params = params
        if not len(self.params) == len(self.region):
            print(colored(f"number of parameters ({len(params)}) and dimension of the region ({len(region)}) is not equal", 'red'))
            raise Exception(f"number of parameters ({len(params)}) and dimension of the region ({len(region)}) is not equal")

        if types is None or types is False:
            self.types = []
            ## IF no types are given
            for i in region:
                self.types.append("Real")
        else:
            self.types = types
            if not isinstance(types, Iterable):
                raise Exception("Given types is not iterable")
            if isinstance(types, tuple):
                self.types = [types]
            else:
                self.types = types

            if not len(self.types) == len(self.region):
                print(colored(
                    f"number of types of parameters ({len(types)}) and dimension of the region ({len(region)}) is not equal",
                    'red'))
                raise Exception(
                    f"number of types ({len(types)}) and dimension of the region ({len(region)}) is not equal")

        ## SAT RECTANGLES
        # print("rectangles_sat", rectangles_sat)
        if rectangles_sat is False:
            rectangles_sat = []
        if not isinstance(rectangles_sat, Iterable):
            raise Exception("Given rectangles_sat is not iterable")
        if isinstance(rectangles_sat, tuple):
            self.rectangles_sat = [rectangles_sat]
        else:
            self.sat = rectangles_sat

        ## UNSAT RECTANGLES
        if rectangles_unsat is False:
            rectangles_sat = []
        # print("rectangles_unsat", rectangles_unsat)
        if not isinstance(rectangles_unsat, Iterable):
            raise Exception("Given rectangles_unsat is not iterable")
        if isinstance(rectangles_unsat, tuple):
            self.rectangles_sat = [rectangles_sat]
        else:
            self.unsat = rectangles_unsat

        ## UNKNOWN RECTANGLES
        # print("rectangles_unknown", rectangles_unknown)
        if rectangles_unknown is None:
            ## TBD THIS IS NOT CORRECT
            self.unknown = [region]
        elif not isinstance(rectangles_unknown, Iterable):
            raise Exception("Given rectangles_unknown is not iterable")
        elif isinstance(rectangles_unknown, tuple):
            self.unknown = [rectangles_unknown]
        else:
            self.unknown = rectangles_unknown

        ## SAT SAMPLES
        if sat_samples is None:
            self.sat_samples = []
        elif not isinstance(sat_samples, Iterable):
            raise Exception("Given samples are not iterable")
        else:
            # print("samples", samples)
            self.sat_samples = sat_samples

        ## UNSAT SAMPLES
        if unsat_samples is None:
            self.unsat_samples = []
        elif not isinstance(unsat_samples, Iterable):
            raise Exception("Given samples are not iterable")
        else:
            # print("samples", samples)
            self.unsat_samples = unsat_samples

    def show(self, title="", green=True, red=True, sat_samples=False, unsat_samples=False, save=False):
        """
        Visualises the space

        Args
        ----------
        title: (String) title of the figure
        green: (Bool) if True showing safe space
        red: (Bool) if True showing unsafe space
        sat_samples: (Bool) if True showing sat samples
        unsat_samples: (Bool) if True showing unsat samples
        save: (String/Bool) output file.format, if False or "" no saving
        """

        # print("default figure name", save)
        if isinstance(save, str):
            if "." not in save:
                save = f"{save}.png"
                save = os.path.join(refinement_results, save)
                # print("figure name:", save)

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

            pretitle = ""
            if green or red:
                pretitle = pretitle + " Refinement,"
            if sat_samples or unsat_samples:
                pretitle = pretitle + " Samples,"
            if pretitle:
                pretitle = pretitle + "\n"
            if green:
                pic.add_collection(self.show_green())
            if red:
                pic.add_collection(self.show_red())
            if sat_samples:
                pic.add_collection(self.show_samples(True))
            if unsat_samples:
                pic.add_collection(self.show_samples(False))

            pic.set_title(pretitle + "red = unsafe region, green = safe region, white = in between \n " + title)

            ## Save the figure
            if save:
                plt.savefig(save, bbox_inches='tight')
                print("Figure stored here: ", save)

            plt.show()
            del region

        else:
            print("Multidimensional space, showing only samples")
            if sat_samples:
                if self.sat_samples:
                    fig, ax = plt.subplots()
                    ## Creates values of the horizontal axis
                    x_axis = []
                    i = 0
                    for dimension in self.sat_samples[0]:
                        i = i + 1
                        x_axis.append(i)

                    ## Get values of the vertical axis for respective line
                    for sample in self.sat_samples:
                        # print("samples", sample)
                        ax.scatter(x_axis, sample)
                        ax.plot(x_axis, sample)
                    ax.set_xlabel("param indices")
                    ax.set_ylabel("parameter value")
                    ax.set_title("Sat sample points of the given hyperspace")
                    ax.autoscale()
                    ax.margins(0.1)

                    ## Save the figure
                    if save:
                        plt.savefig(save, bbox_inches='tight')
                        print("Save sat in space", save)

                    plt.show()
                else:
                    print("No sat samples so far, nothing to show")

            if unsat_samples:
                if self.unsat_samples:
                    fig, ax = plt.subplots()
                    ## Creates values of the horizontal axis
                    x_axis = []
                    i = 0
                    for dimension in self.unsat_samples[0]:
                        i = i + 1
                        x_axis.append(i)

                    ## Get values of the vertical axis for respective line
                    for sample in self.unsat_samples:
                        # print("samples", sample)
                        ax.scatter(x_axis, sample)
                        ax.plot(x_axis, sample)
                    ax.set_xlabel("param indices")
                    ax.set_ylabel("parameter value")
                    ax.set_title("Unsat sample points of the given hyperspace")
                    ax.autoscale()
                    ax.margins(0.1)

                    ## Save the figure
                    if save:
                        plt.savefig(save, bbox_inches='tight')
                        print("Figure stored here: ", save)

                    plt.show()
                else:
                    print("No unsat samples so far, nothing to show")

    def get_volume(self):
        """Returns the volume of the space"""
        intervals = []
        for interval in self.region:
            intervals.append(interval[1] - interval[0])
        return prod(intervals)

    def add_green(self, green):
        """Adds green (hyper)rectangle"""
        self.sat.append(green)

    def add_red(self, red):
        """Adds red (hyper)rectangle"""
        self.unsat.append(red)

    def add_white(self, white):
        """Adds white (hyper)rectangle"""
        self.unknown.append(white)

    def add_sat_samples(self, sat_samples):
        """Adds sat samples

        Args
        -------
        sat_samples: (list) of sat points
        """
        # print("samples", samples)
        self.sat_samples = sat_samples

    def add_unsat_samples(self, unsat_samples):
        """Adds unsat samples

        Args
        -------
        unsat_samples: (list) of unsat points
        """
        # print("samples", samples)
        self.unsat_samples = unsat_samples

    def remove_green(self, green):
        """Removes green (hyper)rectangle"""
        self.sat.remove(green)

    def remove_red(self, red):
        """Removes red (hyper)rectangle"""
        self.unsat.remove(red)

    def remove_white(self, white):
        """Removes white (hyper)rectangle"""
        try:
            self.unknown.remove(white)
        except:
            print("Could not remove white area ", white)
            return False
        return True

    def get_green(self):
        """Returns green (hyper)rectangles"""
        return self.sat

    def get_red(self):
        """Returns red (hyper)rectangles"""
        return self.unsat

    def get_white(self):
        """Returns white (hyper)rectangles"""
        return self.unknown

    def get_green_volume(self):
        """Returns volume of green subspace"""
        cumulative_volume = 0

        ## If there is no hyperrectangle in the sat space
        if not self.sat:
            return 0.0

        for rectangle in self.sat:
            cumulative_volume = cumulative_volume + get_rectangle_volume(rectangle)
        return cumulative_volume

    def get_red_volume(self):
        """Returns volume of red subspace"""
        cumulative_volume = 0

        ## If there is no hyperrectangle in the unsat space
        if not self.unsat:
            return 0.0

        for rectangle in self.unsat:
            cumulative_volume = cumulative_volume + get_rectangle_volume(rectangle)
        return cumulative_volume

    def get_nonwhite_volume(self):
        """Returns volume of nonwhite subspace"""
        return self.get_green_volume() + self.get_red_volume()

    def get_coverage(self):
        """Returns proption of nonwhite subspace (coverage)"""
        return self.get_nonwhite_volume() / self.get_volume()

    ## TBD generalise so that the code is not copied
    def show_green(self):
        """Adds green (hyper)rectangles to be visualised"""
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
        """Adds red (hyper)rectangles to be visualised"""
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

    def show_samples(self, which):
        """Visualises samples"""
        samples = []
        if len(self.region) > 2:
            print("Error while visualising", len(self.region), "dimensional space")
            return
        elif len(self.region) == 2:
            # print("samples", self.samples)
            size_correction =  min(1/(len(self.sat_samples) + len(self.unsat_samples))**(1/len(self.region)), 0.01)

            ## CHOOSING SAT OR UNSAT
            if which:
                for rectangle in self.sat_samples:
                    ## (Rectangle((low_x,low_y), width, height, fc= color)
                    # print("rectangle", rectangle)
                    samples.append(Rectangle((rectangle[0]-0.005, rectangle[1]-0.005), size_correction, size_correction, fc='r'))
                return PatchCollection(samples, facecolor='g', alpha=0.5)
            else:
                for rectangle in self.unsat_samples:
                    ## (Rectangle((low_x,low_y), width, height, fc= color)
                    # print("rectangle", rectangle)
                    samples.append(Rectangle((rectangle[0]-0.005, rectangle[1]-0.005), size_correction, size_correction, fc='r'))
                return PatchCollection(samples, facecolor='r', alpha=0.5)

    def __repr__(self):
        return str([self.region, self.sat, self.unsat, self.unknown])

    def __str__(self):
        return str([self.region, self.sat, self.unsat, self.unknown])


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
