import pickle
import unittest
from src.space import *

curr_dir = os.path.dirname(__file__)


class MyTestCase(unittest.TestCase):
    def test_init_space(self):
        print(colored("Space init tests", 'blue'))
        ## RefinedSpace(region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, true_point=False, title=False):
        with self.assertRaises(Exception):
            RefinedSpace(5, )
        with self.assertRaises(Exception):
            RefinedSpace([], 5)
        with self.assertRaises(Exception):
            RefinedSpace([], [], 5)

        space = RefinedSpace((0, 1), ["x"])
        self.assertEqual(space.get_region(), [[0, 1]])
        self.assertEqual(space.get_params(), ["x"])
        self.assertEqual(space.get_volume(), 1)
        self.assertEqual(space.get_green(), [])
        self.assertEqual(space.get_green_volume(), 0)
        self.assertEqual(space.get_red(), [])
        self.assertEqual(space.get_red_volume(), 0)
        self.assertEqual(space.get_white(), {1: [[[0, 1]]]})
        self.assertEqual(space.get_white_volume(), 1)
        self.assertEqual(space.get_nonwhite(), [])
        self.assertEqual(space.get_nonwhite_volume(), 0)
        self.assertEqual(space.get_coverage(), 0)
        self.assertEqual(space.get_true_point(), None)

        space = RefinedSpace([(0, 1), (2, 4)], ["x", "y"])
        self.assertEqual(space.get_region(), [[0, 1], [2, 4]])
        self.assertEqual(space.get_params(), ["x", "y"])
        self.assertEqual(space.get_volume(), 2)
        self.assertEqual(space.get_green(), [])
        self.assertEqual(space.get_green_volume(), 0)
        self.assertEqual(space.get_red(), [])
        self.assertEqual(space.get_red_volume(), 0)
        self.assertEqual(space.get_white(), {2: [[[0, 1], [2, 4]]]})
        self.assertEqual(space.get_white_volume(), 2)
        self.assertEqual(space.get_nonwhite(), [])
        self.assertEqual(space.get_nonwhite_volume(), 0)
        self.assertEqual(space.get_coverage(), 0)
        self.assertEqual(space.get_true_point(), None)

        space = pickle.load(open(os.path.join(curr_dir, "data/space.p"), "rb"))
        self.assertEqual(space.get_region(), [[0, 1], [0, 1]])
        self.assertEqual(space.get_params(), ["p", "q"])
        self.assertEqual(space.get_volume(), 1)
        self.assertEqual(space.get_green(), [[(0.375, 0.4375), (0.0, 0.125)]])
        self.assertEqual(space.get_green_volume(), 0.0078125)
        self.assertEqual(space.get_red(), [[(0.5, 1.0), [0.0, 1.0]], [(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.25), (0.0, 0.5)], [(0.25, 0.375), (0.375, 0.5)], [(0.375, 0.5), (0.375, 0.5)], [(0.25, 0.3125), (0.0, 0.125)], [(0.25, 0.3125), (0.125, 0.25)], [(0.25, 0.3125), (0.25, 0.375)], [(0.4375, 0.5), (0.25, 0.375)]])
        self.assertEqual(space.get_red_volume(), 0.9375)
        self.assertEqual(space.get_white(), [[(0.3125, 0.375), (0.0, 0.125)], [(0.3125, 0.375), (0.125, 0.25)], [(0.4375, 0.5), (0.0, 0.125)], [(0.375, 0.4375), (0.125, 0.25)], [(0.4375, 0.5), (0.125, 0.25)], [(0.3125, 0.375), (0.25, 0.375)], [(0.375, 0.4375), (0.25, 0.375)]])
        self.assertEqual(space.get_white_volume(), 0.0546875)
        self.assertEqual(space.get_nonwhite(), [[(0.375, 0.4375), (0.0, 0.125)], [(0.5, 1.0), [0.0, 1.0]], [(0.0, 0.5), (0.5, 1.0)], [(0.0, 0.25), (0.0, 0.5)], [(0.25, 0.375), (0.375, 0.5)], [(0.375, 0.5), (0.375, 0.5)], [(0.25, 0.3125), (0.0, 0.125)], [(0.25, 0.3125), (0.125, 0.25)], [(0.25, 0.3125), (0.25, 0.375)], [(0.4375, 0.5), (0.25, 0.375)]])
        self.assertEqual(space.get_nonwhite_volume(), 0.0078125 + 0.9375)
        self.assertEqual(space.get_coverage(), 0.0078125 + 0.9375)
        self.assertEqual(space.get_true_point(), None)
        self.assertEqual(space.get_sat_samples(), [[0.3333333333333333, 0.16666666666666666]])
        self.assertEqual(space.get_unsat_samples(), [[0.0, 0.0], [0.0, 0.16666666666666666], [0.0, 0.3333333333333333], [0.0, 0.5], [0.0, 0.6666666666666666], [0.0, 0.8333333333333333], [0.0, 1.0], [0.16666666666666666, 0.0], [0.16666666666666666, 0.16666666666666666], [0.16666666666666666, 0.3333333333333333], [0.16666666666666666, 0.5], [0.16666666666666666, 0.6666666666666666], [0.16666666666666666, 0.8333333333333333], [0.16666666666666666, 1.0], [0.3333333333333333, 0.0], [0.3333333333333333, 0.3333333333333333], [0.3333333333333333, 0.5], [0.3333333333333333, 0.6666666666666666], [0.3333333333333333, 0.8333333333333333], [0.3333333333333333, 1.0], [0.5, 0.0], [0.5, 0.16666666666666666], [0.5, 0.3333333333333333], [0.5, 0.5], [0.5, 0.6666666666666666], [0.5, 0.8333333333333333], [0.5, 1.0], [0.6666666666666666, 0.0], [0.6666666666666666, 0.16666666666666666], [0.6666666666666666, 0.3333333333333333], [0.6666666666666666, 0.5], [0.6666666666666666, 0.6666666666666666], [0.6666666666666666, 0.8333333333333333], [0.6666666666666666, 1.0], [0.8333333333333333, 0.0], [0.8333333333333333, 0.16666666666666666], [0.8333333333333333, 0.3333333333333333], [0.8333333333333333, 0.5], [0.8333333333333333, 0.6666666666666666], [0.8333333333333333, 0.8333333333333333], [0.8333333333333333, 1.0], [1.0, 0.0], [1.0, 0.16666666666666666], [1.0, 0.3333333333333333], [1.0, 0.5], [1.0, 0.6666666666666666], [1.0, 0.8333333333333333], [1.0, 1.0]])
        self.assertEqual(space.get_all_samples(), [[0.3333333333333333, 0.16666666666666666], [0.0, 0.0], [0.0, 0.16666666666666666], [0.0, 0.3333333333333333], [0.0, 0.5], [0.0, 0.6666666666666666], [0.0, 0.8333333333333333], [0.0, 1.0], [0.16666666666666666, 0.0], [0.16666666666666666, 0.16666666666666666], [0.16666666666666666, 0.3333333333333333], [0.16666666666666666, 0.5], [0.16666666666666666, 0.6666666666666666], [0.16666666666666666, 0.8333333333333333], [0.16666666666666666, 1.0], [0.3333333333333333, 0.0], [0.3333333333333333, 0.3333333333333333], [0.3333333333333333, 0.5], [0.3333333333333333, 0.6666666666666666], [0.3333333333333333, 0.8333333333333333], [0.3333333333333333, 1.0], [0.5, 0.0], [0.5, 0.16666666666666666], [0.5, 0.3333333333333333], [0.5, 0.5], [0.5, 0.6666666666666666], [0.5, 0.8333333333333333], [0.5, 1.0], [0.6666666666666666, 0.0], [0.6666666666666666, 0.16666666666666666], [0.6666666666666666, 0.3333333333333333], [0.6666666666666666, 0.5], [0.6666666666666666, 0.6666666666666666], [0.6666666666666666, 0.8333333333333333], [0.6666666666666666, 1.0], [0.8333333333333333, 0.0], [0.8333333333333333, 0.16666666666666666], [0.8333333333333333, 0.3333333333333333], [0.8333333333333333, 0.5], [0.8333333333333333, 0.6666666666666666], [0.8333333333333333, 0.8333333333333333], [0.8333333333333333, 1.0], [1.0, 0.0], [1.0, 0.16666666666666666], [1.0, 0.3333333333333333], [1.0, 0.5], [1.0, 0.6666666666666666], [1.0, 0.8333333333333333], [1.0, 1.0]])

    def test_space_basics(self):
        print(colored("Basic space tests", 'blue'))
        ## RefinedSpace(region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, true_point=False, title=False):

        space = RefinedSpace((0, 1), ["x"])
        space.show(f"No green, \n achieved_coverage: {space.get_coverage() * 100}%")

        space.add_green([[0, 0.5]])
        self.assertEqual(space.get_green_volume(), 0.5)
        # print(space.get_green())
        space.remove_green([[0, 0.5]])
        # print(space.get_green())
        self.assertEqual(space.get_green_volume(), 0.0)

        space.add_red([[0.5, 1]])
        self.assertEqual(space.get_red_volume(), 0.5)
        space.remove_red([[0.5, 1]])
        self.assertEqual(space.get_red_volume(), 0.0)

        self.assertEqual(space.get_white_volume(), 1)
        # print(space.get_white())
        space.remove_white([[0, 1]])
        # print(space.get_white())
        space.add_green([[0, 0.5]])
        space.add_red([[0.5, 1]])
        self.assertEqual(space.get_white_volume(), 0.0)

        ## TODO - test the rest of methods
        space = RefinedSpace((0, 1), ["x"])
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

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], ["Real", "Real"], [[[0, 0.5], [0, 0.5]]], [])
        space.show(f"Left bottom quarter green added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 2), 0.25)
        self.assertEqual(round(space.get_red_volume(), 1), 0.0)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.25)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], ["Real", "Real"], [], [[[0, 0.5], [0, 0.5]]])
        space.show(f"Left bottom quarter red added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 1), 0.0)
        self.assertEqual(round(space.get_red_volume(), 2), 0.25)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.25)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], ["Real", "Real"], [[[0, 0.2], [0, 0.2]]], [[[0.5, 0.7], [0.1, 0.3]]])
        space.show(f"One green and one red region added,\n achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 2), 0.04)
        self.assertEqual(round(space.get_red_volume(), 2), 0.04)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.08)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], ["Real", "Real"], [[[0, 0.2], [0, 0.2]], [[0.4, 0.6], [0.6, 0.8]]],
                             [[[0.5, 0.7], [0.1, 0.3]], [[0.6, 0.8], [0.8, 1]]])
        space.show(f"Two green and two red regions added,\n  achieved_coverage: {space.get_coverage() * 100}%")
        self.assertEqual(round(space.get_green_volume(), 2), 0.08)
        self.assertEqual(round(space.get_red_volume(), 2), 0.08)
        self.assertEqual(round(space.get_nonwhite_volume(), 2), 0.16)

    def test_add_white(self):
        space = RefinedSpace((0, 1), ["x"])
        space.add_white([[0, 0.5]])

        print(space)
        space.remove_white([[0, 1]])
        space.add_white([[0.5, 1]])

        print(space)

    def test_visualisation(self):
        print(colored("Space visualisations tests", 'blue'))

        # os.chdir(cwd)
        print("curr dir", curr_dir)
        space = pickle.load(open(os.path.join(curr_dir, "data/space.p"), "rb"))

        ## Only sampling here
        space.show(sat_samples=True, unsat_samples=True, green=False, red=False)

        ## Normal refinement should appear now
        space.show(green=True, red=True)

    def test_visualisation_multidim(self):
        print(colored("Multidimensional space visualisations tests", 'blue'))
        space = RefinedSpace([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], ['r_0', 'r_1', 'r_2'], ['Real', 'Real', 'Real'], [[(0.75, 1.0), (0.0, 0.5), (0.0, 0.5)]], [[(0.75, 1.0), (0.0, 0.5), (0.5, 1.0)], [(0.75, 1.0), (0.5, 1.0), (0.0, 0.5)], [(0.75, 1.0), (0.5, 1.0), (0.5, 1.0)], [(0.0, 0.25), (0.0, 0.25), (0.0, 0.5)], [(0.25, 0.5), (0.0, 0.25), (0.0, 0.5)], [(0.0, 0.25), (0.0, 0.25), (0.5, 1.0)], [(0.25, 0.5), (0.0, 0.25), (0.5, 1.0)], [(0.0, 0.25), (0.75, 1.0), (0.0, 0.5)], [(0.25, 0.5), (0.75, 1.0), (0.0, 0.5)], [(0.0, 0.25), (0.75, 1.0), (0.5, 1.0)], [(0.25, 0.5), (0.75, 1.0), (0.5, 1.0)], [(0.5, 0.75), (0.0, 0.25), (0.0, 0.5)], [(0.5, 0.75), (0.0, 0.25), (0.5, 1.0)], [(0.5, 0.75), (0.75, 1.0), (0.0, 0.5)], [(0.5, 0.75), (0.75, 1.0), (0.5, 1.0)]], [[(0.0, 0.25), (0.25, 0.5), (0.0, 0.5)], [(0.25, 0.5), (0.25, 0.5), (0.0, 0.5)], [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)], [(0.25, 0.5), (0.25, 0.5), (0.5, 1.0)], [(0.0, 0.25), (0.5, 0.75), (0.0, 0.5)], [(0.25, 0.5), (0.5, 0.75), (0.0, 0.5)], [(0.0, 0.25), (0.5, 0.75), (0.5, 1.0)], [(0.25, 0.5), (0.5, 0.75), (0.5, 1.0)], [(0.5, 0.75), (0.25, 0.5), (0.0, 0.5)], [(0.5, 0.75), (0.25, 0.5), (0.5, 1.0)], [(0.5, 0.75), (0.5, 0.75), (0.0, 0.5)], [(0.5, 0.75), (0.5, 0.75), (0.5, 1.0)]], [], [], None)

        ## Multiple lines connecting values of respective parameter should appear now
        ## TODO - add space with samples
        # space.show(sat_samples=True, unsat_samples=True)

        ## Boundaries of parameters should appear now
        space.show(green=True, red=True)


if __name__ == "__main__":
    unittest.main()

