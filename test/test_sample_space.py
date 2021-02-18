import os
import unittest

from sympy import Interval
from termcolor import colored

import src.sample_space as sample_space
from common.convert import normalise_constraint, split_constraints
from space import RefinedSpace
from src.refine_space import *

curr_dir = os.path.dirname(__file__)


class MyTestCase(unittest.TestCase):
    def test_check_sample(self):
        print(colored("Testing check sample", 'blue'))
        ## Single constraint
        sample_space.glob_sort = False
        sample_space.glob_space = RefinedSpace(((0, 1), (0, 1)), ["p", "q"])
        sample_space.glob_debug = False
        sample_space.glob_compress = True
        sample_space.glob_constraints = ["0.3 < p+q < 0.8"]
        self.assertEqual(sample_space.check_sample([0, 0]), False)
        self.assertEqual(sample_space.check_sample([0, 0.5]), True)
        self.assertEqual(sample_space.check_sample([0.5, 0]), True)
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), False)
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), True)

        sample_space.glob_compress = False
        self.assertEqual(sample_space.check_sample([0, 0]), [False])
        self.assertEqual(sample_space.check_sample([0, 0.5]), [True])
        self.assertEqual(sample_space.check_sample([0.5, 0]), [True])
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), [False])
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), [True])

        ## Two constraints
        sample_space.glob_constraints = ["0.3 < p+q < 0.8", "p>q"]
        sample_space.glob_compress = True
        self.assertEqual(sample_space.check_sample([0, 0]), False)
        self.assertEqual(sample_space.check_sample([0, 0.5]), False)
        self.assertEqual(sample_space.check_sample([0.5, 0]), True)
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), False)
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), False)

        sample_space.glob_compress = False
        self.assertEqual(sample_space.check_sample([0, 0]), [False, False])
        self.assertEqual(sample_space.check_sample([0, 0.5]), [True, False])
        self.assertEqual(sample_space.check_sample([0.5, 0]), [True, True])
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), [False, False])
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), [True, False])

        ## Error - Division by zero
        sample_space.glob_constraints = ["0.3 < p/q < 0.8", "p>q"]
        sample_space.glob_compress = True
        self.assertEqual(sample_space.check_sample([0, 0]), None)
        sample_space.glob_compress = False
        self.assertEqual(sample_space.check_sample([0, 0]), [None, False])

        ## Single sided constraint
        sample_space.glob_space = RefinedSpace([(0, 1), (0, 1)], ["p", "q"])
        sample_space.glob_constraints = ["p < q"]
        sample_space.glob_compress = True
        self.assertEqual(sample_space.check_sample([0, 0]), False)
        self.assertEqual(sample_space.check_sample([0, 0.5]), True)
        self.assertEqual(sample_space.check_sample([0.5, 0]), False)
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), False)
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), False)

        sample_space.glob_compress = False
        self.assertEqual(sample_space.check_sample([0, 0]), [False])
        self.assertEqual(sample_space.check_sample([0, 0.5]), [True])
        self.assertEqual(sample_space.check_sample([0.5, 0]), [False])
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), [False])
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), [False])

        sample_space.glob_constraints = ["p > q"]
        sample_space.glob_compress = True
        self.assertEqual(sample_space.check_sample([0, 0]), False)
        self.assertEqual(sample_space.check_sample([0, 0.5]), False)
        self.assertEqual(sample_space.check_sample([0.5, 0]), True)
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), False)
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), False)

        sample_space.glob_compress = False
        self.assertEqual(sample_space.check_sample([0, 0]), [False])
        self.assertEqual(sample_space.check_sample([0, 0.5]), [False])
        self.assertEqual(sample_space.check_sample([0.5, 0]), [True])
        self.assertEqual(sample_space.check_sample([0.5, 0.5]), [False])
        self.assertEqual(sample_space.check_sample([0.3, 0.3]), [False])

    def test_sample_sat_degree(self):
        print(colored("Testing satisfaction degree of a sample", 'blue'))
        sample_space.glob_sort = False
        sample_space.glob_space = RefinedSpace((0, 1), ["p"])
        sample_space.glob_debug = False
        sample_space.glob_compress = True

        constraints = ["0.3 < p < 0.8"]
        constraints = list(map(normalise_constraint, constraints))
        constraints = split_constraints(constraints)
        sample_space.glob_constraints = constraints

        self.assertEqual(sample_space.sample_sat_degree([0]), - 0.3)
        self.assertEqual(sample_space.sample_sat_degree([0.5]), 0.2)
        self.assertEqual(sample_space.sample_sat_degree([0.3]), 0)
        self.assertEqual(round(sample_space.sample_sat_degree([0.9]), 2), -0.1)

        constraints = ["0.3 <= p = 0.8"]
        constraints = list(map(normalise_constraint, constraints))
        constraints = split_constraints(constraints)
        sample_space.glob_constraints = constraints

        self.assertEqual(sample_space.sample_sat_degree([0]), - 0.3)
        self.assertEqual(sample_space.sample_sat_degree([0.5]), 0.2)
        self.assertEqual(sample_space.sample_sat_degree([0.3]), 0)
        self.assertEqual(round(sample_space.sample_sat_degree([0.9]), 2), -0.1)

        sample_space.glob_compress = False
        self.assertEqual(sample_space.sample_sat_degree([0]), [- 0.3])
        self.assertEqual(sample_space.sample_sat_degree([0.5]), [0.2])
        self.assertEqual(sample_space.sample_sat_degree([0.3]), [0])
        self.assertEqual(list(map(lambda x: round(x, 2), sample_space.sample_sat_degree([0.9]))), [-0.1])

        sample_space.glob_space = RefinedSpace(((0, 1), (0, 1)), ["p", "q"])
        constraints = ["0.3 < p+q < 0.8", "p > q"]
        constraints = list(map(normalise_constraint, constraints))
        print(constraints)
        constraints = split_constraints(constraints)
        print(constraints)
        sample_space.glob_constraints = constraints
        sample_space.sample_sat_degree([0, 0])

        sample_space.glob_compress = True
        self.assertEqual(round(sample_space.sample_sat_degree([0, 0]), 2), -0.3+0)
        self.assertEqual(round(sample_space.sample_sat_degree([0, 0.5]), 2), 0.2-0.5)
        self.assertEqual(round(sample_space.sample_sat_degree([0.5, 0]), 2), 0.2+0.5)
        self.assertEqual(round(sample_space.sample_sat_degree([0.5, 0.5]), 2), -0.2+0)
        self.assertEqual(round(sample_space.sample_sat_degree([0.3, 0.3]), 2), 0.2+0)

        sample_space.glob_compress = False
        self.assertEqual(list(map(lambda x: round(x, 2), sample_space.sample_sat_degree([0, 0]))), [-0.3, 0])
        self.assertEqual(list(map(lambda x: round(x, 2), sample_space.sample_sat_degree([0, 0.5]))), [0.2, -0.5])
        self.assertEqual(list(map(lambda x: round(x, 2), sample_space.sample_sat_degree([0.5, 0]))), [0.2, 0.5])
        self.assertEqual(list(map(lambda x: round(x, 2), sample_space.sample_sat_degree([0.5, 0.5]))), [-0.2, 0])
        self.assertEqual(list(map(lambda x: round(x, 2), sample_space.sample_sat_degree([0.3, 0.3]))), [0.2, 0])

        sample_space.glob_space = RefinedSpace([(0, 1), (0, 1)], ["p", "q"])
        constraints = ["p < q"]
        constraints = list(map(normalise_constraint, constraints))
        constraints = split_constraints(constraints)
        sample_space.glob_constraints = constraints

        sample_space.glob_compress = True
        self.assertEqual(sample_space.sample_sat_degree([0, 0]), 0)
        self.assertEqual(sample_space.sample_sat_degree([4, 5]), 1)
        self.assertEqual(sample_space.sample_sat_degree([5, 4]), -1)
        sample_space.glob_compress = False
        self.assertEqual(sample_space.sample_sat_degree([0, 0]), [0])
        self.assertEqual(sample_space.sample_sat_degree([4, 5]), [1])
        self.assertEqual(sample_space.sample_sat_degree([5, 4]), [-1])

        constraints = ["p > q"]
        constraints = list(map(normalise_constraint, constraints))
        constraints = split_constraints(constraints)
        sample_space.glob_constraints = constraints

        sample_space.glob_compress = True
        self.assertEqual(sample_space.sample_sat_degree([0, 0]), 0)
        self.assertEqual(sample_space.sample_sat_degree([4, 5]), -1)
        self.assertEqual(sample_space.sample_sat_degree([5, 4]), 1)
        sample_space.glob_compress = False
        self.assertEqual(sample_space.sample_sat_degree([0, 0]), [0])
        self.assertEqual(sample_space.sample_sat_degree([4, 5]), [-1])
        self.assertEqual(sample_space.sample_sat_degree([5, 4]), [1])



    def test_space_sample(self):
        print(colored("Sampling space test here", 'blue'))
        ## Initialisation
        space = RefinedSpace((0, 1), ["x"], ["Real"], [Interval(0, 1)])
        # print(space.nice_print())

        debug = False

        constraints1 = ["x<3"]
        constraints2 = ["x>3"]

        constraints3 = ["x>3", "x<3"]
        constraints4 = ["x<=1", "x>=0"]
        constraints5 = ["x>3", "x>=0"]

        ## def sample(space, constraints, size_q, compress)
        ## Example:  sample(space, constraints1, 1, debug=debug)
        # a = sample(space, constraints1, 1, debug=debug)

        self.assertEqual(sample(space, constraints1, 1, debug=debug)[0][0], True)
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 0)
        self.assertEqual(sample(space, constraints2, 1, debug=debug)[0][0], False)
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 1)
        self.assertEqual(sample(space, constraints3, 1, compress=True, debug=debug)[0], False)
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 2)
        self.assertEqual(sample(space, constraints4, 1, compress=True, debug=debug)[0], True)
        self.assertEqual(len(space.get_sat_samples()), 2)
        self.assertEqual(len(space.get_unsat_samples()), 2)
        self.assertEqual(sample(space, constraints5, 1, compress=True, debug=debug)[0], False)
        self.assertEqual(len(space.get_sat_samples()), 2)
        self.assertEqual(len(space.get_unsat_samples()), 3)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"])

        constraints3 = ["x+y>3", "x+y<3"]
        constraints4 = ["x+y<=1", "x+y>=0"]
        constraints5 = ["x+y>3", "x+y>=0"]

        self.assertEqual(sample(space, constraints3, 1, compress=True, debug=debug)[0], False)
        self.assertEqual(len(space.get_sat_samples()), 0)
        self.assertEqual(len(space.get_unsat_samples()), 1)
        self.assertEqual(sample(space, constraints4, 1, compress=True, debug=debug)[0], True)
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 1)
        self.assertEqual(sample(space, constraints5, 1, compress=True, debug=debug)[0], False)
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 2)

        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"])

        constraints3 = ["x+y>3", "x+y<3"]
        constraints4 = ["x+y<=1", "x+y>=0"]
        constraints5 = ["x+y>3", "x+y>=0"]

        self.assertEqual(sample(space, constraints3, 1, debug=debug)[0], [False, True])
        self.assertEqual(len(space.get_sat_samples()), 0)
        self.assertEqual(len(space.get_unsat_samples()), 1)
        self.assertEqual(sample(space, constraints4, 1, debug=debug)[0], [True, True])
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 1)
        self.assertEqual(sample(space, constraints5, 1, debug=debug)[0], [False, True])
        self.assertEqual(len(space.get_sat_samples()), 1)
        self.assertEqual(len(space.get_unsat_samples()), 2)


if __name__ == "__main__":
    unittest.main()

