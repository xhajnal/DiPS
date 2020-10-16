import unittest
from termcolor import colored


class MyTestCase(unittest.TestCase):
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
