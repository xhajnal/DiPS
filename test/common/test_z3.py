import unittest
from src.common.z3 import *
from termcolor import colored


class MyTestCase(unittest.TestCase):
    def test_z3_eval(self):
        ## TODO
        pass

    def test_is_this_z3_function(self):
        ## TODO
        pass

    def test_is_this_python_function(self):
        ## TODO
        pass

    def test_is_this_exponential_function(self):
        self.assertEqual(is_this_exponential_function("((1/10)**n "), True)
        self.assertEqual(is_this_exponential_function("((1/10)*n + 6*6 "), False)

    def test_is_this_general_function(self):
        ## TODO
        pass

    def test_translate_z3_function(self):
        ## TODO
        pass

    def test_translate_to_z3_function(self):
        ## TODO
        pass

    def test_translate_z3_function(self):
        ## TODO
        pass


if __name__ == '__main__':
    unittest.main()
