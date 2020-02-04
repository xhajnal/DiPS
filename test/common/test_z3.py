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
        self.assertEqual(is_this_exponential_function("((1/10)**n"), True)
        self.assertEqual(is_this_exponential_function("6 ** r"), True)
        self.assertEqual(is_this_exponential_function("((1/10)*n + 6*6"), False)
        self.assertEqual(is_this_exponential_function("2**2"), False)

    def test_is_this_general_function(self):
        ## TODO
        pass

    def test_translate_z3_function(self):
        ## TODO
        function = "(-210)*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )**6*r_0**4+840*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )"

        print(translate_z3_function(function))
        # print(re.sub(r"If\(([^If]*),([^If]*),([^If]*)\)", r"(\g<2> if \g<1> else \g<3>)", function))
        r_0 = 0
        delta = 0
        print(eval(translate_z3_function(function)))
        pass

    def test_translate_to_z3_function(self):
        ## TODO
        pass


if __name__ == '__main__':
    unittest.main()
