import unittest
from src.common.my_z3 import *


class MyTestCase(unittest.TestCase):
    def test_z3_eval(self):
        self.assertEqual(z3_eval("8+3"), 11)
        ## TODO

    def test_is_this_z3_function(self):
        self.assertEqual(is_this_z3_function("(1/10)**n"), False)
        self.assertEqual(is_this_z3_function("6 ** r"), False)
        self.assertEqual(is_this_z3_function("(1/10)*n + 6*6"), False)
        self.assertEqual(is_this_z3_function("2**2"), False)
        self.assertEqual(is_this_z3_function(
            "(-210)*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )**6*r_0**4+840*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )"),
                         True)

    def test_is_this_python_function(self):
        self.assertEqual(is_this_python_function("(1/10)**n"), True)
        self.assertEqual(is_this_python_function("6 ** r"), True)
        self.assertEqual(is_this_python_function("(1/10)*n + 6*6"), True)
        self.assertEqual(is_this_python_function("2**2"), True)
        self.assertEqual(is_this_python_function("(-210)*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )**6*r_0**4+840*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )"), False)

    def test_is_this_exponential_function(self):
        self.assertEqual(is_this_exponential_function("((1/10)**n"), True)
        self.assertEqual(is_this_exponential_function("6 ** r"), True)
        self.assertEqual(is_this_exponential_function("((1/10)*n + 6*6"), False)
        self.assertEqual(is_this_exponential_function("2**2"), False)

    def test_is_this_general_function(self):
        self.assertEqual(is_this_general_function("(1/10)**n"), False)
        self.assertEqual(is_this_general_function("6 ** r"), False)
        self.assertEqual(is_this_general_function("(1/10)*n + 6*6"), False)
        self.assertEqual(is_this_general_function("2**2"), False)
        self.assertEqual(is_this_general_function("(-210)*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )**6*r_0**4+840*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )"), False)
        self.assertEqual(is_this_general_function("("), True)

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
        function = "(-210)*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )**6*r_0**4+840*If(r_0 + 4*delta > 1, 1, r_0 + 4*delta )"
        print(function)
        print(translate_z3_function(function))
        print(translate_to_z3_function(translate_z3_function(function)))

    def test_parse_model_values(self):
        model = '[r_0 = 1/8, r_1 = 9/16, /0 = [(7/16, 7/8) -> 1/2, else -> 0]]'
        self.assertEqual(parse_model_values(model, "z3"), [0.125, 0.5625])


if __name__ == '__main__':
    unittest.main()
