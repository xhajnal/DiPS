import unittest
from z3 import *
from z3 import Solver, Real

from common.solver_parser import parse_model_values, pass_models_to_sons


class MyTestCase(unittest.TestCase):
    def test_parse_model_values(self):
        # model = '[r_0 = 1/8, r_1 = 9/16, /0 = [(7/16, 7/8) -> 1/2, else -> 0]]'
        x = Real('x')
        y = Real('y')
        s = Solver()
        s.add(x > 0)
        s.add(x < 2)
        s.add(y > 0)
        s.add(y < 100)
        print("s.check()", s.check())
        model = s.model()
        print("s.model()", model)
        model_values = parse_model_values(model, "z3")
        self.assertEqual(model_values, [1, 1])

        # Model used as an example
        ## model (1) is below threshold
        self.assertEqual(pass_models_to_sons(model, False, 0, 2, "z3"), ([model, None], [None, None]))
        self.assertEqual(pass_models_to_sons(model, False, 1, 2, "z3"), ([model, None], [None, None]))
        ## model (1) is above threshold
        self.assertEqual(pass_models_to_sons(model, False, 0, 0, "z3"), ([None, None], [model, None]))
        self.assertEqual(pass_models_to_sons(model, False, 1, 0, "z3"), ([None, None], [model, None]))
        ## model (1) is equal to threshold
        self.assertEqual(pass_models_to_sons(model, False, 0, 1, "z3"), ([None, None], [None, None]))
        self.assertEqual(pass_models_to_sons(model, False, 1, 1, "z3"), ([None, None], [None, None]))

        # Now as a counterexample
        ## model (1) is below threshold
        self.assertEqual(pass_models_to_sons(False, model, 0, 2, "z3"), ([None, model], [None, None]))
        self.assertEqual(pass_models_to_sons(False, model, 1, 2, "z3"), ([None, model], [None, None]))
        ## model (1) is above threshold
        self.assertEqual(pass_models_to_sons(False, model, 0, 0, "z3"), ([None, None], [None, model]))
        self.assertEqual(pass_models_to_sons(False, model, 1, 0, "z3"), ([None, None], [None, model]))
        ## model (1) is equal to threshold
        self.assertEqual(pass_models_to_sons(False, model, 0, 1, "z3"), ([None, None], [None, None]))
        self.assertEqual(pass_models_to_sons(False, model, 1, 1, "z3"), ([None, None], [None, None]))

        # Now as both, example and counterexample
        ## model (1) is below threshold
        self.assertEqual(pass_models_to_sons(model, model, 0, 2, "z3"), ([model, model], [None, None]))
        self.assertEqual(pass_models_to_sons(model, model, 1, 2, "z3"), ([model, model], [None, None]))
        ## model (1) is above threshold
        self.assertEqual(pass_models_to_sons(model, model, 0, 0, "z3"), ([None, None], [model, model]))
        self.assertEqual(pass_models_to_sons(model, model, 1, 0, "z3"), ([None, None], [model, model]))
        ## model (1) is equal to threshold
        self.assertEqual(pass_models_to_sons(model, model, 0, 1, "z3"), ([None, None], [None, None]))
        self.assertEqual(pass_models_to_sons(model, model, 1, 1, "z3"), ([None, None], [None, None]))

        x = Real('x')
        y = Real('y')
        s = Solver()
        s.add(x > 0)
        s.add(x < 2)
        s.add(y > 0)
        s.add(y < 100)
        s.add(eval('((1/2)**y / (x + (1/2)**y))**2-2*((1/2)**y / (x + (1/2)**y))+1>=0.726166837943366'))
        print("s.check()", s.check())
        model = s.model()
        print("s.model()", model)
        model_values = parse_model_values(model, "z3")
        self.assertEqual(model_values, [0, 0])

        r_1 = Real("r_1")
        r_0 = Real("r_0")
        s = Solver()
        s.add(r_0 > 0)
        s.add(r_0 < 1)
        s.add(r_1 > 0)
        s.add(r_1 < 1)

        a = ['(r_0 - 1)**2>=0.726166837943366', '(r_0 - 1)**2<=0.907166495456634',
             '2*r_0*(If(r_1 > r_0, (r_1 - r_0)/(1 - r_0), 0) - 1)*(r_0 - 1)>=0.0401642685583240',
             '2*r_0*(If(r_1 > r_0, (r_1 - r_0)/(1 - r_0), 0) - 1)*(r_0 - 1)<=0.193169064841676',
             '-r_0*(2*If(r_1 > r_0, (r_1 - r_0)/(1 - r_0), 0)*r_0 - 2*If(r_1 > r_0, (r_1 - r_0)/(1 - r_0), 0) - r_0)>=0.00536401422323720',
             '-r_0*(2*If(r_1 > r_0, (r_1 - r_0)/(1 - r_0), 0)*r_0 - 2*If(r_1 > r_0, (r_1 - r_0)/(1 - r_0), 0) - r_0)<=0.127969319116763']

        for item in a:
            s.add(eval(item))

        print("s.check()", s.check())
        model = s.model()
        print("s.model()", model)
        model_values = parse_model_values(model, "z3")
        self.assertEqual(model_values, [1/8, 9/16])

        ## TODO dreal

    def test_pass_models_to_sons(self):
        ## Done in the the test above
        pass


if __name__ == '__main__':
    unittest.main()
