import unittest
from src.create_models_props import *


class MyTestCase(unittest.TestCase):
    def test(self):
        create_properties(2)
        create_properties(3)
        create_properties(4)
        create_bee_multiparam_synchronous_model("bee_multiparam_synchronous_" + str(2), 2)
        create_bee_multiparam_synchronous_model("bee_multiparam_synchronous_" + str(3), 3)
        create_bee_multiparam_synchronous_model("bee_multiparam_synchronous_" + str(4), 4)


if __name__ == '__main__':
    unittest.main()