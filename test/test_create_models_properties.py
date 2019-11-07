import unittest

from termcolor import colored
from src.create_models_properties import *


class MyTestCase(unittest.TestCase):
    def test_create_models(self):
        print(colored('Create models here', 'blue'))
        create_bee_multiparam_synchronous_model("bee_multiparam_synchronous_" + str(2), 2)
        create_bee_multiparam_synchronous_model("bee_multiparam_synchronous_" + str(3), 3)
        create_bee_multiparam_synchronous_model("bee_multiparam_synchronous_" + str(4), 4)

    def test_create_properties(self):
        print(colored('Create properties here', 'blue'))
        create_properties(2)
        create_properties(3)
        create_properties(4)


if __name__ == '__main__':
    unittest.main()
