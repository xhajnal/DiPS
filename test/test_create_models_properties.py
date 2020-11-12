import unittest
import warnings

from termcolor import colored
from src.create_models_properties import *

spam = load_config()
model_path = spam["models"]
properties_path = spam["properties"]
del spam


class MyTestCase(unittest.TestCase):
    def test_create_models(self):
        warnings.warn("This test does not contain any assert, only non-halting of the code is tested, you can check the files", RuntimeWarning)
        print(colored('Create models here', 'blue'))
        for population in [2, 3, 4]:
            create_bee_multiparam_synchronous_model(f"bee_multiparam_synchronous_{population}", population)
            ## Try to open the file
            with open(os.path.join(model_path, f"bee_multiparam_synchronous_{population}.pm")) as file:
                pass

    def test_create_properties(self):
        warnings.warn("This test does not contain any assert, only non-halting of the code is tested, you can check the files", RuntimeWarning)
        print(colored('Create properties here', 'blue'))
        for population in [2, 3, 4]:
            create_properties(population)
            ## Try to open the file
            with open(os.path.join(properties_path, f"prop_{population}.pctl")) as file:
                pass


if __name__ == '__main__':
    unittest.main()
