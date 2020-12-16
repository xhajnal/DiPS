import unittest
import warnings
from performance_test.generate_data import *
import numpy


class MyTestCase(unittest.TestCase):
    def test(self):
        warnings.warn("This test does not contain any assert, only non-halting of the code is tested, you can check the files", RuntimeWarning)
        cwd = os.getcwd()
        model_types = ["asynchronous_"]
        n_samples = [3, 2]
        populations = [2]
        dimension_sample_size = 4
        #               def generate_experiments_and_data(model_types, n_samples, populations, dimension_sample_size, sim_length=False, modular_param_space=None, input_folder=False, output_folder=False, silent=False, debug=False):
        debug, debug2 = generate_experiments_and_data(model_types, n_samples, populations, dimension_sample_size, modular_param_space=None, input_folder=os.path.join(cwd, "models"), output_folder=os.path.join(cwd, "tmp"), silent=False, debug=True)
        print(debug)

        p_values = [0.028502714675268215, 0.45223461506339047, 0.8732745414252937, 0.6855555397734584, 0.13075717833714784]
        q_values = [0.5057623641293089, 0.29577906622244676, 0.8440550299528644, 0.8108008054929994, 0.03259111103419188]

        default_2dim_param_space = numpy.zeros((2, 5))
        default_2dim_param_space[0] = p_values
        default_2dim_param_space[1] = q_values

        debug, debug2 = generate_experiments_and_data(["asynchronous_"], [3, 2], [2], 4, modular_param_space=default_2dim_param_space, input_folder=os.path.join(cwd, "models"), output_folder=os.path.join(cwd, "tmp"), silent=False, debug=True)
        print(debug["asynchronous_"][2][3][(0.45223461506339047, 0.29577906622244676)])
        print(debug2["asynchronous_"][2][3][(0.45223461506339047, 0.29577906622244676)])


if __name__ == '__main__':
    unittest.main()
