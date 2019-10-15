import unittest
from src.generate_data import *
import numpy


class MyTestCase(unittest.TestCase):
    def test(self):
        model_types = ["synchronous_parallel_"]
        n_samples = [3, 2]
        populations = [10]
        dimension_sample_size = 5
        Debug, Debug2 = generate_experiments_and_data(model_types, n_samples, populations, 4, None, True)
        print(Debug)

        p_values = [0.028502714675268215, 0.45223461506339047, 0.8732745414252937, 0.6855555397734584, 0.13075717833714784]
        q_values = [0.5057623641293089, 0.29577906622244676, 0.8440550299528644, 0.8108008054929994, 0.03259111103419188]

        default_2dim_param_space = numpy.zeros((2, 5))
        default_2dim_param_space[0] = p_values
        default_2dim_param_space[1] = q_values

        Debug, Debug2 = generate_experiments_and_data(["synchronous_parallel_"], [3, 2], [2], 4, default_2dim_param_space, False)
        print(Debug["synchronous_parallel_"][2][3][(0.45223461506339047, 0.29577906622244676)])
        print(Debug2["synchronous_parallel_"][2][3][(0.45223461506339047, 0.29577906622244676)])


if __name__ == '__main__':
    unittest.main()
