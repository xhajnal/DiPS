import unittest
from src.sample_n_visualise import *


class MyTestCase(unittest.TestCase):
    def test(self):
        ## returns [N, dic_fun[N].index(polynome), datapoint]
        print(sample_dictionary_funs({10: ["x+y"]}, 2, [10]))

        # heatmap("p+0*q",[[1,5],[1,5]],[6,6])
        heatmap("p+q", [[0, 1], [0, 1]], [5, 5])
        hyper_rectangles_sat = [[(0.5, 0.5625), (0.125, 0.25)], [(0.5, 0.5625), (0.0625, 0.125)],
                                [(0.5625, 0.625), (0, 0.0625)], [(0.46875, 0.5), (0.125, 0.1875)],
                                [(0.46875, 0.5), (0.1875, 0.25)]]
        visualise_by_param(hyper_rectangles_sat)
        visualise_sampled_by_param(hyper_rectangles_sat, 8)


if __name__ == '__main__':
    unittest.main()

