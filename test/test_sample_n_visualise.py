import unittest
from time import sleep

from sympy import Interval

from src.sample_n_visualise import *


class MyTestCase(unittest.TestCase):
    def test_bar_err_plot(self):
        bar_err_plot([5])

        bar_err_plot([5], [[0, 7]])

        bar_err_plot([5], titles=["Data indices", "Data values", "single point"])

        bar_err_plot([5], [[0, 7]], ["Data indices", "Data values", "single point, single interval"])

        bar_err_plot([5, 6], [], ["Data indices", "Data values", "two points"])

        bar_err_plot([5, 6], [[0, 7], [5.7, 6.8]], ["Data indices", "Data values", "two points two intervals"])

    def test_eval_and_show(self):
        ## returns [N, dic_fun[N].index(polynome), datapoint]

        print(sample_dictionary_funs({10: ["x+y"]}, 2, [10]))

        # heatmap("p+0*q",[[1,5],[1,5]],[6,6])

        r_0 = 0.209
        delta = 0.036
        spam = ["r_0**2-2*r_0+1", "(-2)*If(r_0 + 1*delta > 1, 1, r_0 + 1*delta)*r_0+2*r_0", "(-1)*r_0**2+2*If(r_0 + 1*delta > 1, 1, r_0 + 1*delta)*r_0"]

        # eval_and_show(fun_list, parameter_value, parameters=False, data=False, intervals=False, cumulative=False, debug=False, where=False):
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], debug=True)

        ## cdf instead of pdf
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], cumulative=True, debug=True)

        ## With data
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], data=[0.1, 0.2, 0.7], debug=True)

        ## Cumulative With data
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], data=[0.1, 0.2, 0.7], cumulative=True, debug=True)

        ## With data and intervals
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], data=[0.1, 0.2, 0.7], data_intervals=[[0, 1], [0.1, 0.2], [0.1, 0.2]], debug=True)

        ## With data and intervals
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], data=[0.1, 0.2, 0.7], data_intervals=[Interval(0, 1), Interval(0.1, 0.2), Interval(0.1, 0.2)], debug=True)

        spam = ["r_0**2-2*r_0+1"]

        ## TODO tweak fix this (a single function makes different bar)
        eval_and_show(spam, [delta, r_0], ["delta", "r_0"], debug=True)

    def test_heatmap(self):
        # heatmap(fun, region, sampling_sizes, posttitle="", where=False, parameters=False)
        heatmap("p+q", [[0, 1], [0, 1]], [5, 5])

        ## more refined
        heatmap("p+q", [[0, 1], [0, 1]], [10, 10])

    def test_visualise_by_param(self):
        hyper_rectangles_sat = [[(0.5, 0.5625), (0.125, 0.25)], [(0.5, 0.5625), (0.0625, 0.125)],
                                [(0.5625, 0.625), (0, 0.0625)], [(0.46875, 0.5), (0.125, 0.1875)],
                                [(0.46875, 0.5), (0.1875, 0.25)]]

        # visualise_by_param(hyper_rectangles, title="", where=False)
        visualise_by_param(hyper_rectangles_sat)

    def test_visualise_sampled_by_param(self):
        hyper_rectangles_sat = [[(0.5, 0.5625), (0.125, 0.25)], [(0.5, 0.5625), (0.0625, 0.125)],
                                [(0.5625, 0.625), (0, 0.0625)], [(0.46875, 0.5), (0.125, 0.1875)],
                                [(0.46875, 0.5), (0.1875, 0.25)]]

        # visualise_sampled_by_param(hyper_rectangles, sample_size)
        visualise_sampled_by_param(hyper_rectangles_sat, 8)


if __name__ == '__main__':
    unittest.main()

