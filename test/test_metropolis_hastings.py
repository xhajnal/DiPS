import unittest
from termcolor import colored
from src.metropolis_hastings import *
import os

cwd = os.getcwd()

try:
    os.mkdir("tmp")
except FileExistsError:
    pass

tmp_dir = model_dir = os.path.join(cwd, "tmp")


class MyTestCase(unittest.TestCase):
    def test_MH_class(self):
        ##  TODO
        pass

    def test_sample_functions(self):
        ## Now deprecated
        pass

    def test_get_truncated_normal(self):
        ##  TODO
        # get_truncated_normal(mean=0.0, sd=1.0, low=0.0, upp=10.0)
        pass

    def test_transition_model_a(self):
        for i in range(10):
            for b in range(1, 100):
                b = b/10
                spam = transition_model([i], [[i - b, i + b]])
                self.assertTrue(spam[0] > i - b)
                self.assertTrue(spam[0] < i + b)

    def test_prior(self):
        ## Now deprecated
        pass

    def test_acceptance(self):
        self.assertTrue(acceptance_rule(8, 9))

    def test_manual_log_like_normal(self):
        params = ["x"]
        parameter_intervals = [(0, 1)]
        functions = ["x+0.9"]
        sample_size = 10
        precision = 4
        #                      manual_log_like_normal(params, theta, functions, data, sample_size, eps=0, parallel=False, debug=False)
        self.assertEqual(round(manual_log_like_normal(params, [0], functions, [0.9], sample_size, eps=0, parallel=True, debug=True), precision), round(-3.250829733914482, precision))
        self.assertEqual(round(manual_log_like_normal(params, [0], functions, [0.8], sample_size, eps=0, parallel=True, debug=True), precision), round(-5.448054311250701, precision))

    def test_metropolis_hastings(self):
        warnings.warn("This test does not contain any assert as it is nondeterministic, please check the results manually", RuntimeWarning)
        params = ["x"]
        parameter_intervals = [(0, 1)]
        functions = ["x"]
        data = [0.2]
        #      metropolis_hastings(params, parameter_intervals, param_init, functions, data, sample_size, iterations, eps, sd, progress=False, timeout=0, debug=False)
        spam = metropolis_hastings(params, parameter_intervals, [0.5], functions, data, 10, 50, eps=0, progress=False, timeout=0, debug=False)
        print()
        print("accepted", spam[0])
        print("rejected", spam[1])

    def test_without_data_nor_observation(self):
        print(colored('Metropolis-Hastings without data - it is sampled', 'blue'))
        metropolis_hastings.tmp_dir = tmp_dir
        params = ["x", "y"]
        parameter_intervals = [(0, 1), (0, 1)]

        g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        g[0] = '(x - 1)**10'
        g[1] = '10*x*(x - 1)**9*(y - 1)**9'
        g[2] = '-45*x*(x - 1)**8*(y - 1)**8*(2*x*y - x - 2*y)'
        g[3] = '120*x*(x - 1)**7*(y - 1)**7*(3*x**2*y**2 - 3*x**2*y + x**2 - 6*x*y**2 + 3*x*y + 3*y**2)'
        g[4] = '-210*x*(x - 1)**6*(y - 1)**6*(2*x*y - x - 2*y)*(2*x**2*y**2 - 2*x**2*y + x**2 - 4*x*y**2 + 2*x*y + 2*y**2)'
        g[5] = '252*x*(x - 1)**5*(y - 1)**5*(5*x**4*y**4 - 10*x**4*y**3 + 10*x**4*y**2 - 5*x**4*y + x**4 - 20*x**3*y**4 + 30*x**3*y**3 - 20*x**3*y**2 + 5*x**3*y + 30*x**2*y**4 - 30*x**2*y**3 + 10*x**2*y**2 - 20*x*y**4 + 10*x*y**3 + 5*y**4)'
        g[6] = '-210*x*(x - 1)**4*(y - 1)**4*(2*x*y - x - 2*y)*(x**2*y**2 - x**2*y + x**2 - 2*x*y**2 + x*y + y**2)*(3*x**2*y**2 - 3*x**2*y + x**2 - 6*x*y**2 + 3*x*y + 3*y**2)'
        g[7] = '120*x*(x - 1)**3*(y - 1)**3*(7*x**6*y**6 - 21*x**6*y**5 + 35*x**6*y**4 - 35*x**6*y**3 + 21*x**6*y**2 - 7*x**6*y + x**6 - 42*x**5*y**6 + 105*x**5*y**5 - 140*x**5*y**4 + 105*x**5*y**3 - 42*x**5*y**2 + 7*x**5*y + 105*x**4*y**6 - 210*x**4*y**5 + 210*x**4*y**4 - 105*x**4*y**3 + 21*x**4*y**2 - 140*x**3*y**6 + 210*x**3*y**5 - 140*x**3*y**4 + 35*x**3*y**3 + 105*x**2*y**6 - 105*x**2*y**5 + 35*x**2*y**4 - 42*x*y**6 + 21*x*y**5 + 7*y**6)'
        g[8] = '-45*x*(x - 1)**2*(y - 1)**2*(2*x*y - x - 2*y)*(2*x**2*y**2 - 2*x**2*y + x**2 - 4*x*y**2 + 2*x*y + 2*y**2)*(2*x**4*y**4 - 4*x**4*y**3 + 6*x**4*y**2 - 4*x**4*y + x**4 - 8*x**3*y**4 + 12*x**3*y**3 - 12*x**3*y**2 + 4*x**3*y + 12*x**2*y**4 - 12*x**2*y**3 + 6*x**2*y**2 - 8*x*y**4 + 4*x*y**3 + 2*y**4)'
        g[9] = '10*x*(x - 1)*(y - 1)*(3*x**2*y**2 - 3*x**2*y + x**2 - 6*x*y**2 + 3*x*y + 3*y**2)*(3*x**6*y**6 - 9*x**6*y**5 + 18*x**6*y**4 - 21*x**6*y**3 + 15*x**6*y**2 - 6*x**6*y + x**6 - 18*x**5*y**6 + 45*x**5*y**5 - 72*x**5*y**4 + 63*x**5*y**3 - 30*x**5*y**2 + 6*x**5*y + 45*x**4*y**6 - 90*x**4*y**5 + 108*x**4*y**4 - 63*x**4*y**3 + 15*x**4*y**2 - 60*x**3*y**6 + 90*x**3*y**5 - 72*x**3*y**4 + 21*x**3*y**3 + 45*x**2*y**6 - 45*x**2*y**5 + 18*x**2*y**4 - 18*x*y**6 + 9*x*y**5 + 3*y**6)'
        g[10] = '-x*(2*x*y - x - 2*y)*(x**4*y**4 - 2*x**4*y**3 + 4*x**4*y**2 - 3*x**4*y + x**4 - 4*x**3*y**4 + 6*x**3*y**3 - 8*x**3*y**2 + 3*x**3*y + 6*x**2*y**4 - 6*x**2*y**3 + 4*x**2*y**2 - 4*x*y**4 + 2*x*y**3 + y**4)*(5*x**4*y**4 - 10*x**4*y**3 + 10*x**4*y**2 - 5*x**4*y + x**4 - 20*x**3*y**4 + 30*x**3*y**3 - 20*x**3*y**2 + 5*x**3*y + 30*x**2*y**4 - 30*x**2*y**3 + 10*x**2*y**2 - 20*x*y**4 + 10*x*y**3 + 5*y**4)'

        #                  (params, parameter_intervals, functions,    data,  sample_size,      mh_sampling_iterations, eps=0, sd=0.15, theta_init=False, where=False, progress=False, burn_in=False, bins=20, timeout=False, debug=False, metadata=True, draw_plot=False)
        init_mh(params, parameter_intervals, functions=g, data=[], sample_size=100, mh_sampling_iterations=100, eps=0, debug=True)

    def test_without_data_nor_observation2(self):
        print(colored('Metropolis-Hastings without data - it is sampled', 'blue'))
        metropolis_hastings.tmp_dir = tmp_dir
        params = ["x", "y"]
        parameter_intervals = [(0, 1), (0, 1)]
        f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
        init_mh(params, parameter_intervals, data=[], functions=f, sample_size=100, mh_sampling_iterations=100, eps=0, debug=True)

    def test_given_observation(self):
        pass  ## current implementation allows only data
        # print(colored('Metropolis-Hastings with observations', 'blue'))
        # space = RefinedSpace([(0, 1), (0, 1)], ["p", "q"], ["Real", "Real"], [[[0, 0.5], [0, 0.5]]], [], true_point=[0.82, 0.92])
        # f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
        # init_mh(space, data=[0, 2, 1, 2, 1, 0, 2, 1, 0, 1], functions=f, observations_count=500, observations_samples_size=100, mh_sampling_iterations=100, eps=0, debug=True) ## using observations

    def test_given_data(self):
        print(colored('Metropolis-Hastings with data', 'blue'))
        metropolis_hastings.tmp_dir = tmp_dir
        params = ["p", "q"]
        parameter_intervals = [(0, 1), (0, 1)]
        functions = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
        data = [0.2, 0.5, 0.3]
        init_mh(params, parameter_intervals, data=data, functions=functions, sample_size=100, mh_sampling_iterations=100, eps=0, debug=True, is_probability=True)

    def test_compute_hpd_univariate(self):
        print(colored('compute hpd univariate, randomised input - please check manually', 'blue'))
        from scipy.stats import norm
        trace = norm.rvs(size=100)
        print(trace)
        hpd = compute_hpd_univariate(trace, 0.95)
        print(hpd)

    def test_compute_hpd_multivariate(self):
        print(colored('compute hpd univariate, randomised input - please check manually', 'blue'))
        from scipy.stats import norm
        dim = 5
        trace_len = 100
        trace = np.zeros((trace_len, dim))
        for i in range(0, dim):
            trace[:, i] = norm.rvs(size=100)
        print(trace)
        hpd = compute_hpd_multivariate(trace, dim, 0.95)
        print(hpd)


if __name__ == '__main__':
    unittest.main()
