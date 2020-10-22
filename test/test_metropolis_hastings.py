import unittest
from termcolor import colored
from src.metropolis_hastings import *


class MyTestCase(unittest.TestCase):
    def test_without_data_nor_observation(self):
        print(colored('Metropolis-Hastings without data - it is sampled', 'blue'))
        #                     (region, params, types=None, rectangles_sat=False, rectangles_unsat=False, rectangles_unknown=None, sat_samples=None, unsat_samples=None, dist_samples=False, true_point=False, title=False, prefer_unsafe=False):
        space = RefinedSpace([(0, 1), (0, 1)], ["x", "y"], types=["Real", "Real"], rectangles_sat=[[[0, 0.5], [0, 0.5]]], rectangles_unsat=[], true_point=[0.82, 0.92])
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

        # N = 5000  # total data
        # N_obs = 100  # samples
        # MH_samples = 50000
        # eps = 0  # very small value used as probability of non-feasible values in prior

        # initialise_sampling(space, [], g, 5000, 100, 50000, 0)
        #                  (space, observations,      functions, observations_count, observations_samples_size,        mh_sampling_iterations: int, eps, theta_init=False, where=False, progress=False, burn_in=False, bins=20, timeout=False, debug=False, metadata=True, draw_plot=False)
        initialise_sampling(space, data=[], functions=g, sample_size=100, mh_sampling_iterations=100, eps=0, debug=True)

    def test_without_data_nor_observation2(self):
        print(colored('Metropolis-Hastings without data - it is sampled', 'blue'))
        space = RefinedSpace([(0, 1), (0, 1)], ["p", "q"], ["Real", "Real"], [[[0, 0.5], [0, 0.5]]], [], true_point=[0.82, 0.92])
        f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
        initialise_sampling(space, data=[], functions=f, sample_size=100, mh_sampling_iterations=100, eps=0, debug=True)

    def test_given_observation(self):
        pass  ## current implementation allows only data
        # print(colored('Metropolis-Hastings with observations', 'blue'))
        # space = RefinedSpace([(0, 1), (0, 1)], ["p", "q"], ["Real", "Real"], [[[0, 0.5], [0, 0.5]]], [], true_point=[0.82, 0.92])
        # f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
        # initialise_sampling(space, data=[0, 2, 1, 2, 1, 0, 2, 1, 0, 1], functions=f, observations_count=500, observations_samples_size=100, mh_sampling_iterations=100, eps=0, debug=True) ## using observations

    def test_given_data(self):
        print(colored('Metropolis-Hastings with data', 'blue'))
        space = RefinedSpace([(0, 1), (0, 1)], ["p", "q"], ["Real", "Real"], [[[0, 0.5], [0, 0.5]]], [], true_point=[0.82, 0.92])
        f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
        initialise_sampling(space, data=[0.2, 0.5, 0.3], functions=f, sample_size=100, mh_sampling_iterations=100, eps=0, debug=True)


if __name__ == '__main__':
    unittest.main()
