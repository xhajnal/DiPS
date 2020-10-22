import unittest
from src.load import *
cwd = os.getcwd()
test = cwd
model_dir = os.path.join(cwd, "models")
data_dir = os.path.join(cwd, "data")
prism_results = os.path.join(cwd, "results/prism_results")
storm_results = os.path.join(cwd, "results/storm_results")


class MyTestCase(unittest.TestCase):
    def test_parse_params_from_model(self):
        print(colored('Parse parameters from a given model', 'blue'))
        self.assertEqual(parse_params_from_model(os.path.join(model_dir, "asynchronous_2.pm")), ([], ["p", "q"]))

    def test_find_param(self):
        print(colored('Parse parameters from a string', 'blue'))
        start_time = time()
        for i in range(1):
            self.assertEqual(find_param("56*4+4**6 +   0.1"), set())
            self.assertEqual(find_param("x+0.1"), {'x'})
            self.assertEqual(find_param("p**2-2*p+1"), {'p'})
            self.assertEqual(find_param("p ** 2 - 2 * p + 1"), {'p'})
            self.assertEqual(find_param("(-2)*q1*p**2+2*q1*p+2*p"), {'p', 'q1'})
            self.assertEqual(find_param('-p*(2*p*If(Or(low<1,1<=high),qmin,qmax)-p-2*If(Or(low<1,1<=high),qmin,qmax))'), {'qmin', 'p', 'low', 'qmax', 'high'})
            self.assertEqual(find_param('10*p*(p - 1)**9*( If ( Or( low < 1 , 1 <= high), qmin, qmax) - 1)**9'), {'p', 'low', "high", "qmin", "qmax"})
            for polynome in ['p**2-2*p+1>=0.255810035160231', 'p**2-2*p+1<=0.453332821982626', '2*q*p**2-2*p**2-2*q*p+2*p>=0.339105082511199', '2*q*p**2-2*p**2-2*q*p+2*p<=0.543752060345944', '(-2)*q*p**2+p**2+2*q*>=0.120019530949760', '(-2)*q*p**2+p**2+2*q*<=0.287980469050240']:
                print(find_param(polynome))
        print(time()-start_time)

    def test_find_param_old(self):
        print(colored('Parse parameters from a string', 'blue'))
        start_time = time()
        for i in range(1):
            self.assertEqual(find_param_old("56*4+4**6 +   0.1"), set())
            self.assertEqual(find_param_old("x+0.1"), {'x'})
            self.assertEqual(find_param_old("p**2-2*p+1"), {'p'})
            self.assertEqual(find_param_old("p ** 2 - 2 * p + 1"), {'p'})
            self.assertEqual(find_param_old("(-2)*q1*p**2+2*q1*p+2*p"), {'p', 'q1'})
            self.assertEqual(find_param_old('-p*(2*p*If(Or(low<1,1<=high),qmin,qmax)-p-2*If(Or(low<1,1<=high),qmin,qmax))'), {'qmin', 'p', 'low', 'qmax', 'high'})
            self.assertEqual(find_param_old('10*p*(p - 1)**9*( If ( Or( low < 1 , 1 <= high), qmin, qmax) - 1)**9'), {'p', 'low', "high", "qmin", "qmax"})
            for polynome in ['p**2-2*p+1>=0.255810035160231', 'p**2-2*p+1<=0.453332821982626', '2*q*p**2-2*p**2-2*q*p+2*p>=0.339105082511199', '2*q*p**2-2*p**2-2*q*p+2*p<=0.543752060345944', '(-2)*q*p**2+p**2+2*q*>=0.120019530949760', '(-2)*q*p**2+p**2+2*q*<=0.287980469050240']:
                print(find_param_old(polynome))
        print(time() - start_time)

    def test_get_f(self):
        print(colored('Parse nonrewards from a given file', 'blue'))

        self.assertEqual(get_f(os.path.join(prism_results, "asynchronous_2.txt"), "prism", False),
                         ['p**2-2*p+1', '2*q*p**2-2*p**2-2*q*p+2*p', '(-2)*q*p**2+p**2+2*q*p'])
        self.assertEqual(get_f(os.path.join(prism_results, "asynchronous_2.txt"), "prism", True),
                         ['(p - 1)**2', '2*p*(p - 1)*(q - 1)', '-p*(2*p*q - p - 2*q)'])

        self.assertEqual(get_f(os.path.join(storm_results, "asynchronous_3_moments.txt"), "storm", False), [])
        self.assertEqual(get_f(os.path.join(storm_results, "asynchronous_3_moments.txt"), "storm", True), [])

    def test_get_rewards(self):
        print(colored('Parse rewards from a given file', 'blue'))

        self.assertEqual(get_rewards(os.path.join(prism_results, "asynchronous_2.txt"), "prism", False), [])
        self.assertEqual(get_rewards(os.path.join(prism_results, "asynchronous_2.txt"), "prism", True), [])

        self.assertEqual(get_rewards(os.path.join(storm_results, "asynchronous_3_moments.txt"), "storm", False),
                         ['(3*((p)*(p**2*q+2*q+(-3)*p*q+1)))/(1)', '(3*((p)*(2*p**2*q**2+6*q+(-4)*p*q**2+p**2*q+2*q**2+(-7)*p*q+2*p+1)))/(1)'])
        self.assertEqual(get_rewards(os.path.join(storm_results, "asynchronous_3_moments.txt"), "storm", True),
                         ['3*p*(p**2*q - 3*p*q + 2*q + 1)', '3*p*(2*p**2*q**2 + p**2*q - 4*p*q**2 - 7*p*q + 2*p + 2*q**2 + 6*q + 1)'])

    def test_load_functions(self):
        print(colored('Parse functions from a given file', 'blue'))

        self.assertEqual(load_functions(os.path.join(prism_results, "asynchronous_2.txt"), "prism", False),
                         (['p**2-2*p+1', '2*q*p**2-2*p**2-2*q*p+2*p', '(-2)*q*p**2+p**2+2*q*p'], []))
        self.assertEqual(load_functions(os.path.join(prism_results, "asynchronous_2.txt"), "prism", True),
                         (['(p - 1)**2', '2*p*(p - 1)*(q - 1)', '-p*(2*p*q - p - 2*q)'], []))

        self.assertEqual(load_functions(os.path.join(storm_results, "asynchronous_3_moments.txt"), "storm", False),
                         ([], ['(3*((p)*(p**2*q+2*q+(-3)*p*q+1)))/(1)', '(3*((p)*(2*p**2*q**2+6*q+(-4)*p*q**2+p**2*q+2*q**2+(-7)*p*q+2*p+1)))/(1)']))
        self.assertEqual(load_functions(os.path.join(storm_results, "asynchronous_3_moments.txt"), "storm", True),
                         ([], ['3*p*(p**2*q - 3*p*q + 2*q + 1)', '3*p*(2*p**2*q**2 + p**2*q - 4*p*q**2 - 7*p*q + 2*p + 2*q**2 + 6*q + 1)']))

    def test_load_all_functions(self):
        print(colored('Parse functions from multiple files', 'blue'))
        agents_quantities = [3, 5]
        self.assertEqual(load_all_functions(os.path.join(storm_results, "asynchronous_*.txt"), "storm", False, agents_quantities), (get_all_f(os.path.join(storm_results, "asynchronous_*.txt"), "storm", False, agents_quantities), get_all_rewards(os.path.join(storm_results, "asynchronous_*.txt"), "storm", False, agents_quantities)))

    def test_load_data(self):
        print(colored('Parsing single data file', 'blue'))
        self.assertEqual(load_data(os.path.join(data_dir, "data.csv")), [0.04, 0.02, 0.94])  ## GOES WITH WARING: Warning while parsing line number 1. Expected number, got <class 'str'>. Skipping this line: "n=2, p_v=0.81, q_v=0.92"
        self.assertEqual(pickle_load(os.path.join(data_dir, "data.p")), [0.8166666667, 0.1166666667, 0.06666666667])

    def test_load_all_data(self):
        print(colored('Parsing multiple data files', 'blue'))
        ## TODO
        pass

    def test_to_variance(self):
        print(colored('Computing variance from rewards', 'blue'))
        ## TODO
        pass

    def test_mpmath_intervals(self):
        print(colored('mpi math sanity check', 'blue'))
        ## Check more here https://docs.sympy.org/0.6.7/modules/mpmath/basics.html
        ## Sanity check test
        from mpmath import mpi  ## Real intervals
        my_interval = mpi(0, 5)
        self.assertEqual(my_interval.a, 0)
        self.assertEqual(my_interval.b, 5)
        self.assertEqual(my_interval.mid, (5+0)/2)
        self.assertEqual(my_interval.delta, abs(0-5))


if __name__ == '__main__':
    unittest.main()
