import configparser
import glob
import math
import os
import pickle
import re
from pathlib import Path
from collections.abc import Iterable
from mpmath import mpi

import unittest

import scipy.stats as st
from sympy import factor, Interval

config = configparser.ConfigParser()
print(os.getcwd())

workspace = os.path.dirname(__file__)
print("workspace", workspace)
config.read(os.path.join(workspace, "../config.ini"))

prism_results = config.get("paths", "prism_results")
if not os.path.exists(prism_results):
    raise OSError("Directory does not exist: " + str(prism_results))

data_path = config.get("paths", "data")
if not os.path.exists(data_path):
    raise OSError("Directory does not exist: " + str(data_path))


def load_all_prism(path, factorize=True, agents_quantities=[], rewards_only=False, f_only=False):
    """ Loads all results of parameter synthesis in *path* folder into two maps - f list of rational functions for each property, and rewards list of rational functions for each reward
    
    Args
    ----------
    path: (string) - file name regex
    factorize: (Bool) if true it will factorise polynomial results
    rewards_only: (Bool) if true it compute only rewards
    f_only: if true it will compute only standard properties
    
    Returns
    ----------
    (f,reward), where
    f: dictionary N -> list of rational functions for each property
    rewards: dictionary N -> list of rational functions for each reward
    agents_quantities: (list) of population sizes to be used
    """
    default_directory = os.getcwd()
    if not Path(path).is_absolute():
        os.chdir(prism_results)

    f = {}
    rewards = {}
    # print(str(path))
    if not glob.glob(str(path)):
        new_dir = os.getcwd()
        if not Path(path).is_absolute():
            os.chdir(default_directory)
        raise OSError("No valid files in the given directory " + os.path.join(new_dir, path))

    ## Choosing files with the given pattern
    for file in glob.glob(str(path)):
        N = int(re.findall('\d+', file)[0])
        ## Parsing only selected agents quantities
        if agents_quantities:
            if N not in agents_quantities:
                continue
            else:
                print(os.path.join(os.getcwd(), file))
        # print(os.getcwd(),file)
        file = open(file, "r")
        i = -1
        here = ""
        f[N] = []
        rewards[N] = []
        for line in file:
            if line.startswith('Parametric model checking:'):
                i = i + 1
            if line.startswith('Parametric model checking: R'):
                here = "r"
            if i >= 0 and line.startswith('Result'):
                line = line.split(":")[2]
                line = line.replace("{", "")
                line = line.replace("}", "")
                line = line.replace("p", "* p")
                line = line.replace("q", "* q")
                line = line.replace("**", "*")
                line = line.replace("* *", "*")
                line = line.replace("*  *", "*")
                line = line.replace("+ *", "+")
                line = line.replace("^", "**")
                line = line.replace(" ", "")
                if line.startswith('*'):
                    line = line[1:]
                if here == "r" and not f_only:
                    if factorize:
                        try:
                            rewards[N].append(str(factor(line[:-1])))
                        except TypeError:
                            print("Error while factorising rewards, used not factorised instead")
                            rewards[N].append(line[:-1])
                            # os.chdir(cwd)
                    else:
                        rewards[N] = line[:-1]
                elif not here == "r" and not rewards_only:
                    # print(f[N])
                    # print(line[:-1])
                    if factorize:
                        try:
                            f[N].append(str(factor(line[:-1])))
                        except TypeError:
                            print(
                                "Error while factorising polynome f[{}][{}], used not factorised instead".format(N, i))
                            f[N].append(line[:-1])
                            # os.chdir(cwd)
                    else:
                        f[N].append(line[:-1])
    os.chdir(default_directory)
    return (f, rewards)


def get_f(path, factorize, agents_quantities=[]):
    return load_all_prism(path, factorize, agents_quantities=agents_quantities, rewards_only=False, f_only=True)[0]


def get_rewards(path, factorize, agents_quantities=[]):
    return load_all_prism(path, factorize, agents_quantities=agents_quantities, rewards_only=True, f_only=False)[1]


def load_all_data(path):
    """ loads all experimental data for respective property, returns as dictionary D
    
    Args
    ----------
    path: (string) - file name regex
    
    Returns
    ----------
    D: dictionary N -> list of probabilities for respective property
    """
    cwd = os.getcwd()
    if not Path(path).is_absolute():
        os.chdir(data_path)

    D = {}
    if not glob.glob(str(path)):
        raise OSError("No valid files in the given directory " + os.path.join(os.getcwd(), path))

    for file in glob.glob(str(path)):
        print(os.path.join(os.getcwd(), file))
        file = open(file, "r")
        N = 0
        for line in file:
            # print("line: ",line)
            if re.search("n", line) is not None:
                N = int(line.split(",")[0].split("=")[1])
                # print("N, ",N)
                D[N] = []
                continue
            D[N] = line[:-1].split(",")
            # print(D[N])
            for value in range(len(D[N])):
                # print(D[N][value])
                try:
                    D[N][value] = float(D[N][value])
                except:
                    print("error while parsing N=", N, " i=", value, " of value=", D[N][value])
                    D[N][value] = 0
                # print(type(D[N][value]))
            # D[N].append(1-sum(D[N]))
            break
            # print(D[N])
    os.chdir(cwd)
    if D:
        return D
    else:
        print("Error, No data loaded, please check path")


def load_pickled_data(file):
    """ returns pickled experimental data for
    
    Args
    ----------
    file: (string) filename of the data to be loaded
    
    """
    return pickle.load(open(os.path.join(data_path, file + ".p"), "rb"))


def catch_data_error(data, minimum, maximum):
    """ Corrects all data value to be in range min,max
    
    Args
    ----------
    data: (dictionary) structure of data
    minimum: (float) minimal value in data to be set to
    maximum: (float) maximal value in data to be set to
    
    """
    for n in data.keys():
        for i in range(len(data[n])):
            if data[n][i] < minimum:
                data[n][i] = minimum
            if data[n][i] > maximum:
                data[n][i] = maximum


def create_intervals(alpha, n_samples, data):
    """ Returns intervals of data_point +- margin

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data: (list of floats), values to be margined
    """
    foo = []
    if not isinstance(data, Iterable):
        return [create_interval(alpha, n_samples, data)]
    for data_point in data:
        foo.append(create_interval(alpha, n_samples, data_point))
    return foo


def create_interval(alpha, n_samples, data_point):
    """ Returns interval of data_point +- margin

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data_point: (float), the value to be margined
    """
    change = margin(alpha, n_samples, data_point)
    return Interval(data_point - change, data_point + change)


def margin(alpha, n_samples, data_point):
    """ Estimates expected interval with respect to parameters
    TBA shortly describe this type of margin

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data_point: (float), the value to be margined
    """
    return st.norm.ppf(1 - (1 - alpha) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples) + 0.5 / n_samples


def margin_experimental(alpha, n_samples, data_point):
    """ Estimates expected interval with respect to parameters
    This margin was used to produce the visual outputs for hsb19 

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data_point: (float), the value to be margined
    """
    return st.norm.ppf(1 - (1 - alpha) / 2) * math.sqrt(
        data_point * (1 - data_point) / n_samples) + 0.5 / n_samples + 0.005


def find_param(polynome):
    """ Finds parameters of a polynomes

    Args
    ----------
    polynome : polynome as string
    
    Returns set of strings - parameters
    """
    parameters = polynome.replace('(', '').replace(')', '').replace('**', '*').replace(' ', '')
    parameters = re.split('\+|\*|\-|/', parameters)
    parameters = [i for i in parameters if not i.replace('.','',1).isdigit()]
    parameters = set(parameters)
    parameters.add("")
    parameters.remove("")
    # print("hello",set(parameters))
    return set(parameters)


class TestLoad(unittest.TestCase):
    def test_find_param(self):
        self.assertEqual(find_param("56*4+4**6 +   0.1"), set())
        self.assertEqual(find_param("x+0.1"),{'x'})

    def test_intervals(self):
        my_interval = mpi(0, 5)
        self.assertEqual(my_interval.a, 0)
        self.assertEqual(my_interval.b, 5)
        self.assertEqual(my_interval.mid, (5+0)/2)
        self.assertEqual(my_interval.delta, abs(0-5))

if __name__ == "__main__":
    unittest.main()







