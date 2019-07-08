import configparser
import glob
import math
import os
import pickle
import copy
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
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)


config.read(os.path.join(workspace, "../config.ini"))

prism_results = config.get("paths", "prism_results")
if not os.path.exists(prism_results):
    raise OSError("Directory does not exist: " + str(prism_results))

storm_results = config.get("paths", "storm_results")
if not os.path.exists(prism_results):
    raise OSError("Directory does not exist: " + str(storm_results))

data_path = config.get("paths", "data")
if not os.path.exists(data_path):
    raise OSError("Directory does not exist: " + str(data_path))

os.chdir(cwd)


###########################
### RATIONAL FUNCTIONS  ###
###########################


def load_all_functions(path, tool, factorize=True, agents_quantities=False, rewards_only=False, f_only=False):
    """ Loads all results of parameter synthesis from *path* folder into two maps - f list of rational functions for each property, and rewards list of rational functions for each reward
    
    Args
    ----------
    path: (string) - file name regex
    factorize: (Bool) if true it will factorise polynomial results
    rewards_only: (Bool) if true it compute only rewards
    f_only: if true it will compute only standard properties
    agents_quantities: (list) of population sizes to be used
    tool: (string) a tool of which is the output from (PRISM/STORM)

    Returns
    ----------
    (f,reward), where
    f: dictionary N -> list of rational functions for each property
    rewards: dictionary N -> list of rational functions for each reward
    """

    ## Setting the current directory
    default_directory = os.getcwd()
    if not Path(path).is_absolute():
        if tool.lower().startswith("p"):
            os.chdir(prism_results)
        elif tool.lower().startswith("s"):
            os.chdir(storm_results)
        else:
            print("Selected tool unsupported.")
            return ({}, {})

    f = {}
    rewards = {}
    # print(str(path))
    new_dir = os.getcwd()
    if not glob.glob(str(path)):
        if not Path(path).is_absolute():
            os.chdir(default_directory)
        print("No files match the pattern " + os.path.join(new_dir, path))
        return ({}, {})

    no_files = True
    ## Choosing files with the given pattern
    for file in glob.glob(str(path)):
        N = int(re.findall('\d+', file)[0])
        ## Parsing only selected agents quantities
        if agents_quantities:
            if N not in agents_quantities:
                continue
            else:
                no_files = False
                print("parsing ", os.path.join(os.getcwd(), file))
        # print(os.getcwd(), file)
        file = open(file, "r")
        i = -1
        here = ""
        f[N] = []
        rewards[N] = []
        ## PARSING PRISM/STORM OUTPUT
        line_index = 0
        for line in file:
            if line_index == 0:
                if tool is "unknown":
                    # print(line)
                    if line.lower().startswith("prism"):
                        tool = "prism"
                    elif line.lower().startswith("storm"):
                        tool = "storm"
                    else:
                        print("Tool not recognised!!")
            if line.startswith('Parametric model checking:') or line.startswith('Model checking property'):
                i = i + 1
                here = ""
                ## STORM check if rewards
                if "R[exp]" in line:
                    here = "r"
            ## PRISM check if rewards
            if line.startswith('Parametric model checking: R'):
                here = "r"
            if i >= 0 and line.startswith('Result'):
                ## PARSE THE EXPRESSION
                # print("line:", line)
                if tool.lower().startswith("p"):
                    line = line.split(":")[2]
                elif tool.lower().startswith("s"):
                    line = line.split(":")[1]
                ## CONVERT THE EXPRESSION TO PYTHON FORMAT
                line = line.replace("{", "")
                line = line.replace("}", "")
                ## PUTS "* " BEFORE EVERY WORD (VARIABLE)
                line = re.sub(r'([a-z|A-Z]+)', r'* \1', line)
                # line = line.replace("p", "* p")
                # line = line.replace("q", "* q")
                line = line.replace("**", "*")
                line = line.replace("* *", "*")
                line = line.replace("*  *", "*")
                line = line.replace("+ *", "+")
                line = line.replace("^", "**")
                line = line.replace(" ", "")
                line = line.replace("(*", "(")
                line = line.replace("+*", "+")
                line = line.replace("-*", "-")
                if line.startswith('*'):
                    line = line[1:]
                if here == "r" and not f_only:
                    # print(f"pop: {N}, formula: {i+1}", line[:-1])
                    if factorize:
                        try:
                            rewards[N].append(str(factor(line[:-1])))
                        except TypeError:
                            print("Error while factorising rewards, used not factorised instead")
                            rewards[N].append(line[:-1])
                            # os.chdir(cwd)
                    else:
                        rewards[N].append(line[:-1])
                elif not here == "r" and not rewards_only:
                    # print(f"pop: {N}, formula: {i+1}", line[:-1])
                    if factorize:
                        try:
                            f[N].append(str(factor(line[:-1])))
                        except TypeError:
                            print(f"Error while factorising polynomial f[{N}][{i+1}], used not factorised instead")
                            f[N].append(line[:-1])
                            # os.chdir(cwd)
                    else:
                        f[N].append(line[:-1])
            line_index = line_index + 1
        file.close()
    os.chdir(default_directory)
    if no_files and agents_quantities:
        print("No files match the pattern " + os.path.join(new_dir, path) + "and restriction " + agents_quantities)
    return (f, rewards)


def get_f(path, tool, factorize, agents_quantities=False):
    """ Loads all nonreward results of parameter synthesis from *path* folder"""
    return load_all_functions(path, tool, factorize, agents_quantities=agents_quantities, rewards_only=False, f_only=True)[0]


def get_rewards(path, tool, factorize, agents_quantities=False):
    """ Loads all reward results of parameter synthesis from *path* folder"""
    return load_all_functions(path, tool, factorize, agents_quantities=agents_quantities, rewards_only=True, f_only=False)[1]


def save_functions(dic, name):
    """ Exports the dic in a compact but readable manner

    Args
    ----------
    dic: (map) to be exported
    name: (string) name of the dictionary (used for the file name)
    """

    for key in dic.keys():
        ## If value not empty list
        if dic[key]:
            with open(f"export_{name}_{key}.txt", "w") as file:
                i = 0
                for pol in dic[key]:
                    file.write(f"f[{i}]='{pol}' \n")
                    i = i + 1


def to_variance(dic):
    """ Computes variance specifically for the dictionary in a form dic[key][0] = EX, dic[key][1] = E(X^2)
    Args
    ----------
    dic: (map) for which the variance is computed
    """

    for key in dic.keys():
        dic[key][1] = str(factor(f"{dic[key][1]} - ({dic[key][0]} * {dic[key][0]})"))
    return dic


###########################
###       DATA HERE     ###
###########################


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
        file.close()
    os.chdir(cwd)
    if D:
        return D
    else:
        print("Error, No data loaded, please check path")


def load_pickled_data(file):
    """ returns pickled data
    
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


def find_param(my_string):
    """ Finds parameters of a string (also deals with Z3 expressions)

    Args
    ----------
    my_string : input string

    Returns set of strings - parameters
    """
    my_string = copy.copy(my_string)
    parameters = set()
    hippie = True
    while hippie:
        try:
            eval(str(my_string))
            hippie = False
        except NameError as my_error:
            parameter = str(str(my_error).split("'")[1])
            parameters.add(parameter)
            locals()[parameter] = 0
            # print("my_string ", my_string)
            # print("parameter ", parameter)
            my_string = my_string.replace(parameter, "2")
            # print("my_string ", my_string)
        except TypeError as my_error:
            # print(str(my_error))
            if str(my_error) == "'int' object is not callable":
                # print("I am catching the bloody bastard")
                my_string = my_string.replace(",", "-")
                my_string = my_string.replace("(", "+").replace(")", "")
                my_string = my_string.replace("<=", "+").replace(">=", "+")
                my_string = my_string.replace("<", "+").replace(">", "+")
                my_string = my_string.replace("++", "+")
                # print("my_string ", my_string)
            else:
                # print(f"Dunno why this error '{my_error}' happened, sorry ")
                hippie = False
        except SyntaxError as my_error:
            if str(my_error) == "invalid syntax":
                my_string = my_string.replace("*>", ">")
                my_string = my_string.replace("*<", "<")
                my_string = my_string.replace("*=", "=")

                my_string = my_string.replace("+>", ">")
                my_string = my_string.replace("+<", "<")
                my_string = my_string.replace("+=", "=")

                my_string = my_string.replace("->", ">")
                my_string = my_string.replace("-<", "<")
                my_string = my_string.replace("-=", "=")
            else:
                print(f" I was not able to fix this SyntaxError buddy,'{my_error}' happened. Sorry.")
                hippie = False

    parameters.discard("Not")
    parameters.discard("Or")
    parameters.discard("And")
    parameters.discard("If")
    parameters.discard("Implies")
    return parameters


def find_param_old(polynomial):
    """ Finds parameters of a polynomials (also deals with Z3 expressions)

    Args
    ----------
    polynomial : polynomial as string

    Returns set of strings - parameters
    """

    ## Get the e-/e+ notation away
    parameters = re.sub('[0-9]e[+|-][0-9]', '0', polynomial)
    parameters = parameters.replace('(', '').replace(')', '').replace('**', '*').replace(' ', '')
    parameters = parameters.replace("Not", " ").replace("Or", " ").replace("And", " ").replace("Implies", " ").replace("If", " ").replace(',', ' ')
    parameters = parameters.replace("<", " ").replace(">", " ").replace("=", " ")
    parameters = re.split('\+|\*|\-|/| ', parameters)
    parameters = [i for i in parameters if not i.replace('.', '', 1).isdigit()]
    parameters = set(parameters)
    parameters.discard("")
    # print("hello",set(parameters))
    return set(parameters)


def find_param_older(polynomial):
    """ Finds parameters of a polynomials

    Args
    ----------
    polynomial : polynomial as string
    
    Returns set of strings - parameters
    """
    parameters = polynomial.replace('(', '').replace(')', '').replace('**', '*').replace(' ', '')
    parameters = re.split('\+|\*|\-|/', parameters)
    parameters = [i for i in parameters if not i.replace('.','',1).isdigit()]
    parameters = set(parameters)
    parameters.discard("")
    # print("hello",set(parameters))
    return set(parameters)


class TestLoad(unittest.TestCase):
    def test_load_expressions(self):
        ## THIS WILL PASS ONLY AFTER CREATING THE THE STORM RESULTS
        agents_quantities = [2]
        f_storm = get_f("./asyn*[0-9]_moments.txt", "storm", True, agents_quantities)
        # print(f_storm)
        self.assertFalse(f_storm[2])
        rewards_storm = get_rewards("./asyn*[0-9]_moments.txt", "storm", True, agents_quantities)
        # print(rewards_storm)
        self.assertTrue(rewards_storm[2])

    def test_find_param(self):
        self.assertEqual(find_param("56*4+4**6 +   0.1"), set())
        self.assertEqual(find_param("x+0.1"), {'x'})
        self.assertEqual(find_param("(-2)*q1*p**2+2*q1*p+2*p"), {'p', 'q1'})
        self.assertEqual(find_param('-p*(2*p*If(Or(low<1,1<=high),qmin,qmax)-p-2*If(Or(low<1,1<=high),qmin,qmax))'), {'qmin', 'p', 'low', 'qmax', 'high'})
        self.assertEqual(find_param('10*p*(p - 1)**9*( If ( Or( low < 1 , 1 <= high), qmin, qmax) - 1)**9'), {'p', 'low', "high", "qmin", "qmax"})

    def test_intervals(self):
        my_interval = mpi(0, 5)
        self.assertEqual(my_interval.a, 0)
        self.assertEqual(my_interval.b, 5)
        self.assertEqual(my_interval.mid, (5+0)/2)
        self.assertEqual(my_interval.delta, abs(0-5))


if __name__ == "__main__":
    unittest.main()







