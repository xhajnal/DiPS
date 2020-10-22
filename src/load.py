import glob
import os
import copy
import re
from pathlib import Path
from time import time

from termcolor import colored
from sympy import factor

## Importing my code
from common.config import load_config

spam = load_config()
data_path = spam["data"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
del spam

###############################
###   RATIONAL FUNCTIONS    ###
###############################


def load_functions(file_path, tool="unknown", factorize=True, rewards_only=False, f_only=False):
    """ Loads parameter synthesis from file into two maps - f list of rational functions for each property, and rewards list of rational functions for each reward

    Args:
        file_path (string):
        factorize (bool): if true it will factorise polynomial results
        rewards_only (bool): if true it parse only rewards
        f_only (bool): if true it will parse only standard properties
        tool (string): a tool of which is the output from (PRISM/STORM)

    Returns:
        (f,reward), where
        f (list of strings): rational functions for each property
        rewards (list of strings): rational functions for each reward
    """

    ## Time statistics
    time_to_factorise = 0

    ## Setting the current directory
    if not Path(file_path).is_absolute():
        if tool.lower().startswith("p"):
            file_path = os.path.join(prism_results, file_path)
        elif tool.lower().startswith("s"):
            file_path = os.path.join(storm_results, file_path)
        else:
            print("Selected tool unsupported.")
            return False, False
    f = []
    rewards = []

    with open(file_path, "r") as file:
        i = -1
        here = ""
        ## PARSING PRISM/STORM OUTPUT
        ## Getting the tool
        for line in file:
            if tool == "unknown":
                # print(line)
                if line.lower().startswith("prism"):
                    tool = "prism"
                elif line.lower().startswith("storm"):
                    tool = "storm"
            else:
                break
        if tool == "unknown":
            print("Tool not recognised")
            return False, False

        ## Parsing Rational functions
        line_index = 0
        for line in file:
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
                if line[-1] == "\n":
                    line = line[:-1]
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
                line = line.replace("*|", "|")
                line = line.replace("|*", "|")
                ## Redoing PRISM non-standard operator orders
                if tool.lower().startswith("p"):
                    if "|" in line:
                        line = f"({line})"
                        line = line.replace("|", ")/(")
                else:
                    line = line.replace("|", "/")
                line = line.replace("(*", "(")
                line = line.replace("+*", "+")
                line = line.replace("-*", "-")
                if line.startswith('*'):
                    line = line[1:]
                if here == "r" and not f_only:
                    # print(f"formula: {i+1}", line)
                    if factorize:
                        start_time = time()
                        try:
                            rewards.append(str(factor(line)))
                        except TypeError:
                            print("Error while factorising rewards, used not factorised instead")
                            rewards.append(line)
                            # os.chdir(cwd)
                        finally:
                            time_to_factorise = time_to_factorise + (time() - start_time)
                    else:
                        rewards.append(line)
                elif not here == "r" and not rewards_only:
                    # print(f"formula: {i+1}", line[:-1])
                    if factorize:
                        start_time = time()
                        try:
                            f.append(str(factor(line)))
                        except TypeError:
                            print(f"Error while factorising polynomial f[{i + 1}], used not factorised instead")
                            f.append(line)
                        finally:
                            time_to_factorise = time_to_factorise + (time() - start_time)
                    else:
                        f.append(line)
            line_index = line_index + 1

    if factorize:
        print(colored(f"Factorisation took {time_to_factorise} seconds", "yellow"))
    return f, rewards


def get_f(path, tool, factorize):
    """ Loads all nonreward results of parameter synthesis from *path* folder """
    return load_functions(path, tool, factorize, rewards_only=False, f_only=True)[0]


def get_rewards(path, tool, factorize):
    """ Loads all reward results of parameter synthesis from *path* folder """
    return load_functions(path, tool, factorize, rewards_only=True, f_only=False)[1]


## TODO rewrite this using load_functions
def load_all_functions(path, tool, factorize=True, agents_quantities=False, rewards_only=False, f_only=False):
    """ Loads all results of parameter synthesis from *path* folder into two maps - f list of rational functions for each property, and rewards list of rational functions for each reward
    
    Args:
        path (string): file name regex
        factorize (bool): if true it will factorise polynomial results
        rewards_only (bool): if true it parse only rewards
        f_only (bool): if true it will parse only standard properties
        agents_quantities (list of numbers or False): of population sizes to be used, if False, the whole path used
        tool (string): a tool of which is the output from (PRISM/STORM)

    Returns:
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
    for functions_file in glob.glob(str(path)):
        try:
            population_size = int(re.findall(r'\d+', functions_file)[0])
        except IndexError:
            population_size = 0
        ## Parsing only selected agents quantities
        if agents_quantities:
            if population_size not in agents_quantities:
                continue
            else:
                no_files = False
                print("parsing ", os.path.join(os.getcwd(), functions_file))
        # print(os.getcwd(), file)
        with open(functions_file, "r") as file:
            i = -1
            here = ""
            f[population_size] = []
            rewards[population_size] = []
            ## PARSING PRISM/STORM OUTPUT
            line_index = 0
            if tool == "unknown":
                # print(line)
                if line.lower().startswith("prism"):
                    tool = "prism"
                elif line.lower().startswith("storm"):
                    tool = "storm"
                else:
                    print("Tool not recognised!!")
            for line in file:
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
                    line = line.replace("*|", "|")
                    line = line.replace("|*", "|")
                    line = line.replace("|", "/")
                    line = line.replace("(*", "(")
                    line = line.replace("+*", "+")
                    line = line.replace("-*", "-")
                    if line.startswith('*'):
                        line = line[1:]
                    if line[-1] == "\n":
                        line = line[:-1]
                    if here == "r" and not f_only:
                        # print(f"pop: {N}, formula: {i+1}", line)
                        if factorize:
                            try:
                                rewards[population_size].append(str(factor(line)))
                            except TypeError:
                                print("Error while factorising rewards, used not factorised instead")
                                rewards[population_size].append(line)
                                # os.chdir(cwd)
                        else:
                            rewards[population_size].append(line)
                    elif not here == "r" and not rewards_only:
                        # print(f"pop: {N}, formula: {i+1}", line[:-1])
                        if factorize:
                            try:
                                f[population_size].append(str(factor(line)))
                            except TypeError:
                                print(f"Error while factorising polynomial f[{population_size}][{i + 1}], used not factorised instead")
                                f[population_size].append(line)
                                # os.chdir(cwd)
                        else:
                            f[population_size].append(line)
                line_index = line_index + 1
    os.chdir(default_directory)
    if no_files and agents_quantities:
        print("No files match the pattern " + os.path.join(new_dir, path) + " and restriction " + str(agents_quantities))
    return (f, rewards)


def get_all_f(path, tool, factorize, agents_quantities=False):
    """ Loads all nonreward results of parameter synthesis from *path* folder

    Args:
        path (string): file name regex
        tool (string): a tool of which is the output from (PRISM/STORM)
        factorize (bool): if true it will factorise polynomial results
        agents_quantities (list of numbers or False): of population sizes to be used, if False, the whole path used
    """
    return load_all_functions(path, tool, factorize, agents_quantities=agents_quantities, rewards_only=False, f_only=True)[0]


def get_all_rewards(path, tool, factorize, agents_quantities=False):
    """ Loads all reward results of parameter synthesis from *path* folder

    Args:
        path (string): file name regex
        tool (string): a tool of which is the output from (PRISM/STORM)
        factorize (bool): if true it will factorise polynomial results
        agents_quantities (list of numbers or False): of population sizes to be used, if False, the whole path used
    """
    return load_all_functions(path, tool, factorize, agents_quantities=agents_quantities, rewards_only=True, f_only=False)[1]


def save_functions(dic, name):
    """ Exports the dic in a compact but readable manner

    Args:
        dic (dict): to be exported
        name (string): name of the dictionary (used for the file name)
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

    Args:
        dic (dict): for which the variance is computed
    """

    for key in dic.keys():
        dic[key][1] = str(factor(f"{dic[key][1]} - ({dic[key][0]} * {dic[key][0]})"))
    return dic


###########################
###       DATA HERE     ###
###########################
## TODO rewrite this to be able to load pickled data
def load_data(path, silent: bool = False, debug: bool = False):
    """ Loads experimental data, returns as list "data"

    Args:
        path (string): file name
        silent (bool): if silent printed output is set to minimum
        debug (bool): if debug extensive print will be used

    Returns:
    data: dictionary N -> list of probabilities for respective property
    """
    cwd = os.getcwd()
    if not Path(path).is_absolute():
        os.chdir(data_path)

    data = []
    with open(path, "r") as file:
        for line in file:
            line = line[:-1]
            if debug:
                print("Unparsed line: ", line)
            if "," not in line:
                print(colored(f"Comma not in line {line}, skipping this line", "red"))
                continue
            correct_line = True
            data = line.split(",")
            if debug:
                print("Parsed line: ", data)

            for value in range(len(data)):
                try:
                    assert isinstance(data, list)
                    data[value] = float(data[value])
                except ValueError:
                    print(colored(f"Warning while parsing line number {value + 1}. Expected number, got {type(data[value])}. Skipping this line: {line}", "red"))
                    correct_line = False
                    break
            if correct_line:
                break
    os.chdir(cwd)
    if data:
        return data
    else:
        return None


def load_all_data(path):
    """ loads all experimental data for respective property, returns as dictionary "data"
    
    Args:
        path (string): file name regex
    
    Returns:
        D: dictionary N -> list of probabilities for respective property
    """
    cwd = os.getcwd()
    if not Path(path).is_absolute():
        os.chdir(data_path)

    data = {}
    if not glob.glob(str(path)):
        raise OSError("No valid files in the given directory " + os.path.join(os.getcwd(), path))

    for file_name in glob.glob(str(path)):
        print(os.path.join(os.getcwd(), file_name))
        with open(file_name, "r") as file:
            population = 0
            for line in file:
                # print("line: ",line)
                if re.search("population", line) is not None:
                    population = int(line.split(",")[0].split("=")[1])
                    # print("population, ",population)
                    data[population] = []
                    continue
                data[population] = line[:-1].split(",")
                # print(D[N])
                for value in range(len(data[population])):
                    # print(D[N][value])
                    try:
                        data[population][value] = float(data[population][value])
                    except:
                        print("error while parsing population =", population, " i =", value, " of value =",
                              data[population][value])
                        data[population][value] = 0
                    # print(type(D[population][value]))
                # D[population].append(1-sum(D[population]))
                break
                # print(D[population])
    os.chdir(cwd)
    if data:
        return data
    else:
        print("Error, No data loaded, please check path")


#######################
###   PARAMETERS    ###
#######################


def parse_params_from_model(file, silent: bool = False):
    """ Parses the parameters from a given prism file

    Args:
        file: ((path/string)) a prism model file to be parsed
        silent (bool): if silent command line output is set to minimum
    """
    consts = []
    params = []
    # print("file", file)
    with open(file, 'r') as input_file:
        for line in input_file:
            if line.startswith("const"):
                print(line[-1])
                line = line.split(";")[0]
                print(line)
                if "=" in line:
                    print()
                    continue
                if "bool" in line or "int" in line:  ## parsing constants
                    line = line.split(" ")[-1]
                    print(f"const {line}")
                    consts.append(line)
                else:
                    line = line.split(" ")[-1]
                    print(f"param {line}")
                    params.append(line)
                print()
    if not silent:
        print("params", params)
        print("consts", consts)
    return consts, params


def find_param(my_string, debug: bool = False):
    """ Finds parameters of a string (also deals with Z3 expressions)

    Args:
        my_string : input string
        debug (bool): if debug extensive output is provided

    Returns:
         set of strings - parameters
    """
    try:
        return my_string.free_symbols
    except AttributeError:
        pass

    my_string = copy.copy(my_string)
    if debug:
        print("my_default_string ", my_string)
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
            if debug:
                print("my_string ", my_string)
                print("parameter ", parameter)
            my_string = my_string.replace(parameter, "2")
            if debug:
                print("my_string ", my_string)
        except TypeError as my_error:
            if debug:
                print(str(my_error))
            if str(my_error) == "'int' object is not callable":
                if debug:
                    print("I am catching the bloody bastard")
                my_string = my_string.replace(",", "-")
                my_string = my_string.replace("(", "+").replace(")", "")
                my_string = my_string.replace("<=", "+").replace(">=", "+")
                my_string = my_string.replace("<", "+").replace(">", "+")
                my_string = my_string.replace("++", "+")
                if debug:
                    print("my_string ", my_string)
            else:
                if debug:
                    print(f"Dunno why this error '{my_error}' happened, sorry ")
                hippie = False
        except SyntaxError as my_error:
            if str(my_error).startswith("invalid syntax"):
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


def find_param_old(expression, debug: bool = False):
    """ Finds parameters of a polynomials (also deals with Z3 expressions)

    Args:
        expression : polynomial as string
        debug (bool): if True extensive print will be used

    Returns:
        set of strings - parameters
    """

    ## Get the e-/e+ notation away
    try:
        return expression.free_symbols
    except AttributeError:
        pass

    parameters = re.sub('[0-9]e[+|-][0-9]', '0', expression)

    parameters = parameters.replace('(', '').replace(')', '').replace('**', '*').replace(' ', '')
    ## replace python expressions
    parameters = re.sub(r"([^a-z, A-Z])(if|else|elif|not|or|and|min|max)", r"\1", parameters)
    # parameters = parameters.replace("not", " ").replace("or", " ").replace("and", " ")
    # parameters = parameters.replace("min", " ").replace("max", " ")

    ## Replace z3 expression
    parameters = re.sub(r"(Not|Or|And|Implies|If|,|<|>|=)", " ", parameters)

    parameters = re.split(r'\+|\*|\-|/| ', parameters)
    parameters = [i for i in parameters if not i.replace('.', '', 1).isdigit()]
    parameters = set(parameters)
    parameters.discard("")
    # print("hello",set(parameters))
    return set(parameters)


def find_param_older(expression, debug: bool = False):
    """ Finds parameters of a polynomials

    Args:
        expression : polynomial as string
        debug (bool): if True extensive print will be used
    
    Returns:
         set of strings - parameters
    """
    parameters = expression.replace('(', '').replace(')', '').replace('**', '*').replace(' ', '')
    parameters = re.split(r'\+|\*|\-|/', parameters)
    parameters = [i for i in parameters if not i.replace('.', '', 1).isdigit()]
    parameters = set(parameters)
    parameters.discard("")
    # print("hello",set(parameters))
    return set(parameters)


def parse_constraints(file, silent=True, debug=False):
    """ Loads constraints from text file, returns as list of strings "constraints"

    Args:
        file (string/Path): file path to parse constraints
        silent (bool): if silent printed output is set to minimum
        debug (bool): if debug extensive print will be used

    Returns:
    constraints: (list of string) constraints
    """
    with open(file) as f:
        constraints = []
        content = f.readlines()
        for line in content:
            line = re.sub(r', *\n', '', line)
            constraints.append(line)
    return constraints


def parse_functions(file, silent=True, debug=False):
    """ Loads functions from text file, returns as list of strings "functions"

    Args:
        file (string/Path): file path to parse functions
        silent (bool): if silent printed output is set to minimum
        debug (bool): if debug extensive print will be used

    Returns:
    (list of string) functions
    """
    return parse_constraints(file, silent, debug)

