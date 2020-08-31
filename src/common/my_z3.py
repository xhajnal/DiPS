import re
from z3 import *
from numpy import mean


def z3_eval(function: str):
    """ Returns value of z3 expression

    Args:
        function (string): expression to be evaluated
    """
    value = eval(function)
    try:
        value = str(z3.simplify(value).as_decimal(30))
        if value[-1] == "?":
            value = value[:-1]
    except Z3Exception:
        pass
    return value


## TODO not fully checked what the z3 parser cannot eat
def is_this_z3_function(function: str):
    """ Checks whether this is z3 expression

    Args:
        function (string): expression to be checked
    """
    # print("is_this_z3_function?", function)
    spam = str(function)
    return ("If" in spam) or ("And" in spam) or ("Or" in spam) or ("Not" in spam) or ("Pow" in spam)


def is_this_python_function(function: str):
    """ Checks whether this is python expression

    Args:
        function (string): expression to be checked
    """
    if is_this_z3_function(function):
        return False
    go_on = True
    while go_on:
        try:
            eval(function)
            return True
        except NameError as err:
            variable = str(err).split("'")[1]
            locals()[variable] = 0
            continue
        except Exception:
            return False
    return True
    # return "if" in function or "and" in function or "or" in function or "not" in function


def is_this_exponential_function(function: str):
    """ Checks whether the expression contains exponential function

    Args:
        function (string): expression to be checked
    """
    if re.findall("\*\* *[a-z|A-Z]", function):
        return True
    else:
        return False


def is_this_general_function(function: str):
    """ Checks whether this is general (not z3 nor python) expression

    Args:
        function (string): expression to be checked
    """
    return not(is_this_python_function(function) or is_this_z3_function(function))


## TODO check whether I got all
def translate_z3_function(function: str):
    """ Translates z3 expression into python expression

    Args:
        function (string): expression to be translated
    """
    ## ? is non greedy flag
    if "If" in function:
        function = re.sub(r"If\((.*?),(.*?),(.*?)\)", r"(\g<2> if \g<1> else \g<3>)", function)
    if "And" in function:
        function = re.sub(r"And\((.*?),(.*?)\)", r"(\g<1>) and (\g<2>)", function)
    if "Or" in function:
        function = re.sub(r"Or\((.*?),(.*?)\)", r"(\g<1>) or (\g<2>)", function)
    if "Not" in function:
        function = re.sub(r"Not\((.*?)\)", r" not (\g<1>)", function)
    return function


## TODO check whether I got all
def translate_to_z3_function(function: str):
    """ Translates python expression into z3 expression

    Args:
        function (string): expression to be translated
    """
    ## TODO wrong if the expressions contains expression with () inside
    ## necessity to use syntactic trees
    if "if" in function:
        function = re.sub(r"([^(]*?) +if +(.*?) +else +([^)]*)", r"If(\g<2>, \g<1>, \g<3>)", function)
    if "and" in function:
        function = re.sub(r"([^(]*?) +and +([^)]*?)", r"And(\g<1>, \g<2>)", function)
    if "or" in function:
        function = re.sub(r"([^(]*?) +or +([^)]*?)", r"Or(\g<1>, \g<2>)", function)
    if "not" in function:
        function = re.sub(r"not +(.*?) +", r" Not(\g<1>) ", function)
    return function


def parse_model_values(model: str, solver="z3"):
    """ Parses z3.solver.model() into list of values
        Ignores /0

    Example: [r_0 = 1/8, r_1 = 9/16, /0 = [(7/16, 7/8) -> 1/2, else -> 0]] -> [0.125, 0.5625]
    """
    if solver == "z3":
        ## Delete brackets
        model = model[1:-1]
        ## Delete /0 part
        if "/0" in model:
            model = model.split("/0")[0]
            ## Delete spaces
            model = re.sub(r', +', ",", model)
            ## Delete last comma
            if model[-1] == ",":
                model = model[:-1]
        model = model.split(",")
        model.sort()

        ## Parse the values
        values = []
        for value in model:
            values.append(float(eval(value.split("=")[1])))

        return values

    elif solver == "dreal":
        values = []
        model = model.split("\n")
        for line in model:
            ## parse the value
            line = line.split(":")[1]
            ## Delete brackets
            line = line.split("[")[1]
            line = line.split("]")[0]
            line = line.split(",")
            line = list(map(lambda x: float(x), line))
            values.append(float(mean(line)))
            print(values)
            print()
        return values
