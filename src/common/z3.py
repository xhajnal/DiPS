import re
from z3 import *


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


def is_this_z3_function(function: str):
    """ Checks whether this is z3 expression

    Args:
        function (string): expression to be checked
    """
    return "If" in function or "And" in function or "Or" in function or "Not" in function


## TODO not fully checked what the z3 parser cannot eat
def is_this_python_function(function: str):
    """ Checks whether this is python expression

    Args:
        function (string): expression to be checked
    """
    return "if" in function or "and" in function or "or" in function or "not" in function


def is_this_general_function(function: str):
    """ Checks whether this is general (not z3 nor python) expression

    Args:
        function (string): expression to be checked
    """
    return not(is_this_python_function(function) or is_this_python_function(function))


def translate_z3_function(function: str):
    """ Translates z3 expression into python expression

    Args:
        function (string): expression to be translated
    """
    if "If" in function:
        function = re.sub(r"If\((.*),(.*),(.*)\)", r"(\g<2> if \g<1> else \g<3>)", function)
    if "And" in function:
        function = re.sub(r"And\((.*),(.*)\)", r"(\g<1>) and (\g<2>)", function)
    if "Or" in function:
        function = re.sub(r"Or\((.*),(.*)\)", r"(\g<1>) or (\g<2>)", function)
    if "Not" in function:
        function = re.sub(r"Not\((.*)\)", r" not (\g<1>)", function)
    return function


## TODO
def translate_to_z3_function(function: str):
    """ Translates python expression into z3 expression

    Args:
        function (string): expression to be translated
    """
    pass
