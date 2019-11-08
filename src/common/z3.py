import re
from z3 import *


def z3_eval(function: str):
    value = eval(function)
    try:
        value = str(z3.simplify(value).as_decimal(30))
        if value[-1] == "?":
            value = value[:-1]
    except Z3Exception:
        pass
    return value


def is_this_z3_function(function: str):
    return "If" in function or "And" in function or "Or" in function or "Not" in function


## TODO not fully checked what the z3 parser cannot eat
def is_this_python_function(function: str):
    return "if" in function or "and" in function or "or" in function or "not" in function


def is_this_general_function(function: str):
    return is_this_python_function(function) or is_this_python_function(function)


def translate_z3_function(function: str):
    if "If" in function:
        function = re.sub(r"If\((.*),(.*),(.*)\)", r"[\g<3>,\g<2>](\g<1>)", function)
    if "And" in function:
        function = re.sub(r"And\((.*),(.*)\)", r"(\g<1>) and (\g<2>)", function)
    if "Or" in function:
        function = re.sub(r"Or\((.*),(.*)\)", r"(\g<1>) or (\g<2>)", function)
    if "Not" in function:
        function = re.sub(r"Not\((.*)\)", r" not (\g<1>)", function)
    return function


## TODO
def translate_to_z3_function(function: str):
    if "If" in function:
        function = re.sub(r"If\((.*),(.*),(.*)\)", r"[\g<3>,\g<2>](\g<1>)", function)
    if "And" in function:
        function = re.sub(r"And\((.*),(.*)\)", r"(\g<1>) and (\g<2>)", function)
    if "Or" in function:
        function = re.sub(r"Or\((.*),(.*)\)", r"(\g<1>) or (\g<2>)", function)
    if "Not" in function:
        function = re.sub(r"Not\((.*)\)", r" not (\g<1>)", function)
    return function
