import re


def is_this_z3_function(function: str):
    if "If" in function or "And" in function or "Or" in function or "Not" in function:
        return True
    else:
        return False


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
