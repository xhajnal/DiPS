import copy
import re


#######################
###   PARAMETERS    ###
#######################


def parse_params_from_model(file, silent: bool = False, debug=False):
    """ Parses the constants and parameters from a given prism file

    Args:
        file: ((path/string)) a prism model file to be parsed
        silent (bool): if silent command line output is set to minimum
        debug (bool): if debug extensive print will be used
    Returns:
        tuple (list of constants, list of parameters)
    """
    consts = []
    params = []
    # print("file", file)
    with open(file, 'r') as input_file:
        for line in input_file:
            if re.compile(r"^\s*const").search(line) is not None:
                if debug:
                    print(line[-1])
                line = line.split(";")[0]
                if debug:
                    print(line)
                if "=" in line:
                    if debug:
                        print()
                    continue
                if "bool" in line or "int" in line:  ## parsing constants
                    line = line.split(" ")[-1]
                    if debug:
                        print(f"const {line}")
                    consts.append(line)
                else:
                    line = line.split(" ")[-1]
                    if debug:
                        print(f"param {line}")
                    params.append(line)
                if debug:
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
        symbols = list(expression.free_symbols)
        for index, item in enumerate(symbols):
            symbols[index] = str(item)
        return symbols

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

    parameters = re.split(r'[+*\-/ ]', parameters)
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
    parameters = re.split(r'[+*\-/]', parameters)
    parameters = [i for i in parameters if not i.replace('.', '', 1).isdigit()]
    parameters = set(parameters)
    parameters.discard("")
    # print("hello",set(parameters))
    return set(parameters)