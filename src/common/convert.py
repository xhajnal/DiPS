import re
from termcolor import colored
from sympy import Interval
import locale
locale.setlocale(locale.LC_ALL, '')


def niceprint(text):
    """ Converts string into printable text

    Args:
        text (int or string): input string
    """
    return locale.format_string("%d", text, grouping=True)


def parse_numbers(text: str):
    """ Converts string into a list of numbers

    Args:
        text (string): input string
    """
    newstr = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in text)
    return [float(i) for i in newstr.split()]


def to_sympy_intervals(intervals):
    """ converts list of lists or pairs into list of Intervals"""
    return list(map(lambda x: x if isinstance(x, Interval) else Interval(x[0], x[1]), intervals))


def ineq_to_constraints(functions: list, intervals: list, decoupled=True, silent: bool = True):
    """ Converts expressions and intervals into constraints
        list of expressions, list of intervals -> constraints

    Args:
        functions:  (list of strings) array of functions
        intervals (list of intervals): array of pairs, low and high bound
        decoupled (bool): if True returns 2 constraints for a single interval
        silent (bool): if silent printed output is set to minimum

    Example:
        ["x+3"],[[0,1]] ->  ["0<= x+3 <=1"]

    Returns:
        (list) of constraints
    """

    if len(functions) is not len(intervals):
        if not silent:
            print(colored(f"len of functions {len(functions)} and intervals {len(intervals)} does not correspond", "red"))
        return False

    ## Catching wrong interval errors
    try:
        spam = []
        for index in range(len(functions)):
            ## debug
            # print(colored(f"type of the function is {type(functions[index])}", "blue"))
            ## name intervals
            if isinstance(intervals[index], Interval):
                low = intervals[index].start
                high = intervals[index].end
            else:
                low = intervals[index][0]
                high = intervals[index][1]

            if decoupled or not isinstance(functions[index], str):
                if not isinstance(functions[index], str):
                    print("SYMPY")
                    spam.append(functions[index] >= low)
                    spam.append(functions[index] <= high)
                else:
                    spam.append(functions[index] + " >= " + str(low))
                    spam.append(functions[index] + " <= " + str(high))
            else:
                ## Old
                # spam.append(functions[index] + " >= " + str(low))
                # spam.append(functions[index] + " <= " + str(high))
                ## New
                spam.append(str(low) + " <= " + functions[index] + " <= " + str(high))
                ## Slightly slower
                # spam.append(f"{low} <= {functions[index]} <= {high}")
                ## Slow
                # spam.append(f"{functions[index]} in Interval({low}, {high})")

        return spam
    except TypeError as error:
        if "EmptySet" in str(error):
            raise Exception("ineq_to_constraints", "Some intervals are incorrect (lover bound > upper bound)")
        elif "FiniteSet" in str(error):
            raise Exception("ineq_to_constraints", "Some intervals are incorrect (empty)")
    except Exception as err:
        print("Unhandled exception", err)
        raise err


def constraints_to_ineq(constraints: list, silent: bool = True, debug: bool = False):
    """ Converts constraints to inequalities if possible
        constraints ->  list of expressions, list of intervals

    Args:
        constraints  (list of strings): properties to be converted
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used

    Example:
        ["x+3>=0","x+3<=1"] -> ["x+3"],[[0,1]]
    """
    if debug:
        silent = False
    if len(constraints) % 2:
        if not silent:
            print(colored("Number of properties is not even, some interval will be invalid", "red"))
        raise Exception(f"Number of properties is not even, some interval will be invalid")
    funcs = []
    intervals = []
    is_odd = False
    index = 0

    for prop in constraints:
        spam = "None"
        if debug:
            print(f"Constraint {index + 1} before splitting", prop)
        try:
            spam = prop.replace("<=", "<").replace(">=", "<").replace("=>", "<").replace("=<", "<").replace(">", "<")
            spam = spam.split("<")
        except AttributeError:
            print()
        if debug:
            print(f"Constraint {index+1} after splitting", spam)
        if len(spam) <= 1:
            if not silent:
                print(colored(f"Constraint {index+1} is not in a form of inequality", "red"))
            return False
        elif len(spam) > 2:
            if not silent:
                print(colored(f"Constraint {index+1} has more than one inequality sign", "red"))
            if spam[0].replace('.', '', 1).isdigit():
                egg = [spam[0], spam[1]]
                for indexx in range(2, len(spam)):
                    egg[1] = egg[1] + "".join(filter(lambda x: x in ["=", "<", ">"], prop[len(egg[0]):len(egg[0]) + 2])) + spam[indexx]
                spam = egg
                print(spam)
            elif spam[-1].replace('.', '', 1).isdigit():
                egg = [spam[0]]
                for indexx in range(1, len(spam)-1):
                    egg[0] = egg[0] + "".join(filter(lambda x: x in ["=", "<", ">"], prop[len(egg[0]):len(egg[0]) + 2])) + spam[indexx]
                egg.append(spam[-1])
                spam = egg
                print(spam)
            else:
                return False
            # ## Searching for < > which were in brackets
            # brackets = []
            # for part in spam:
            #     brackets.append(part.count("(") - part.count(")"))
            # brackets_count = 0
            # ## more right brackets in the first part
            # if brackets[0] < 0:
            #     return False
            # ## sum the brackets until I find balance
            # for index, part in enumerate(brackets):
            #     brackets_count = brackets_count + part
            #     ## found the split
            #     if brackets_count == 0 and sum(brackets[index +1:]):
            #         TODO
        try:
            ## The right-hand-side is number
            float(spam[1])
            if debug:
                print("right-hand-side ", float(spam[1]))
        except ValueError:
            spam = [f"{spam[0]} -( {spam[1]})", 0]

        ## If we are at odd position check
        if is_odd:
            if debug:
                print("is odd")
                print("funcs[-1]", funcs[-1])
                print("spam[0]", spam[0])
            ## whether the previous function is the same as the new one
            if funcs[-1] == spam[0]:
                # if yes, add the other value of the interval
                if debug:
                    print("Adding value")
                    print("len(funcs)", len(funcs))
                    print("[intervals[len(funcs)-1], spam[1]]", [intervals[len(funcs)-1], spam[1]])
                intervals[len(funcs)-1] = [intervals[len(funcs)-1], spam[1]]
        else:
            funcs.append(spam[0])
            intervals.append(spam[1])
        is_odd = not is_odd
        index = index + 1

    ## Sort the intervals
    index = 0
    for interval_index in range(len(intervals)):
        if len(intervals[interval_index]) != 2:
            if not silent:
                print(colored(f"Constraint {index + 1} does not have proper number of boundaries", "red"))
            raise Exception(f"Constraint {index + 1} does not have proper number of boundaries")
        if debug:
            print("sorted([float(intervals[interval_index][0]), float(intervals[interval_index][1])])", sorted([float(intervals[interval_index][0]), float(intervals[interval_index][1])]))
        intervals[interval_index] = sorted([float(intervals[interval_index][0]), float(intervals[interval_index][1])])
        if debug:
            print("Interval(intervals[interval_index][0], intervals[interval_index][1]) ", Interval(intervals[interval_index][0], intervals[interval_index][1]))
        intervals[interval_index] = Interval(intervals[interval_index][0], intervals[interval_index][1])
        index = index + 1

    if debug:
        print("funcs: ", funcs)
        print("intervals: ", intervals)

    return funcs, intervals


def decouple_constraints(constraints: list, silent: bool = True, debug: bool = False):
    """ Decouples constrains with more two inequalities into two separate constraints:

    Args:
        constraints  (list of strings): properties to be converted
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used

    Example:
        ["-8 <= x+3 <= 0"] -> ["-8 <= x+3", "x+3 <= 0"]
    """
    pattern = r" < | > | >= | <= | = | => | =<"
    new_constraints = []
    for index, constraint in enumerate(constraints):
        match = re.findall(pattern, constraint)
        if debug:
            print("constraint", constraint)
            print("match", match)
        if len(match) == 0:
            raise Exception("Constraints", f"No <,>,>=, <=,= symbols in constrain {index+1}")
        elif len(match) == 1:
            new_constraints.append(constraint)
        elif len(match) == 2:
            parts = re.split(pattern, constraint)
            new_constraints.append(match[0].join(parts[:2]))
            new_constraints.append(match[0].join(parts[1:]))
        else:
            raise Exception("Constraints", f"More than two <,>,>=, <=,= symbols in constrain {index+1}")
    return new_constraints


def add_white_spaces(expression):
    """ Adds white spaces in between <,>,=,<=, and >= so it can be easily parsed """
    just_equal = r"[^\s<>]=[^<>]|[^<>]=[^\s<>]"
    match = re.findall(just_equal, expression)
    print(match)
    if match:
        expression.replace("=", " = ")

    with_equal = r"[^\s]>=|[^\s]<=|[^\s]=>|[^\s]=<|>=[^\s]|<=[^\s]|=>[^\s]|=<[^\s]"
    match = re.findall(with_equal, expression)
    print(match)
    if match:
        greater_eq_check = True
        smaller_eq_check = True
        eq_greater_check = True
        eq_smaller_check = True
        for item in match:
            if ">=" in item and greater_eq_check:
                expression = expression.replace(">=", " >= ")
                greater_eq_check = False
            if "<=" in item and smaller_eq_check:
                expression = expression.replace("<=", " <= ")
                smaller_eq_check = False
            if "=>" in item and eq_greater_check:
                expression = expression.replace("=>", " >= ")
                greater_eq_check = False
            if "=<" in item and eq_smaller_check:
                expression = expression.replace("=<", " <= ")
                smaller_eq_check = False

    without_equal = r"<[^\s=]|>[^\s=]|[^\s=]<|[^\s=]>"
    match = re.findall(without_equal, expression)
    print(match)
    if match:
        greater_check = True
        smaller_check = True
        for item in match:
            if ">" in item and greater_check:
                expression = expression.replace(">", " > ")
                greater_check = False
            if "<" in item and smaller_check:
                expression = expression.replace("<", " < ")
                smaller_check = False

    return expression

# print(add_white_spaces("0.270794145078059 >= p*q >= 0.129205854921941"))
# print(add_white_spaces("0.270794145078059>=p*q>=0.129205854921941"))


def normalise_constraint(constraint: list, silent: bool = True, debug: bool = False):
    """ Transforms the set constraint into normalised form

    Args:
        constraint  (string): constraint to be normalised
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used

    Example:
          0.2 >= p >= 0.1    --->  0.1 <= p <= 0.2
          0.2 >= p           --->  p <= 0.2
    """
    add_white_spaces(constraint)
    pattern = r" < | > | >= | <= | = | => | =<"
    match = re.findall(pattern, constraint)
    if debug:
        print("constraint", constraint)
        print("match", match)
    print(re.split(pattern, constraint))

# normalise_constraint("0.270794145078059 >= p*q >= 0.129205854921941", debug=True)


def to_interval(points: list):
    """ Transforms the set of points into set of intervals

    Args:
        points (list of pairs): which are the points

    Example:
        [(0, 2), (1, 3)] --> [[0, 1], [2, 3]]
    """
    intervals = []
    for dimension in range(len(points[0])):
        interval = [points[0][dimension], points[0][dimension]]
        for point in range(len(points)):
            if interval[0] > points[point][dimension]:
                interval[0] = points[point][dimension]
            if interval[1] < points[point][dimension]:
                interval[1] = points[point][dimension]
        intervals.append(interval)
    return intervals
