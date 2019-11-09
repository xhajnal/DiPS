from termcolor import colored
from sympy import Interval


def ineq_to_constraints(functions: list, intervals: list, silent: bool = True):
    """ Converts inequalities of the function given by the interval to properties

    Example: ["x+3"],[[0,1]] ->  ["x+3>=0","x+3<=1"]

    Args
    ----------
    functions:  (list of strings) array of functions
    intervals: (list of intervals) array of pairs, low and high bound
    silent: (Bool): if silent printed output is set to minimum

    :returns list of constraints
    """

    if len(functions) is not len(intervals):
        if not silent:
            print(colored(f"len of functions {len(functions)} and intervals {len(intervals)} does not correspond", "red"))
        return False

    ## Catching wrong interval errors
    try:
        spam = []
        for index in range(len(functions)):
            if isinstance(intervals[index], Interval):
                spam.append(functions[index] + ">=" + str(intervals[index].start))
                spam.append(functions[index] + "<=" + str(intervals[index].end))
            else:
                spam.append(functions[index] + ">=" + str(intervals[index][0]))
                spam.append(functions[index] + "<=" + str(intervals[index][1]))
        return spam
    except Exception as error:
        if "'EmptySet' object does not support indexing" in str(error):
            raise Exception("ineq_to_constraints", "Some intervals are incorrect (lover bound > upper bound)")
        elif "'FiniteSet' object does not support indexing" in str(error):
            raise Exception("ineq_to_constraints", "Some intervals are incorrect (empty)")
        else:
            raise error


def constraints_to_ineq(constraints: list, silent: bool = True, debug: bool = False):
    """ Converts properties to functions and inequalities if possible

    Example: ["x+3>=0","x+3<=1"] -> ["x+3"],[[0,1]]

    Args
    ----------
    constraints:  (list of strings) properties to be converted
    silent: (Bool): if silent printed output is set to minimum
    debug: (Bool) if True extensive print will be used
    """
    if debug:
        silent = False
    if len(constraints) % 2:
        if not silent:
            print(colored("Number of properties is not even, some interval will be invalid", "red"))
        return False
    funcs = []
    intervals = []
    is_odd = False
    index = 0

    for prop in constraints:
        spam = "None"
        if debug:
            print(f"property {index + 1} before splitting", prop)
        try:
            prop = prop.replace("<=", "<").replace(">=", "<").replace("=>", "<").replace("=<", "<").replace(">", "<")
            spam = prop.split("<")
        except AttributeError:
            print()
        if debug:
            print(f"property {index+1} after splitting", spam)
        if len(spam) <= 1:
            if not silent:
                print(colored(f"Property {index+1} is not in a form of inequality", "red"))
            return False
        elif len(spam) > 2:
            if not silent:
                print(colored(f"Property {index+1} has more than one inequality sign", "red"))
            return False
        else:
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
        if len(intervals[interval_index]) is not 2:
            if not silent:
                print(colored(f"Property {index + 1} does not have proper number of boundaries", "red"))
            return False
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


def to_interval(points: list):
    """ Transforms the set of points into set of intervals

    Args
    ----------
    points: (list of pairs) which are the points
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
