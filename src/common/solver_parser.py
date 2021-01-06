from z3 import *
from numpy import mean


def parse_model_values(model, solver="z3"):
    """ Parses z3.solver.model() into list of values
        Ignores /0
    Args:
        model (z3.solver.model() or ): model to parse value from
        solver (str): solver from which the model comes from

    Example z3: [r_0 = 1/8, r_1 = 9/16, /0 = [(7/16, 7/8) -> 1/2, else -> 0]] -> [0.125, 0.5625]
    Example dreal: "r_0 : [] \nr_1: []"
    """
    # example_points = re.findall(r'[0-9./]+', str(model))
    # print(example_points)

    if solver == "z3":
        # via z3 guide
        values = []
        for param in model.decls():
            spam = model[param]
            if isinstance(spam, AlgebraicNumRef):
                spam = spam.approx(20)
            if str(param) == "/0":
                continue
            values.append(spam)
        #
        # ## Delete brackets
        # model = model[1:-1]
        # ## Delete /0 part
        # if "/0" in model:
        #     model = model.split("/0")[0]
        #     ## Delete spaces
        #     model = re.sub(r', +', ",", model)
        #     ## Delete last comma
        #     if model[-1] == ",":
        #         model = model[:-1]
        # model = model.split(",")
        # model.sort()
        #
        # ## Parse the values
        # values = []
        # for value in model:
        #     try:
        #         values.append(float(eval(value.split("=")[1])))
        #     except SyntaxError as err:
        #         raise err
        return values

    elif solver == "dreal":
        values = []
        model = str(model)
        model = model.split("\n")
        for line in model:
            ## parse the value
            line = line.split(":")[1]
            ## Delete brackets
            line = line.split("[")[1]
            line = line.split("]")[0]
            line = line.split(",")
            line = list(map(lambda x: float(x), line))
            try:
                values.append(float(mean(line)))
            except Exception as err:
                print("model", model)
                print("single parameter intervals", line)
                raise err
        return values

    else:
        raise NotImplementedError


def pass_models_to_sons(example, counterexample, index, threshold, solver):
    """ For a given example and counterexample (as a result of given solver)
        parse the values in the given dimension (index) where low and high are bounds of the dimension

    Args:
        example (z3.model, dreal._dreal_py.Box, or bool): example of satisfaction
        counterexample (z3.model, dreal._dreal_py.Box, or bool): counterexample of satisfaction
        index (int): index of dimension to parse the values - split between two sons
        threshold (float): threshold of splitting in the given dimension
        solver (str): solver from which the models (example and counterexample) comes from
    """
    ## Initialisation of example and counterexample
    model_low = [9, 9]
    model_high = [9, 9]

    ## Assign example and counterexample to children
    if example is False:
        ## If no example obtained
        model_low[0] = None
        model_high[0] = None
    else:
        ## Parse example
        example_points = parse_model_values(example, solver)
        ## compute above and exact
        if solver == "z3":

            s = Solver()
            s.add(example_points[index] > threshold)
            above = s.check() != z3.unsat

            s = Solver()
            s.add(example_points[index] == threshold)
            exact = s.check() != z3.unsat
        else:
            above = example_points[index] > threshold
            exact = example_points[index] == threshold

        if exact:
            ## as exact thresholds can be ambiguous for a solver we do not pass this information
            model_low[0] = None
            model_high[0] = None
        elif above:
            model_low[0] = None
            model_high[0] = example
        else:
            model_low[0] = example
            model_high[0] = None

    if counterexample is False:
        ## If no counterexample obtained
        model_low[1] = None
        model_high[1] = None
    else:
        ## Parse counterexample
        counterexample_points = parse_model_values(counterexample, solver)
        ## compute above and exact
        if solver == "z3":
            s = Solver()
            s.add(counterexample_points[index] > threshold)
            above = s.check() != z3.unsat

            s = Solver()
            s.add(counterexample_points[index] == threshold)
            exact = s.check() != z3.unsat
        else:
            above = counterexample_points[index] > threshold
            exact = counterexample_points[index] == threshold

        if exact:
            ## as exact thresholds can be ambiguous for a solver we do not pass this information
            model_low[1] = None
            model_high[1] = None
        elif above:
            model_low[1] = None
            model_high[1] = counterexample
        else:
            model_low[1] = counterexample
            model_high[1] = None

    if 9 in model_low or 9 in model_high:
        raise Exception

    return model_low, model_high