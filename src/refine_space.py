import datetime
import os
import socket
import sys
from platform import system
from time import strftime, localtime, time
from collections.abc import Iterable
from termcolor import colored
from mpmath import mpi

## Importing my code
from common.solver_parser import pass_models_to_sons
from common.model_stuff import find_param
from rectangle import My_Rectangle
from sample_space import sample_space as sample
from space import RefinedSpace
from common.convert import to_interval, decouple_constraints, normalise_constraint, put_exp_left
from common.convert import constraints_to_ineq
from common.config import load_config
from common.space_stuff import refine_by, is_in, get_rectangle_volume

gIsDrealAvailable = False

if "wind" not in system().lower():
    try:
        from dreal import logical_and, logical_or, logical_not, Variable, CheckSatisfiability
        gIsDrealAvailable = True
    except ImportError as err:
        print(f"Error while loading dreal {err}")
        gIsDrealAvailable = False

config = load_config()
results_dir = config["results"]
refinement_results = config["refinement_results"]
refine_timeout = config["refine_timeout"]
z3_path = config["z3_path"]
del config

# import struct
# print("You are running "+ str(struct.calcsize("P") * 8)+"bit Python, please verify that installed z3 is compatible")
# print("path: ", os.environ["PATH"])

# print(os.environ["PATH"])

cwd = os.getcwd()

try:
    from z3 import *

    os.chdir(cwd)
    p = Real('p')
except ImportError:
    if not os.path.exists(z3_path):
        raise OSError("Directory does not exist: " + str(z3_path))
    print("z3_path", z3_path)
    os.chdir(cwd)

    ##  Add z3 to PATH
    if '/' in z3_path:
        z3_path_short = '/'.join(z3_path.split("/")[:-1])
    elif '\\' in z3_path:
        z3_path_short = '\\'.join(z3_path.split("\\")[:-1])
    else:
        z3_path_short = ""
        print("Warning: Could not set path to add to the PATH, please add it manually")

    if "PATH" not in os.environ:
        os.environ["PATH"] = z3_path
    else:
        if z3_path_short not in os.environ["PATH"]:
            if z3_path_short.replace("/", "\\") not in os.environ["PATH"]:
                if "wind" in system().lower():
                    os.environ["PATH"] = os.environ["PATH"] + ";" + z3_path_short
                else:
                    os.environ["PATH"] = os.environ["PATH"] + ":" + z3_path_short
    sys.path.append(z3_path)

    ## Add z3 to PYTHON PATH
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = z3_path
    else:
        if z3_path not in os.environ["PYTHONPATH"]:

            if "wind" in system().lower():
                os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ";" + z3_path
            else:
                os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + z3_path

    ## Add z3 to LDLIB PATH
    if "wind" not in system().lower():
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":" + z3_path_short
        else:
            os.environ["LD_LIBRARY_PATH"] = z3_path_short

    ##  Add z3 to other variables
    os.environ["Z3_LIBRARY_PATH"] = z3_path_short
    os.environ["Z3_LIBRARY_DIRS"] = z3_path_short

    ## Try to import z3 from a given folder
    try:
        from z3 import *
        # print(os.getcwd())
        # import subprocess
        # subprocess.call(["python", "example.py"])
    except ImportError:
        raise Exception("could not load z3 from: ", z3_path)

## Try to run z3
try:
    p = Real('p')
except NameError:
    raise Exception("z3 not loaded properly")


## This is depricated
# def call_refine_from_que(space: RefinedSpace, queu, alg=4):
#     globals()["space"] = space
#     globals()["que"] = queu
#     globals()["start_time"] = time()
#     globals()["parameters"] = space.get_params()
#     globals()["init_coverage"] = space.get_coverage()
#     # assert globals()["init_coverage"] == 0
#
#     start_time = time()
#
#     while globals()["que"].size() > 0:
#         ## TODO refactor this as it is using old methods
#         if alg == 1:
#             raise NotImplementedError("alg 1 does not work with queue, we cannot use it")
#         elif alg == 2:
#             private_check_deeper(*globals()["que"].dequeue())
#         elif alg == 3:
#             private_check_deeper_checking(*globals()["que"].dequeue())
#         elif alg == 4:
#             private_check_deeper_checking_both(*globals()["que"].dequeue())
#         elif alg == 5:
#             raise NotImplementedError("Interval arithmetic not implemented so far fo MHMH method")
#
#     ## Refresh parameters:
#     for param in globals()["parameters"]:
#         del globals()[param]
#
#     ## Saving how much time refinement took
#     space.refinement_took(time() - start_time)


def are_param_types_assigned():
    """ Verifies that all params types are assigned. """
    try:
        for param in globals()["parameters"]:
            spam = globals()[param]
        return True
    except KeyError:
        return False
    except Exception:
        raise Exception("Uncaught exception raised during checking parameter types.")


def assign_param_types(solver):
    """ Assigns parameter types to parameters.

    Args:
        solver (string): solver "z3" or "dreal"
    """
    for param in globals()["parameters"]:
        if solver == "z3":
            globals()[param] = Real(param)
        elif solver == "dreal":
            globals()[param] = Variable(param)
        else:
            try:
                raise Exception(f"Unknown solver: {solver}")
            except:
                raise Exception("Unknown solver.")
    ## EXAMPLE: p = Real(p)


def check_unsafe(region, constraints, silent: bool = False, called=False, solver="z3", delta=0.001,
                 debug: bool = False):
    """ Check if the given region is unsafe or not using z3 or dreal.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of functions (rational functions in the case of Markov Chains)
        silent (bool): if silent printed output is set to minimum
        called (bool): True for standalone call - it updates the global variables and creates new space
        solver (string): specified solver, allowed: z3, dreal
        delta (number): used for delta solving using dreal
        debug (bool): if True extensive print will be used
    """
    ## Initialisation
    if debug:
        silent = False

    if isinstance(region, My_Rectangle):
        region = region.region

    print(f"Checking unsafe {region} using {('dreal', 'z3')[solver == 'z3']} solver, current time is {datetime.datetime.now()}") if not silent else None

    if solver == "z3":  ## avoiding collision name
        del delta
    elif solver == "dreal":
        if not gIsDrealAvailable:
            raise Exception("Dreal is not properly loaded, install it or use z3 instead, please.")

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynomial in constraints:
            parameters.update(find_param(polynomial))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        assign_param_types(solver)

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]
        ## Check whether the parameters are stored as globals
        if not are_param_types_assigned():
            assign_param_types(solver)

    b = time()
    ## Choosing solver
    if solver == "z3":
        set_param(max_lines=1, max_width=1000000)
        s = Solver()

        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        for param in globals()["parameters"]:
            if debug:
                print(f"globals()[param] {globals()[param]}")
                print(f"region[{j}] {region[j]}")
            s.add(globals()[param] > float(region[j][0]))
            s.add(globals()[param] < region[j][1])
            j = j + 1

        ## Adding properties to solver
        for i in range(0, len(constraints)):
            # print(f"constraints[{i}] {constraints[i]}") if debug else None
            try:
                s.add(eval(constraints[i]))
            except Z3Exception as z3_err:
                print(z3_err)
                print(f"constraints[{i}] {constraints[i]}")
                print(f"evaled constraints[{i}] {eval(constraints[i])}")

        check = s.check()
        ## If there is no example of satisfaction, hence all unsat, hence unsafe, hence red
        space.time_smt = space.time_smt + time() - b
        if check == unsat:
            print(f'The region {region} is {colored("unsafe", "red")}') if not silent else None
            space.add_red(region)
            return True
        elif check == unknown:
            return False
        ## Else there is an example for satisfaction, hence not all unsat
        else:
            print(f"Counterexample of unsafety: {s.model()}") if debug else None
            return s.model()

    elif solver == "dreal":
        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        f_sat = None
        for param in globals()["parameters"]:
            if debug:
                print(f"globals()[param] {globals()[param]}")
                print(f"region[{j}] {region[j]}")
            ## TODO possibly a problematic when changing the solver with the same space
            globals()[param] = Variable(param)
            if j == 0:
                f_sat = logical_and(globals()[param] > region[j][0], globals()[param] < region[j][1])
            else:
                f_sat = logical_and(f_sat, globals()[param] > region[j][0])
                f_sat = logical_and(f_sat, globals()[param] < region[j][1])
            j = j + 1

        ## Adding properties to dreal solver
        for i in range(0, len(constraints)):
            # print(f"constraints[{i}] {constraints[i]}") if debug else None
            f_sat = logical_and(f_sat, eval(constraints[i]))

        result = CheckSatisfiability(f_sat, delta)

        space.time_smt = space.time_smt + time() - b
        if result is not None:
            print(f"Counterexample of unsafety: {result}") if debug else None
            return result
        else:
            space.add_red(region)
            print(f'The region {region} is {colored("unsafe", "red")}') if not silent else None
            return True


def check_safe(region, constraints, silent: bool = False, called=False, solver="z3", delta=0.001, debug: bool = False):
    """ Check if the given region is safe or not using z3 or dreal.

    It means whether for all parametrisations in **region** every property(prop) is evaluated within the given
    **interval**, otherwise it is not safe and counterexample is returned.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        silent (bool): if silent printed output is set to minimum
        called (bool): True for standalone call - it updates the global variables and creates new space
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        debug (bool): if True extensive print will be used
    """
    ## Initialisation
    if debug:
        silent = False

    if isinstance(region, My_Rectangle):
        region = region.region

    print(f"Checking safe {region} using {('dreal', 'z3')[solver == 'z3']} solver, current time is {datetime.datetime.now()}") if not silent else None

    if solver == "z3":  ## avoiding collision name
        del delta
    elif solver == "dreal":
        if not gIsDrealAvailable:
            raise Exception("Dreal is not properly loaded, install it or use z3 instead, please.")

    if called:
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynomial in constraints:
            parameters.update(find_param(polynomial))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        assign_param_types(solver)
        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]
        ## Check whether the parameters are stored as globals
        if not are_param_types_assigned():
            assign_param_types(solver)

    b = time()
    ## Choosing solver
    if solver == "z3":
        set_param(max_lines=1, max_width=1000000)
        ## Initialisation of z3 solver
        s = Solver()

        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        for param in globals()["parameters"]:
            if debug:
                print(f"globals()[param] {globals()[param]}")
                print(f"region[{j}] {region[j]}")
            s.add(globals()[param] > region[j][0])
            s.add(globals()[param] < region[j][1])
            j = j + 1

        ## Adding properties to z3 solver
        formula = Not(eval(constraints[0]))
        for i in range(1, len(constraints)):
            formula = Or(formula, Not(eval(constraints[i])))
        print(f"formula {formula}") if debug else None
        s.add(formula)

        # print(s.check_unsafe())
        # return s.check_unsafe()
        check = s.check()
        space.time_smt = space.time_smt + time() - b

        ## If there is an example of falsification
        if check == sat:
            print(f"Counterexample of safety: {s.model()}") if debug else None
            return s.model()
        elif check == unknown:
            return False
        ## Else there is no example of falsification, hence all sat, hence safe, hence green
        else:
            print(f"The region {region} is " + colored("is safe", "green")) if  not silent else None
            space.add_green(region)
            return True

    elif solver == "dreal":
        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        f_sat = None
        for param in globals()["parameters"]:
            if debug:
                print(f"globals()[param] {globals()[param]}")
                print(f"region[{j}] {region[j]}")
            if j == 0:
                f_sat = logical_and(globals()[param] > region[j][0], globals()[param] < region[j][1])
            else:
                f_sat = logical_and(f_sat, globals()[param] > region[j][0])
                f_sat = logical_and(f_sat, globals()[param] < region[j][1])
            j = j + 1

        ## Adding properties to solver
        formula = logical_not(eval(constraints[0]))
        for i in range(1, len(constraints)):
            formula = logical_or(formula, logical_not(eval(constraints[i])))
        f_sat = logical_and(f_sat, formula)

        result = CheckSatisfiability(f_sat, delta)

        space.time_smt = space.time_smt + time() - b
        if result is None:
            print(f"The region {region} is " + colored("is safe", "green")) if  not silent else None
            space.add_green(region)
            return True
        else:
            print(f"Counterexample of safety: {result}") if debug else None
            return result


def check_deeper(region, constraints, recursion_depth, epsilon, coverage, silent, version=4, sample_size=False,
                 debug=False, save=False, title="", where=False, show_space=True, solver="z3", delta=0.001, gui=False,
                 iterative=False, timeout=0):
    """ Refining the parameter space into safe and unsafe regions with respective alg/method

    Args:
        region: (list of intervals or space) array of pairs, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        coverage (float): coverage threshold to stop computation
        silent (bool): if silent printed output is set to minimum
        version (Int): version of the algorithm to be used
        sample_size (Int): number of samples in dimension used for presampling
        debug (bool): if True extensive print will be used
        save (bool): if True output is stored
        title (string):: text to be added in Figure titles
        where (tuple/list): output matplotlib sources to output created figure
        show_space (bool): if show_space the refined space will be visualised
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        gui (bool or Callable): called from the graphical user interface
        iterative (bool) : iterative approach, TBD
        timeout (int): timeout in seconds (set 0 for no timeout)
    """

    ## TODO infer max_max_recursion, when it crashes if maxrecursion is higher
    max_max_recursion = 12
    # if recursion_depth > max_max_recursion:
    #     recursion_depth = max_max_recursion

    if iterative:
        raise NotImplementedError("This feature is deprecated and was removed.")

    # INITIALISATION
    initialisation_start_time = time()

    if version == 5:
        solver = "interval"

    if debug:
        silent = False

    ## Save file
    if save is True:
        # save = f"{},{n},{epsilon},{coverage},{version}"
        save = strftime("%d-%b-%Y-%H-%M-%S", localtime())
        # save = os.path.join(refinement_results, str(strftime("%d-%b-%Y-%H:%M:%S", localtime())))
        print(save) if debug else None

    ## Store the recursion_depth ## TODO DELETE THIS
    globals()["init_recursion_depth"] = recursion_depth

    ## Store whether init recursion_depth is 0
    globals()["flat_refinement"] = (recursion_depth == 0)

    ## If the given region is space
    if isinstance(region, RefinedSpace):
        space = region
        globals()["space"] = space
        del region
        region = space.region

        ## Check whether the set of params is equal
        print("space parameters: ", space.params) if not silent else None
        globals()["parameters"] = space.params
        parameters = globals()["parameters"]
        print("parsed parameters: ", parameters) if not silent else None

    ## If the region is just list of intervals - a space is to be created
    elif isinstance(region, list) or isinstance(region, tuple):
        globals()["parameters"] = set()

        if isinstance(constraints[0], list):
            for polynomial in constraints[0]:
                globals()["parameters"].update(find_param(polynomial))
        else:
            for polynomial in constraints:
                globals()["parameters"].update(find_param(polynomial))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        parameters = globals()["parameters"]
        print("parsed parameters: ", parameters) if debug else None

        ## Regions
        ### Taking care of unchangeable tuples
        if isinstance(region, tuple):
            region = list(region)

        for interval_index in range(len(region)):
            region[interval_index] = [region[interval_index][0], region[interval_index][1]]

        ## Constraints
        if not isinstance(constraints, Iterable):
            raise Exception("Refine Space",
                            "Given properties are not iterable, to use single property use list of length 1")

        ## Params
        globals()["space"] = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[],
                                          rectangles_unsat=[], title=title)
        space = globals()["space"]

        globals()["default_region"] = copy.deepcopy(region)
    else:
        raise Exception("Refine space", "region type is not accepted")

    ## Checking zero or negative size of dimension
    if space.get_volume() <= 0:
        raise Exception("Some dimension of the parameter space has nonpositive size.")

    if not silent:
        print("the whole area to be checked is: ", space.region)
        print("the volume of the whole area is:", space.get_volume())
        print()

    ## Reset space times
    space.time_check = 0
    space.time_smt = 0

    ## Convert max recursion into epsilon
    if recursion_depth >= 0 and epsilon > 0:
        epsilon = max(epsilon, space.get_volume()/2**(recursion_depth + 1))

    ## Checking coverage
    space_coverage = space.get_coverage()
    globals()["init_coverage"] = space_coverage
    # globals()["init_white_rectangles"] = len(space.get_white())

    if space_coverage >= coverage:
        print(colored(f"Space refinement - The coverage threshold already reached: {space_coverage} >= {coverage}", "green"))
        return space

    print(colored(f"Refinement initialisation took {socket.gethostname()} {round(time() - initialisation_start_time, 4)} seconds", "yellow")) if not silent else None
    start_time = time()
    globals()["start_time"] = start_time

    print("constraints", constraints) if debug else None

    ## Decoupling constraints
    if version != 5:
        ## In case of two inequalities on a line decouple it
        constraints = decouple_constraints(constraints, silent=silent, debug=debug)

    ## White space
    numb_of_white_rectangles = space.count_white_rectangles()
    globals()["numb_of_white_rectangles"] = numb_of_white_rectangles

    ## PRESAMPLING
    if sample_size:
        private_presample(region, constraints, sample_size, where, show_space, gui, save, silent, debug)

    # NORMAL REFINEMENT - WITHOUT/AFTER PRESAMPLING
    ## Parsing version/algorithm
    ## If using z3 initialise the parameters
    if version <= 4:
        index = 0
        for param in space.params:
            if space.types[index] == "Real":
                globals()[param] = Real(param)
            elif space.types[index] == "Int":
                globals()[param] = Int(param)
            elif space.types[index] == "Bool":
                globals()[param] = Bool(param)
            elif space.types[index][0] == "BitVec":
                globals()[param] = BitVec(param, space.types[index][1])
            else:
                print(colored(f"Type of parameter {param} which was set as {space.types[index]} does not correspond with any known type", "red"))
                raise TypeError(f"Type of parameter {param} which was set as {space.types[index]} does not correspond with any known type")
            index = index + 1

    ## Converting coupled constraints to inequalities
    elif version == 5:
        if isinstance(constraints[0], list):
            egg = constraints
        else:
            decoupled_constraints = decouple_constraints(constraints)
            decoupled_constraints = [normalise_constraint(item, silent=silent, debug=debug) for item in decoupled_constraints]
            decoupled_constraints = [put_exp_left(item, silent=silent, debug=debug) for item in decoupled_constraints]
            egg = constraints_to_ineq(decoupled_constraints, silent=silent, debug=debug)
        if not egg:
            return space
    else:
        print(colored("Chosen version not found", "red"))

    ## Iterating hyperectangles
    while space.get_coverage() < coverage:
        keys = space.rectangles_unknown.keys()
        key = max(keys)

        ## Check for size
        if key < epsilon < 0:
            print(colored("The biggest unknown rectangles are smaller than epsilon"), "blue")
            break

        ## check for timeout
        if time() - start_time > timeout > 0:
            print(colored(f"Timeout of {timeout} seconds reached", "blue"))
            break

        recursion_depth = recursion_depth - 1

        for item in copy.copy(space.rectangles_unknown[key]):
            # (region, constraints, recursion_depth, epsilon, coverage, silent, model=None, solver="z3", delta=0.01, debug: bool = False, progress=False, timeout=0)
            print(colored(f"We check rectangle {item.region} with given model {item.model}", "blue")) if not silent else None
            ## Select version

            if version == 2:
                print(colored(f"Selecting biggest rectangles method with {('dreal', 'z3')[solver == 'z3']} solver", "green")) if not silent else None
                result = private_check_deeper(item.region, constraints, solver=solver, delta=delta, silent=silent, debug=debug)
            elif version == 3:
                print(colored(f"Selecting biggest rectangles method with passing examples with {('dreal', 'z3')[solver == 'z3']} solver", "green")) if not silent else None
                result = private_check_deeper_checking(item.region, constraints, model=item.model, solver=solver, delta=delta, silent=silent, debug=debug)
            elif version == 4:
                print(colored(f"Selecting biggest rectangles method with passing examples and counterexamples with {('dreal', 'z3')[solver == 'z3']} solver", "green")) if not silent else None
                result = private_check_deeper_checking_both(item.region, constraints, model=item.model, solver=solver, delta=delta, silent=silent, debug=debug)
            elif version == 5:
                print(colored(f"Selecting biggest rectangles method with interval arithmetics", "green")) if not silent else None
                result = private_check_deeper_interval(item.region, egg[0], egg[1], silent=silent, debug=debug)

            ## Parse result
            if result is True:
                print(colored(f"Rectangle {item.region} safe", "green")) if not silent else None
            elif result is False:
                print(colored(f"Rectangle {item.region} unsafe", "red")) if not silent else None
            else:
                print(f"We split {item.region} into {result[0].region} and {result[1].region}") if not silent else None

            if gui and (result is True or result is False):
                if version == 5:
                    ## Fixing overflow of progress when refinement continuous for "flat" refinement
                    if globals()["flat_refinement"]:
                        ## Proportion of que lenght and initial number of white rectangles
                        gui(1 - globals()["que"].size() / globals()["numb_of_white_rectangles"])
                    else:
                        gui(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (coverage - globals()["init_coverage"]))
                else:
                    gui(False, (get_rectangle_volume(region) / space.get_volume()) / (coverage - globals()["init_coverage"]))

        print(colored(f"Current coverage is {space.get_coverage()}, current time is {datetime.datetime.now()}", "blue")) if not silent else None

        ## End if recursion depth was 0 (now -1)
        if recursion_depth == -1:
            break

    ## Saving how much time refinement took
    space.refinement_took(time() - start_time)

    ## VISUALISATION

    space.title = f"using max_recursion_depth:{globals()['init_recursion_depth']}, min_rec_size:{epsilon}, achieved_coverage:{str(space.get_coverage())}, alg{version}, {solver}"
    if not sample_size:
        ## If the visualisation of the space did not succeed space_shown = (None, error message)
        if show_space:
            space_shown = space.show(green=True, red=True, sat_samples=gui and len(space.params) <= 2,
                                     unsat_samples=gui and len(space.params) <= 2, save=save, where=where,
                                     show_all=not gui, is_presampled=False)
        else:
            space_shown = [False]
    else:  ## TODO THIS IS A HOTFIX
        if show_space:
            space_shown = space.show(green=True, red=True, sat_samples=gui and len(space.params) <= 2,
                                     unsat_samples=gui and len(space.params) <= 2, save=save, where=where,
                                     show_all=not gui, is_presampled=True)
        else:
            space_shown = [False]

    if not silent:
        print(colored(f"Result coverage is: {space.get_coverage()}", "blue"))
        print(colored(f"Refinement took {round(space.time_last_refinement, 4)} seconds", "yellow"))
        print(colored(f"Check calls took {round(space.time_check, 4)} seconds, {round(100 * space.time_check / space.time_last_refinement, 2)}% of refinement", "yellow"))
        print(colored(f"SMT calls took {round(space.time_smt, 4)} seconds, {round(100 * space.time_smt / space.time_check, 2) if space.time_check > 0 else None}% of checks, {round(100 * space.time_smt / space.time_last_refinement, 2)}% of refinement", "yellow"))
    if where:
        if space_shown[0] is None:
            return space, space_shown[1]
        else:
            return space
    else:
        return space


def check_deeper_iter(region, constraints, recursion_depth, epsilon, coverage, silent, solver="z3", delta=0.01,
                      debug: bool = False, timeout=0):
    """ New Refining the parameter space into safe and unsafe regions with iterative method using alg1

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        coverage (float): coverage threshold to stop computation
        silent (bool): if silent printed output is set to minimum
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        debug (bool): if True extensive print will be used
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    new_tresh = copy.deepcopy(region)

    ## TODO Implement ordering of the constraints with intervals
    for i in range(len(constraints) - 1):
        # print("white: ",globals()["hyper_rectangles_white"])
        print("check_deeper(", new_tresh, [constraints[i]], ")") if not silent else None
        check_deeper(new_tresh, [constraints[i]], recursion_depth, epsilon, coverage, True, 1)

        new_tresh = []
        for interval_index in range(len(region)):
            minimum = 9001
            maximum = 0
            ## Iterate though green regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_sat"])):
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0] < minimum:
                    minimum = globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1] > maximum:
                    maximum = globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1]
            ## Iterate though white regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_white"])):
                if globals()["hyper_rectangles_white"][rectangle_index][interval_index][0] < minimum:
                    minimum = globals()["hyper_rectangles_white"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_white"][rectangle_index][interval_index][1] > maximum:
                    maximum = globals()["hyper_rectangles_white"][rectangle_index][interval_index][1]
            new_tresh.append((minimum, maximum))

        print("Computed hull of nonred region is:", new_tresh) if not silent else None
        # globals()["hyper_rectangles_white"]=[new_tresh]
    globals()["default_region"] = None
    check_deeper(new_tresh, constraints, recursion_depth, epsilon, coverage, True, 1, solver=solver, delta=delta)


def check_interval_in(region, constraints, intervals, silent: bool = False, called=False, debug: bool = False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        called (bool): True for standalone call - it updates the global variables and creates new space
        debug (bool): if True extensive print will be used
    """
    if debug:
        silent = False

    print("Checking interval in", region, "current time is ", datetime.datetime.now()) if not silent else None

    if called:
        print("CALLED") if not silent else None
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynomial in constraints:
            parameters.update(find_param(polynomial))
        parameters = sorted(list(parameters))
        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]

    ## Assign each parameter its interval
    i = 0
    for param in globals()["parameters"]:
        globals()[param] = mpi(region[i][0], region[i][1])
        i = i + 1

    ## Check that all prop are in its interval
    i = 0

    b = time()
    for prop in constraints:
        # print(eval(prop))
        # print(intervals[i])
        # print(float(intervals[i].start), float(intervals[i].end))
        # print(mpi(float(intervals[i].start), float(intervals[i].end)))

        ## TODO THIS CAN BE OPTIMISED
        try:
            interval = mpi(float(intervals[i].start), float(intervals[i].end))
        except AttributeError:
            interval = mpi(float(intervals[i][0]), float(intervals[i][1]))

        if not eval(prop) in interval:
            print(f"property {constraints.index(prop) + 1} ϵ {eval(prop)}, which is not in the interval {interval}") if debug else None
            space.time_smt = space.time_smt + time() - b
            return False
        else:
            print(f'property {constraints.index(prop) + 1} ϵ {eval(prop)} {colored(" is safe", "green")}') if debug else None

        i = i + 1

    space.time_smt = space.time_smt + time() - b
    space.add_green(region)
    return True


def check_interval_out(region, constraints, intervals, silent: bool = False, called=False, debug: bool = False):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): of functions
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        called (bool): True for standalone call - it updates the global variables and creates new space
        debug (bool): if True extensive print will be used
    """
    if debug:
        silent = False

    print("Checking interval_out", region, "current time is ", datetime.datetime.now()) if not silent else None

    if called:
        print("CALLED") if not silent else None
        globals()["parameters"] = set()
        parameters = globals()["parameters"]
        for polynomial in constraints:
            parameters.update(find_param(polynomial))
        parameters = sorted(list(parameters))
        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        space = globals()["space"]
    ## Assign each parameter its interval
    i = 0
    for param in globals()["parameters"]:
        # print(f"param {} in interval {mpi(region[i][0], region[i][1])}")
        try:
            # print(mpi(region[i][0], region[i][1]))
            globals()[param] = mpi(region[i][0], region[i][1])
        except:
            print(
                f"Error occurred while region: {region}, with param {globals()[param]} of interval {mpi(region[i][0], region[i][1])}")

        i = i + 1

    ## Check that all prop are in its interval
    i = 0
    b = time()
    for prop in constraints:
        # print(eval(prop))
        # print(intervals[i])
        # print((intervals[i].start, intervals[i].end))
        # print(mpi(0,1) in mpi(0,2))
        # try:
        prop_eval = eval(prop)
        # except :
        #    raise ValueError("Error with prop: ", prop)

        ## TODO THIS CAN BE OPTIMISED
        try:
            ## print(intervals)
            interval = mpi(float(intervals[i].start), float(intervals[i].end))
        except AttributeError:
            interval = mpi(float(intervals[i][0]), float(intervals[i][1]))

        ## If there exists an intersection (neither of these interval is greater in all points)
        if not (prop_eval > interval or prop_eval < interval):
            print(f"property {constraints.index(prop) + 1} ϵ {eval(prop)}, which is not outside of interval {interval}") if debug else None
        else:
            space.time_smt = space.time_smt + time() - b
            space.add_red(region)
            print(f'property {constraints.index(prop) + 1} ϵ {eval(prop)} {colored(" is unsafe", "red")}') if debug else None
            return True
        i = i + 1

    space.time_smt = space.time_smt + time() - b
    return False


def private_check_deeper(region, constraints, solver="z3", delta=0.01, silent=False, debug=False):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    space = globals()["space"]

    if check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        space.remove_white(region)
        return False
    elif check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        space.remove_white(region)
        return True

    ## Find index of maximum dimension to be split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]

    ## Compute the half of region
    threshold = low + (high - low) / 2

    ##  Update space
    foo = copy.deepcopy(region)
    foo[index] = (low, threshold)
    foo2 = copy.deepcopy(region)
    foo2[index] = (threshold, high)
    space.remove_white(region)

    rectangle_low = My_Rectangle(foo, is_white=True)
    rectangle_high = My_Rectangle(foo2, is_white=True)
    space.add_white(rectangle_low)
    space.add_white(rectangle_high)

    return rectangle_low, rectangle_high


def private_check_deeper_checking(region, constraints, model=None,  solver="z3", delta=0.01, silent=False, debug=False):
    """ WE SUGGEST USING METHOD PASSING BOTH, EXAMPLE AND COUNTEREXAMPLE
        Refining the parameter space into safe and unsafe regions with passing examples,

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        model (pair of example, counterexample): of the satisfaction in the given region
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    space = globals()["space"]

    if model is None:
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    elif model[0] is None:
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    else:
        print("Skipping check_unsafe at", region, "since example", model[0]) if not silent else None
        example = model[0]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        return False
    elif check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        space.remove_white(region)
        return True

    ## Find index of maximum dimension to be split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]

    ## Compute the half of region
    threshold = low + (high - low) / 2

    ##  Update space
    foo = copy.deepcopy(region)
    foo[index] = (low, threshold)
    foo2 = copy.deepcopy(region)
    foo2[index] = (threshold, high)
    space.remove_white(region)
    # space.add_white(foo)
    # space.add_white(foo2)

    ## Compute passing example and counterexample to sons
    model_low, model_high = pass_models_to_sons(example, False, index, threshold, solver)
    assert isinstance(model_low, list)
    assert len(model_low) == 2
    assert isinstance(model_high, list)
    assert len(model_high) == 2

    rectangle_low = My_Rectangle(foo, is_white=True, model=model_low)
    rectangle_high = My_Rectangle(foo2, is_white=True, model=model_high)
    space.add_white(rectangle_low)
    space.add_white(rectangle_high)

    return rectangle_low, rectangle_high


def private_check_deeper_checking_both(region, constraints, model=None, solver="z3", delta=0.01, silent=False, debug=False):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints (list of strings): list of properties
        model (pair of example, counterexample): of the satisfaction in the given region
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    space = globals()["space"]

    ## Resolving if the region safe/unsafe/unknown
    ## If both example and counterexample are None
    if model is None:
        a = time()
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        space.time_check = space.time_check + time() - a
        # print("example", example)
        if example is True:
            counterexample = None
        else:
            a = time()
            counterexample = check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
            space.time_check = space.time_check + time() - a
            ## Elif the example is None
    elif model[0] is None:
        a = time()
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        space.time_check = space.time_check + time() - a
    else:
        print("Skipping check_unsafe at", region, "since example", model[0]) if not silent else None
        example = model[0]
    ## If counterexample is not None
    if model is not None:
        if model[1] is None:
            a = time()
            counterexample = check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
            space.time_check = space.time_check + time() - a
        else:
            print("Skipping check_safe at", region, "since counterexample", model[1]) if not silent else None
            counterexample = model[1]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        return False
    elif counterexample is True:
        space.remove_white(region)
        return True

    ## Find index of maximum dimension to be split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    rectangle_low = region[index][0]
    rectangle_high = region[index][1]

    ## Compute the half of region
    threshold = rectangle_low + (rectangle_high - rectangle_low) / 2

    ##  Update space
    foo = copy.deepcopy(region)
    foo[index] = (rectangle_low, threshold)
    foo2 = copy.deepcopy(region)
    foo2[index] = (threshold, rectangle_high)
    space.remove_white(region)
    # space.add_white(foo)
    # space.add_white(foo2)

    ## Compute passing example and counterexample to sons
    model_low, model_high = pass_models_to_sons(example, counterexample, index, threshold, solver)
    assert isinstance(model_low, list)
    assert len(model_low) == 2
    assert isinstance(model_high, list)
    assert len(model_high) == 2

    rectangle_low = My_Rectangle(foo, is_white=True, model=model_low)
    rectangle_high = My_Rectangle(foo2, is_white=True, model=model_high)
    space.add_white(rectangle_low)
    space.add_white(rectangle_high)

    return rectangle_low, rectangle_high


def private_check_deeper_interval(region, constraints, intervals, silent=False, debug=False):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    space = globals()["space"]

    ## Resolve the result
    if check_interval_out(region, constraints, intervals, silent=silent, called=False, debug=debug) is True:
        space.remove_white(region)
        return False
    elif check_interval_in(region, constraints, intervals, silent=silent, called=False, debug=debug) is True:
        space.remove_white(region)
        return True

    ## Find index of maximum dimension to be split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]

    ## Compute the half of region
    threshold = low + (high - low) / 2

    ##  Update space
    foo = copy.deepcopy(region)
    foo[index] = (low, threshold)
    foo2 = copy.deepcopy(region)
    foo2[index] = (threshold, high)
    space.remove_white(region)
    # space.add_white(foo)
    # space.add_white(foo2)

    rectangle_low = My_Rectangle(foo, is_white=True)
    rectangle_high = My_Rectangle(foo2, is_white=True)
    space.add_white(rectangle_low)
    space.add_white(rectangle_high)

    return rectangle_low, rectangle_high


def private_presample(region, constraints, sample_size, where, show_space, gui, save, silent, debug):
    start_time = time()
    space = globals()["space"]
    if not ([region] == space.get_flat_white()):
        raise Exception("Presampling of prerefined space is not implemented yet.")

    ## If there are some samples already
    sat_points = space.get_sat_samples()
    unsat_points = space.get_unsat_samples()
    samples = sat_points + unsat_points
    current_sample_size = 0
    if samples:
        current_sample_size = int(len(samples) ** (1 / len(region)))

    if current_sample_size < sample_size:
        ## Else run sampling
        sample(space, constraints, sample_size, compress=True, silent=not debug, save=save)
        sat_points = space.get_sat_samples()
        unsat_points = space.get_unsat_samples()
        current_sample_size = sample_size

    sample_size = current_sample_size

    if debug:
        print("Satisfying points: ", sat_points)
        print("Unsatisfying points: ", unsat_points)
    if debug and save:
        print("I am showing sampling_sat_" + str(save))
    if not where and show_space:
        space.show(red=False, green=False, sat_samples=True, unsat_samples=False, save=save, where=where,
                   show_all=not gui)

    ## COMPUTING THE ORTHOGONAL HULL OF SAT POINTS
    ## Initializing the min point and max point as the first point from the list
    if sat_points:
        sat_min = copy.deepcopy(sat_points[0])
        print("initial min", sat_min) if debug else None
        sat_max = copy.deepcopy(sat_points[0])
        print("initial max", sat_max) if debug else None

        ## TODO - POSSIBLE OPTIMISATION HERE DOING IT IN THE REVERSE ORDER AND STOPPING IF A BORDER OF THE REGION IS ADDED
        for point in sat_points:
            print(point) if debug else None
            for dimension in range(0, len(sat_points[0])):
                print(point[dimension]) if debug else None
                if point[dimension] < sat_min[dimension]:
                    print("current point:", point[dimension], "current min:", sat_min[dimension], "change min") if debug else None
                    sat_min[dimension] = point[dimension]
                if point[dimension] > sat_max[dimension]:
                    print("current point:", point[dimension], "current max:", sat_max[dimension], "change max") if debug else None
                    sat_max[dimension] = point[dimension]
        print(f"Points bordering the sat hull are: {sat_min}, {sat_max}") if debug else None

        if is_in(region, to_interval([sat_min, sat_max])):
            print("The orthogonal hull of sat points actually covers the whole region")
        else:
            ## Expanding the hull in each dimension by value of 1 sample distance
            ## THIS FIXING WORKS ONLY FOR THE UNIFORM SAMPLING
            bordering_intervals = to_interval([sat_min, sat_max])
            for interval_index in range(len(bordering_intervals)):
                ## increase the space to the left
                bordering_intervals[interval_index][0] = max(region[interval_index][0],
                                                             bordering_intervals[interval_index][0] - (
                                                                         region[interval_index][1] -
                                                                         region[interval_index][0]) / (
                                                                         sample_size - 1))
                ## increase the space to the right
                bordering_intervals[interval_index][1] = min(region[interval_index][1],
                                                             bordering_intervals[interval_index][1] + (
                                                                         region[interval_index][1] -
                                                                         region[interval_index][0]) / (
                                                                         sample_size - 1))
            print(f"Intervals bordering the sat hull are: {bordering_intervals}") if not silent else None

            ## SPLIT THE WHITE REGION INTO 3-5 AREAS (in 2D) (DEPENDING ON THE POSITION OF THE HULL)
            space.remove_white(region)
            regions = refine_by(region, bordering_intervals, debug)
            for subregion in regions:
                space.add_white(subregion)
    else:
        print("No sat points in the samples") if not silent else None

    ## SHOW UNSAT POINTS
    if debug and save:
        print("I am showing sampling_unsat_" + str(save))
    if not where and show_space:
        space.show(red=False, green=False, sat_samples=False, unsat_samples=True, save=save, where=where,
                   show_all=not gui)

    numb_of_white_rectangles = globals()["numb_of_white_rectangles"]

    ## If there is only the default region to be refined in the whitespace
    if numb_of_white_rectangles == 1:
        ## COMPUTING THE ORTHOGONAL HULL OF UNSAT POINTS
        ## Initializing the min point and max point as the first point in the list
        if unsat_points:
            unsat_min = copy.deepcopy(unsat_points[0])
            print("initial min", unsat_min) if debug else None
            unsat_max = copy.deepcopy(unsat_points[0])
            print("initial max", unsat_max) if debug else None

            ## TODO - POSSIBLE OPTIMISATION HERE DOING IT IN THE REVERSE ORDER AND STOPPING IF A BORDER OF THE REGION IS ADDED
            for point in unsat_points:
                print(point) if debug else None
                for dimension in range(0, len(unsat_points[0])):
                    print(point[dimension]) if debug else None
                    if point[dimension] < unsat_min[dimension]:
                        print("current point:", point[dimension], "current min:", unsat_min[dimension], "change min") if debug else None
                        unsat_min[dimension] = point[dimension]
                    if point[dimension] > unsat_max[dimension]:
                        print("current point:", point[dimension], "current max:", unsat_max[dimension], "change max") if debug else None
                        unsat_max[dimension] = point[dimension]
            print(f"Intervals bordering the unsat hull are:: {unsat_min},{unsat_max}") if debug else None

            if is_in(region, to_interval([unsat_min, unsat_max])):
                print(colored("The orthogonal hull of unsat points actually covers the whole region", "blue")) if not silent else None
            else:
                ## Expanding the hull in each dimension by value of 1 sample distance
                ## THIS FIXING WORKS ONLY FOR THE UNIFORM SAMPLING
                bordering_intervals = to_interval([unsat_min, unsat_max])
                for interval_index in range(len(bordering_intervals)):
                    ## increase the space to the left
                    bordering_intervals[interval_index][0] = max(region[interval_index][0],
                                                                 bordering_intervals[interval_index][0] - (
                                                                             region[interval_index][1] -
                                                                             region[interval_index][0]) / (
                                                                             sample_size - 1))
                    ## increase the space to the right
                    bordering_intervals[interval_index][1] = min(region[interval_index][1],
                                                                 bordering_intervals[interval_index][1] + (
                                                                             region[interval_index][1] -
                                                                             region[interval_index][0]) / (
                                                                             sample_size - 1))
                print(f"Intervals bordering the unsat hull are: {bordering_intervals}")

                ## SPLIT THE WHITE REGION INTO 3-5 AREAS (in 2D) (DEPENDING ON THE POSITION OF THE HULL)
                space.remove_white(region)
                regions = refine_by(region, bordering_intervals, debug)
                for subregion in regions:
                    space.add_white(subregion)
        else:
            print("No unsat points in the samples")

    if not silent:
        print("Presampling resulted in splicing the region into these subregions: ", space.get_white())
        print(colored(f"Presampling took {socket.gethostname()} {round(time() - start_time, 4)} second(s)", "yellow"))
        print()
