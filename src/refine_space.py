import datetime
import os
import socket
import sys
from platform import system
import itertools
from time import strftime, localtime, time
from collections.abc import Iterable
from termcolor import colored
from math import log
from mpmath import mpi
from matplotlib.patches import Rectangle

## Importing my code
from common.my_z3 import parse_model_values
from load import find_param
from sample_space import sample
from space import RefinedSpace
from space import get_rectangle_volume
from common.mathematics import is_in
from common.convert import to_interval, decouple_constraints
from common.convert import constraints_to_ineq
from common.queue import Queue
from common.config import load_config
from common.space_stuff import refine_by

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

    if not silent:
        print(
            f"Checking unsafe {region} using {('dreal', 'z3')[solver == 'z3']} solver, current time is {datetime.datetime.now()}")

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

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        ## The parameters are stored as globals
        space = globals()["space"]

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
            s.add(globals()[param] >= region[j][0])
            s.add(globals()[param] <= region[j][1])
            j = j + 1

        ## Adding properties to solver
        for i in range(0, len(constraints)):
            if debug:
                print(f"constraints[{i}] {constraints[i]}")
            try:
                s.add(eval(constraints[i]))
            except Z3Exception as z3_err:
                print(z3_err)
                print(f"constraints[{i}] {constraints[i]}")
                print(f"evaled constraints[{i}] {eval(constraints[i])}")

        check = s.check()
        ## If there is no example of satisfaction, hence all unsat, hence unsafe, hence red
        if check == unsat:
            if debug:
                print(f'The region {region} is {colored("is unsafe", "red")}')
            space.add_red(region)
            return True
        elif check == unknown:
            return False
        ## Else there is an example for satisfaction, hence not all unsat
        else:
            if debug:
                print(f"Counterexample of unsafety: {s.model()}")
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
                f_sat = logical_and(globals()[param] >= region[j][0], globals()[param] <= region[j][1])
            else:
                f_sat = logical_and(f_sat, globals()[param] >= region[j][0])
                f_sat = logical_and(f_sat, globals()[param] <= region[j][1])
            j = j + 1

        ## Adding properties to dreal solver
        for i in range(0, len(constraints)):
            if debug:
                print(f"constraints[{i}] {constraints[i]}")
            f_sat = logical_and(f_sat, eval(constraints[i]))

        result = CheckSatisfiability(f_sat, delta)

        if result is not None:
            if debug:
                print(f"Counterexample of unsafety: {result}")
            return result
        else:
            space.add_red(region)
            if debug:
                print(f'The region {region} is {colored("is unsafe", "red")}')
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

    if not silent:
        print(
            f"checking safe {region} using {('dreal', 'z3')[solver == 'z3']} solver, current time is {datetime.datetime.now()}")

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

        for param in globals()["parameters"]:
            if solver == "z3":
                globals()[param] = Real(param)
            elif solver == "dreal":
                globals()[param] = Variable(param)
            else:
                try:
                    raise Exception(f"Unknown solver: {solver}")
                except:
                    raise Exception(
                        "Unknown solver.")
        ## EXAMPLE: p = Real(p)

        space = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
    else:
        ## The parameters are stored as globals
        space = globals()["space"]

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
            s.add(globals()[param] >= region[j][0])
            s.add(globals()[param] <= region[j][1])
            j = j + 1

        ## Adding properties to z3 solver
        formula = Not(eval(constraints[0]))
        for i in range(1, len(constraints)):
            formula = Or(formula, Not(eval(constraints[i])))
        if debug:
            print(f"formula {formula}")
        s.add(formula)

        # print(s.check_unsafe())
        # return s.check_unsafe()
        check = s.check()
        ## If there is an example of falsification
        if check == sat:
            if debug:
                print(f"Counterexample of safety: {s.model()}")
            return s.model()
        elif check == unknown:
            return False
        ## Else there is no example of falsification, hence all sat, hence safe, hence green
        else:
            if debug:
                print(f"The region {region} is " + colored("is safe", "green"))
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
                f_sat = logical_and(globals()[param] >= region[j][0], globals()[param] <= region[j][1])
            else:
                f_sat = logical_and(f_sat, globals()[param] >= region[j][0])
                f_sat = logical_and(f_sat, globals()[param] <= region[j][1])
            j = j + 1

        ## Adding properties to solver
        formula = logical_not(eval(constraints[0]))
        for i in range(1, len(constraints)):
            formula = logical_or(formula, logical_not(eval(constraints[i])))
        f_sat = logical_and(f_sat, formula)

        result = CheckSatisfiability(f_sat, delta)

        if result is None:
            if debug:
                print(f"The region {region} is " + colored("is safe", "green"))
            space.add_green(region)
            return True
        else:
            if debug:
                print(f"Counterexample of safety: {result}")
            return result


def check_deeper(region, constraints, recursion_depth, epsilon, coverage, silent, version, sample_size=False,
                 debug=False, save=False, title="", where=False, show_space=True, solver="z3", delta=0.001, gui=False,
                 iterative=False, timeout=0):
    """ Refining the parameter space into safe and unsafe regions with respective alg/method

    Args:
        region: (list of intervals/space) array of pairs, low and high bound, defining the parameter space to be refined
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
        if debug:
            print(save)

    ## Store the recursion_depth
    globals()["init_recursion_depth"] = recursion_depth

    ## Store whether init recursion_depth is 0
    globals()["flat_refinement"] = (recursion_depth == 0)

    ## If the given region is space
    ## TODO correct this
    if isinstance(region, RefinedSpace):
        space = region
        globals()["space"] = space
        del region
        region = space.region

        ## Check whether the set of params is equal
        print("space parameters: ", space.params)
        globals()["parameters"] = space.params
        parameters = globals()["parameters"]
        print("parsed parameters: ", parameters)

        ## TODO add possible check like this
        # if not sorted(space.params) == sorted(parameters):
        #     raise Exception("The set of parameters of the given space and properties does not correspond")

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
        if debug:
            print("parsed parameters: ", parameters)

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

    ## Checking coverage
    space_coverage = space.get_coverage()
    globals()["init_coverage"] = space_coverage
    # globals()["init_white_rectangles"] = len(space.get_white())

    if space_coverage >= coverage:
        print(colored(f"Space refinement - The coverage threshold already reached: {space_coverage} >= {coverage}",
                      "green"))
        return space

    print(colored(f"Initialisation took {socket.gethostname()} {round(time() - initialisation_start_time, 2)} seconds",
                  "blue"))
    start_time = time()
    globals()["start_time"] = start_time

    if debug:
        print("constraints", constraints)

    ## Decoupling constraints
    if version != 5:
        ## In case of two inequalities on a line decouple it
        constraints = decouple_constraints(constraints, silent=silent, debug=debug)

    ## White space
    numb_of_white_rectangles = space.count_white_rectangles()
    globals()["numb_of_white_rectangles"] = numb_of_white_rectangles

    # PRESAMPLING
    if sample_size:
        if not ([region] == space.get_flat_white()):
            raise Exception("Presampling of prerefined space is not implemented yet.")

        if version == 1:
            print("Using presampled DFS method")
        elif version == 2:
            print("Using presampled BFS method")
            globals()["que"] = Queue()
        elif version == 3:
            print("Using presampled BFS method with passing examples")
            globals()["que"] = Queue()
        elif version == 4:
            print("Using presampled BFS method with passing examples and counterexamples")
            globals()["que"] = Queue()
        elif version == 5:
            print("Using presampled interval arithmetic")
            globals()["que"] = Queue()
        else:
            print(colored("Chosen version not found", "red"))
            return

        # globals()["space"] = RefinedSpace(copy.deepcopy(region), parameters, types=False, rectangles_sat=[], rectangles_unsat=[])
        # funcs, intervals = constraints_to_ineq(constraints)

        ## If there are some samples already
        samples = space.get_sat_samples() + space.get_unsat_samples()
        if samples:
            sat_points = space.get_sat_samples()
            unsat_points = space.get_unsat_samples()
            sample_size = int(len(samples) ** (1 / len(region)))
        else:
            to_be_searched = sample(space, constraints, sample_size, compress=True, silent=not debug, save=save)

            if debug:
                print("Sampled space type (should be array): ", type(to_be_searched))
                print("Sampled space as array: ", to_be_searched)

            ## CONVERT SAMPLED SPACE TO LIST
            print(to_be_searched)
            while not isinstance(to_be_searched[0][1], type(True)):
                to_be_searched = list(itertools.chain.from_iterable(to_be_searched))

            if debug:
                print("Sampled space type (should be list): ", type(to_be_searched))
                print("Unfolded sampled space: ", to_be_searched)
                print("An element from sampled space:", to_be_searched[0])

            ## PARSE SAT and UNSAT POINTS
            sat_points = []
            unsat_points = []
            for point in to_be_searched:
                ## If the point is True
                if point[1] is True:
                    sat_points.append(point[0])
                else:
                    unsat_points.append(point[0])
        if debug:
            print("Satisfying points: ", sat_points)
            print("Unsatisfying points: ", unsat_points)
        if debug and save:
            print("I am showing sampling_sat_" + str(save))
        if not where:
            space.show(red=False, green=False, sat_samples=True, unsat_samples=False, save=save, where=where,
                       show_all=not gui)

        ## COMPUTING THE ORTHOGONAL HULL OF SAT POINTS
        ## Initializing the min point and max point as the first point from the list
        if sat_points:
            sat_min = copy.deepcopy(sat_points[0])
            if debug:
                print("initial min", sat_min)
            sat_max = copy.deepcopy(sat_points[0])
            if debug:
                print("initial max", sat_max)

            ## TODO - POSSIBLE OPTIMISATION HERE DOING IT IN THE REVERSE ORDER AND STOPPING IF A BORDER OF THE REGION IS ADDED
            for point in sat_points:
                if debug:
                    print(point)
                for dimension in range(0, len(sat_points[0])):
                    if debug:
                        print(point[dimension])
                    if point[dimension] < sat_min[dimension]:
                        if debug:
                            print("current point:", point[dimension], "current min:", sat_min[dimension], "change min")
                        sat_min[dimension] = point[dimension]
                    if point[dimension] > sat_max[dimension]:
                        if debug:
                            print("current point:", point[dimension], "current max:", sat_max[dimension], "change max")
                        sat_max[dimension] = point[dimension]
            if debug:
                print(f"Points bordering the sat hull are: {sat_min}, {sat_max}")

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
                print(f"Intervals bordering the sat hull are: {bordering_intervals}")

                ## SPLIT THE WHITE REGION INTO 3-5 AREAS (in 2D) (DEPENDING ON THE POSITION OF THE HULL)
                space.remove_white(region)
                regions = refine_by(region, bordering_intervals, debug)
                for subregion in regions:
                    space.add_white(subregion)
        else:
            print("No sat points in the samples")

        ## SHOW UNSAT POINTS
        if debug and save:
            print("I am showing sampling_unsat_" + str(save))
        if not where:
            space.show(red=False, green=False, sat_samples=False, unsat_samples=True, save=save, where=where,
                       show_all=not gui)

        ## If there is only the default region to be refined in the whitespace
        if numb_of_white_rectangles == 1:
            ## COMPUTING THE ORTHOGONAL HULL OF UNSAT POINTS
            ## Initializing the min point and max point as the first point in the list
            if unsat_points:
                unsat_min = copy.deepcopy(unsat_points[0])
                if debug:
                    print("initial min", unsat_min)
                unsat_max = copy.deepcopy(unsat_points[0])
                if debug:
                    print("initial max", unsat_max)

                ## TODO - POSSIBLE OPTIMISATION HERE DOING IT IN THE REVERSE ORDER AND STOPPING IF A BORDER OF THE REGION IS ADDED
                for point in unsat_points:
                    if debug:
                        print(point)
                    for dimension in range(0, len(unsat_points[0])):
                        if debug:
                            print(point[dimension])
                        if point[dimension] < unsat_min[dimension]:
                            if debug:
                                print("current point:", point[dimension], "current min:", unsat_min[dimension],
                                      "change min")
                            unsat_min[dimension] = point[dimension]
                        if point[dimension] > unsat_max[dimension]:
                            if debug:
                                print("current point:", point[dimension], "current max:", unsat_max[dimension],
                                      "change max")
                            unsat_max[dimension] = point[dimension]
                if debug:
                    print(f"Intervals bordering the unsat hull are:: {unsat_min},{unsat_max}")

                if is_in(region, to_interval([unsat_min, unsat_max])):
                    print("The orthogonal hull of unsat points actually covers the whole region")
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

        print("Presampling resulted in splicing the region into these subregions: ", space.get_white())
        print(colored(f"Presampling took {socket.gethostname()} {round(time() - start_time)} second(s)", "yellow"))
        print()

    # NORMAL REFINEMENT - WITHOUT/AFTER PRESAMPLING
    ## Choosing version/algorithm here
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
                print(colored(
                    f"Type of parameter {param} which was set as {space.types[index]} does not correspond with any known type",
                    "red"))
                raise TypeError(
                    f"Type of parameter {param} which was set as {space.types[index]} does not correspond with any known type")
            index = index + 1

    ## Iterating through the regions
    if numb_of_white_rectangles == 1:
        rectangle = space.get_flat_white()[0]
        if version == 1:
            print(f"Using DFS method with {('dreal', 'z3')[solver == 'z3']} solver")
            private_check_deeper(rectangle, constraints, recursion_depth, epsilon, coverage, silent, solver=solver,
                                 delta=delta, debug=debug, progress=gui, timeout=timeout)
        elif version == 2:
            print(f"Using BFS method with {('dreal', 'z3')[solver == 'z3']} solver")
            globals()["que"] = Queue()
            private_check_deeper_queue(rectangle, constraints, recursion_depth, epsilon, coverage, silent,
                                       solver=solver, delta=delta, debug=debug, progress=gui, timeout=timeout)
        elif version == 3:
            print(f"Using BFS method with passing examples with {('dreal', 'z3')[solver == 'z3']} solver")
            globals()["que"] = Queue()
            private_check_deeper_queue_checking(rectangle, constraints, recursion_depth, epsilon, coverage, silent,
                                                model=None, solver=solver, delta=delta, debug=debug, progress=gui, timeout=timeout)
        elif version == 4:
            print(
                f"Using BFS method with passing examples and counterexamples with {('dreal', 'z3')[solver == 'z3']} solver")
            globals()["que"] = Queue()
            private_check_deeper_queue_checking_both(rectangle, constraints, recursion_depth, epsilon, coverage, silent,
                                                     model=None, solver=solver, delta=delta, debug=debug, progress=gui, timeout=timeout)
        elif version == 5:
            print("Using Interval arithmetic")
            globals()["que"] = Queue()

            ## If already feed with funcs, intervals
            if isinstance(constraints[0], list):
                egg = constraints
            else:
                egg = constraints_to_ineq(constraints, silent=silent, debug=debug)

            if not egg:
                return space
            if not silent:
                print("constraints", constraints)
                print("converted_intervals", egg)

            # private_check_deeper_interval(region, constraints, intervals, recursion_depth, epsilon, coverage, silent, debug=False, progress=False):
            private_check_deeper_interval(rectangle, egg[0], egg[1], recursion_depth, epsilon, coverage, silent,
                                          debug=debug, progress=gui, timeout=timeout)
        else:
            print(colored("Chosen version not found", "red"))
        space_coverage = space.get_coverage()
    else:
        ## Prerefined space with more white hyperectangles

        ## Copied so it is not changed while iteration
        copy_white_space = copy.deepcopy(space.get_white())
        keys = list(copy_white_space.keys())
        keys.sort(reverse=True)

        ## To get more similar result substituting the number of splits from the max_depth
        ## Setting the depth as a proportion
        ## TODO discuss this setting
        next_depth = max(0, recursion_depth - (int(log(numb_of_white_rectangles, 2))))
        if debug:
            print("max_depth = ", next_depth)

        if next_depth == 0:
            globals()["flat_refinement"] = True

        ## Iterating hyperectangles
        for volume in keys:
            if volume < epsilon:
                print(colored("Following rectangles are too small, skipping them", "blue"))
                break
            else:
                for index, rectangle in enumerate(copy_white_space[volume]):
                    # OLD Stuff
                    # ## Setting the coverage as lower value between desired coverage and the proportional expected coverage
                    # next_coverage = min(coverage, (space_coverage + (volume / space.get_volume()) * coverage))

                    ## Check if the que created (if alg1 used before it is not)
                    try:
                        type(globals()["que"])
                    except KeyError:
                        globals()["que"] = Queue()

                    if version == 1:
                        if debug:
                            print("refining", rectangle)
                            print("with constraints", constraints)
                        single_rectangle_start_time = time()
                        if not silent:
                            print(
                                f"Using DFS method with {('dreal', 'z3')[solver == 'z3']} solver to solve spliced rectangle number {index + 1} of {numb_of_white_rectangles}")
                        private_check_deeper(rectangle, constraints, next_depth, epsilon, coverage, silent,
                                             solver=solver, delta=delta, debug=debug, timeout=timeout)

                        ## Showing the step refinements of respective rectangles from the white space
                        ## If the visualisation of the space did not succeed space_shown = (None, error message)
                        space_coverage = space.get_coverage()
                        if not where:
                            space_shown = space.show(
                                title=f"max_recursion_depth:{next_depth}, min_rec_size:{epsilon}, achieved_coverage:{str(space_coverage)}, alg{version}, {solver}, \n Refinement took {socket.gethostname()} {round(time() - single_rectangle_start_time)} second(s)",
                                green=True, red=True, save=save, where=where, show_all=not gui)
                        if not silent:
                            print()
                        if space_coverage >= coverage:
                            break
                    elif version in [2, 3, 4]:
                        ## Add call to the Queue
                        if debug:
                            print("Adding ", rectangle, "to queue")
                            print("with constraints", constraints)
                        globals()["que"].enqueue(
                            [rectangle, constraints, next_depth, epsilon, coverage, silent, None, solver, delta, debug,
                             gui])
                    elif version == 5:
                        if debug:
                            print("Adding ", rectangle, "to queue")
                            print("with constraints", constraints)
                        ## If already feed with (funcs, intervals)
                        if isinstance(constraints[0], list):
                            egg = constraints
                        else:
                            egg = constraints_to_ineq(constraints, silent=silent, debug=debug)
                        if not egg:
                            return space
                        globals()["que"].enqueue(
                            [rectangle, egg[0], egg[1], next_depth, epsilon, coverage, silent, debug, gui])
                    else:
                        print(colored("Chosen version not found", "red"))
                        return space

        if globals()["que"].size() == 0:
            print(colored("No rectangles added, please decrease epsilon or increase recursion depth ", "blue"))

        if not silent and globals()["que"].size() > 0:
            if version == 2:
                print(colored(
                    f"Using BFS method with {('dreal', 'z3')[solver == 'z3']} solver to solve {numb_of_white_rectangles} white rectangles",
                    "blue"))
            elif version == 3:
                print(colored(
                    f"Using BFS method with passing examples with {('dreal', 'z3')[solver == 'z3']} solver to solve {numb_of_white_rectangles} white rectangles",
                    "blue"))
            elif version == 4:
                print(colored(
                    f"Using BFS method with passing examples and counterexamples with {('dreal', 'z3')[solver == 'z3']} solver to solve {numb_of_white_rectangles} white rectangles",
                    "blue"))
            elif version == 5:
                print(
                    colored(f"Using interval arithmetic to solve {numb_of_white_rectangles} white rectangles", "blue"))

            # print(colored(f"with proportional coverage: {next_coverage} and proportional depth {next_depth}", "blue")) ## old stuff
            print(colored(f"with proportional depth: {next_depth}", "blue"))

        while globals()["que"].size() > 0:
            single_rectangle_start_time = time()
            if version == 2:
                private_check_deeper_queue(*globals()["que"].dequeue())
            elif version == 3:
                private_check_deeper_queue_checking(*globals()["que"].dequeue())
            elif version == 4:
                private_check_deeper_queue_checking_both(*globals()["que"].dequeue())
            elif version == 5:
                private_check_deeper_interval(*globals()["que"].dequeue())

            space_coverage = space.get_coverage()
            if not where:
                space_shown = space.show(
                    title=f"max_recursion_depth:{next_depth}, min_rec_size:{epsilon}, achieved_coverage:{str(space_coverage)}, alg{version} \n Refinement took {socket.gethostname()} {round(time() - single_rectangle_start_time)} second(s)",
                    green=True, red=True, save=save, where=where, show_all=not gui)
            if not silent:
                print()
            if space_coverage >= coverage:
                globals()["que"] = Queue()
                break

    ## Saving how much time refinement took
    space.refinement_took(time() - start_time)

    ## VISUALISATION
    space.title = f"using max_recursion_depth:{recursion_depth}, min_rec_size:{epsilon}, achieved_coverage:{str(space_coverage)}, alg{version}, {solver}"
    if not sample_size:
        ## If the visualisation of the space did not succeed space_shown = (None, error message)
        if show_space:
            space_shown = space.show(green=True, red=True, sat_samples=gui and len(space.params) <= 2,
                                     unsat_samples=gui and len(space.params) <= 2, save=save, where=where,
                                     show_all=not gui)
        else:
            space_shown = [False]
    else:  ## TODO THIS IS A HOTFIX
        if show_space:
            space_shown = space.show(green=True, red=True, sat_samples=gui and len(space.params) <= 2,
                                     unsat_samples=gui and len(space.params) <= 2, save=save, where=where,
                                     show_all=not gui)
        else:
            space_shown = [False]
    print(colored(f"Result coverage is: {space_coverage}", "blue"))
    print(colored(f"Refinement took: {space.time_last_refinement} seconds", "yellow"))
    if where:
        if space_shown[0] is None:
            return space, space_shown[1]
        else:
            return space
    else:
        return space


def private_check_deeper(region, constraints, recursion_depth, epsilon, coverage, silent, solver="z3", delta=0.01,
                         debug: bool = False, progress=False, timeout=0):
    """ Refining the parameter space into safe and unsafe regions

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
        progress (function or False): function(update_to, update_by) to update progress
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    if debug:
        silent = False

        ## TODO check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Halt if timeout - done after checking as it is sequential

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"hyperrectangle {region} too small, skipped", "grey"))
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"rectangle {region} too small, skipped", "grey"))
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print("depth:", recursion_depth, colored(f"interval {region} too small, skipped", "grey"))
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    space_coverage = space.get_coverage()
    if space_coverage >= coverage:
        print(colored(f"coverage {space_coverage} is above the threshold", "blue"))
        return f"coverage {space_coverage} is above the threshold"

    # HERE MAY ADDING THE MODEL
    print("region", region)
    print("constraints", constraints)
    print("space", space.nice_print())
    if check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        result = "unsafe"
        if progress:
            progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                        coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "red"))
    elif check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        result = "safe"
        if progress:
            progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                        coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "green"))
    else:
        result = "unknown"
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "grey"))

    # print("result",result)
    if result == "safe" or result == "unsafe":
        space.remove_white(region)
        return result

    ## Find max interval and split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)  ## Add this region as white
    space.add_white(foo2)  ## Add this region as white
    # print("white area",globals()["hyper_rectangles_white"])

    ## Halt if max recursion reached
    if recursion_depth == 0:
        if not silent:
            print(f"maximal recursion reached here with coverage: {space.get_coverage()}")
        return result

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return result

    ## Call the alg for the children
    # private_check_deeper(region, constraints, recursion_depth, epsilon, coverage, silent, solver="z3", delta=0.01, debug: bool = False, progress=False)
    if silent:
        private_check_deeper(foo, constraints, recursion_depth - 1, epsilon, coverage, silent, solver, delta, debug,
                             progress, timeout)
        ## Probably not necessary
        # space_coverage = space.get_coverage()
        # if space_coverage >= coverage:
        #     print(colored(f"coverage {space.get_coverage()} is above the threshold", "blue"))
        #     return f"coverage {space.get_coverage()} is above the threshold"
        private_check_deeper(foo2, constraints, recursion_depth - 1, epsilon, coverage, silent, solver, delta, debug,
                             progress, timeout)
    else:
        print(recursion_depth, foo, space.get_coverage(),
              private_check_deeper(foo, constraints, recursion_depth - 1, epsilon, coverage, silent, solver, delta,
                                   debug, progress, timeout))
        ## Probably not necessary
        #  space_coverage = space.get_coverage()
        #  if space_coverage >= coverage:
        #     print(colored(f"coverage {space.get_coverage()} is above the threshold", "blue"))
        #     return f"coverage {space.get_coverage()} is above the threshold"
        print(recursion_depth, foo2, space.get_coverage(),
              private_check_deeper(foo2, constraints, recursion_depth - 1, epsilon, coverage, silent, solver, delta,
                                   debug, progress, timeout))


def private_check_deeper_queue(region, constraints, recursion_depth, epsilon, coverage, silent, model=None, solver="z3",
                               delta=0.01, debug: bool = False, progress=False, timeout=0):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        coverage (float): coverage threshold to stop computation
        silent (bool): if silent printed output is set to minimum
        model (None): does nothing for this alg, used only for queue compatibility
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        debug (bool): if True extensive print will be used
        progress (function or False): function(update_to, update_by) to update progress
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    if debug:
        silent = False

    ## TODO check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"hyperrectangle {region} too small, skipped", "grey"))
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"rectangle {region} too small, skipped", "grey"))
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print("depth:", recursion_depth, colored(f"interval {region} too small, skipped", "grey"))
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    space_coverage = space.get_coverage()
    if space_coverage >= coverage:
        globals()["que"] = Queue()
        print(colored(f"coverage {space_coverage} is above the threshold", "blue"))
        return "coverage ", space_coverage, " is above the threshold"

    ## HERE I CAN APPEND THE VALUE OF EXAMPLE AND COUNTEREXAMPLE
    if check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        result = "unsafe"
        if progress:
            progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                        coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "red"))
    elif check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        result = "safe"
        if progress:
            progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                        coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "green"))
    else:
        result = "unknown"
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "grey"))

    if result == "safe" or result == "unsafe":
        if debug:
            print("removing region:", region)
        space.remove_white(region)

    if result == "safe" or result == "unsafe":
        return

    ## Find maximum interval and split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Halt if max recursion reached
    if recursion_depth == 0:
        return

    ## Check if the queue created (if alg1 used before it wont)
    try:
        type(globals()["que"])
    except KeyError:
        globals()["que"] = Queue()

    ## Add calls to the Queue
    # private_check_deeper_queue(region, constraints, recursion_depth, epsilon, coverage, silent, solver="z3", delta=0.01, debug: bool = False, progress = False):
    globals()["que"].enqueue([copy.deepcopy(foo), constraints, recursion_depth - 1, epsilon, coverage, silent, None,
                              solver, delta, debug, progress, timeout])
    globals()["que"].enqueue([copy.deepcopy(foo2), constraints, recursion_depth - 1, epsilon, coverage, silent, None,
                              solver, delta, debug, progress, timeout])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue(*globals()["que"].dequeue())


def private_check_deeper_queue_checking(region, constraints, recursion_depth, epsilon, coverage, silent, model=None,
                                        solver="z3", delta=0.01, debug: bool = False, progress=False, timeout=0):
    """ THIS IS OBSOLETE METHOD, HERE JUST TO BE COMPARED WITH THE NEW ONE

    Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        coverage (float): coverage threshold to stop computation
        silent (bool): if silent printed output is set to minimum
        model (pair of example, counterexample): of the satisfaction in the given region
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        debug (bool): if True extensive print will be used
        progress (function or False): function(update_to, update_by) to update progress
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    if debug:
        silent = False

    ## TODO check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"hyperrectangle {region} too small, skipped", "grey"))
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"rectangle {region} too small, skipped", "grey"))
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print("depth:", recursion_depth, colored(f"interval {region} too small, skipped", "grey"))
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    space_coverage = space.get_coverage()
    if space_coverage >= coverage:
        globals()["que"] = Queue()
        print(colored(f"coverage {space_coverage} is above the threshold", "blue"))
        return f"coverage {space_coverage} is above the threshold"

    if model is None:
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        # counterexample = check_safe(region,prop,silent, solver=solver, delta=delta, debug=debug)
    elif model[0] is None:
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        if progress:
            progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                        coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} is unsafe", "red"))
        return
    elif check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        space.remove_white(region)
        if progress:
            progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                        coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()}  is safe", "red"))
        return
    else:  ## Unknown
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()}, {example}, is unknown", "grey"))

    ## Parse example
    example_points = parse_model_values(str(example), solver)

    # example_points = re.findall(r'[0-9./]+', str(example))
    # print(example_points)

    ## Find maximum dimension and split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Halt if max recursion reached
    if recursion_depth == 0:
        return

    ## Initialisation of example and counterexample
    model_low = [9, 9]
    model_high = [9, 9]

    ## Assign example and counterexample to children
    if example is False:
        model_low[0] = None
        model_high[0] = None
    else:
        if example_points[index] > low + (high - low) / 2:  ## skipped converting example point to float
            model_low[0] = None
            model_high[0] = example
        else:
            model_low[0] = example
            model_high[0] = None
        ## Overwrite if equal
        if example_points[index] == low + (high - low) / 2:  ## skipped converting example point to float
            model_low[0] = None
            model_high[0] = None

    ## Check if the queue created (if alg1 used before it wont)
    try:
        type(globals()["que"])
    except KeyError:
        globals()["que"] = Queue()

    ## Add calls to the Queue
    # private_check_deeper_queue_checking(region, constraints, recursion_depth, epsilon, coverage, silent, model=None, solver="z3", delta=0.01, debug: bool = False, progress = False):
    globals()["que"].enqueue([copy.deepcopy(foo), constraints, recursion_depth - 1, epsilon, coverage, silent,
                              model_low, solver, delta, debug, progress, timeout])
    globals()["que"].enqueue([copy.deepcopy(foo2), constraints, recursion_depth - 1, epsilon, coverage, silent,
                              model_high, solver, delta, debug, progress, timeout])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking(*globals()["que"].dequeue())


def private_check_deeper_queue_checking_both(region, constraints, recursion_depth, epsilon, coverage, silent,
                                             model=None, solver="z3", delta=0.01, debug: bool = False, progress=False,
                                             timeout=0):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints (list of strings): array of properties
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        coverage (float): coverage threshold to stop computation
        silent (bool): if silent printed output is set to minimum
        model (pair of example, counterexample): of the satisfaction in the given region
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        debug (bool): if True extensive print will be used
        progress (function or False): function(update_to, update_by) to update progress
        timeout (int): timeout in seconds (set 0 for no timeout)
    """

    ## TODO check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return f"timeout reached here with coverage: {space.get_coverage()}"

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"hyperrectangle {region} too small, skipped", "grey"))
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"rectangle {region} too small, skipped", "grey"))
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print("depth:", recursion_depth, colored(f"interval {region} too small, skipped", "grey"))
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    space_coverage = space.get_coverage()
    if space_coverage >= coverage:
        globals()["que"] = Queue()

        print(colored(f"coverage {space_coverage} is above the threshold", "blue"))
        return f"coverage {space_coverage} is above the threshold"

    ## Resolving if the region safe/unsafe/unknown
    ## If both example and counterexample are None
    if model is None:
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        # print("example", example)
        if example is True:
            counterexample = None
        else:
            counterexample = check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    ## elif the example is None
    elif model[0] is None:
        example = check_unsafe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    else:
        if not silent:
            print("skipping check_unsafe at", region, "since example", model[0])
        example = model[0]
    ## if counterexample is not None
    if model is not None:
        if model[1] is None:
            counterexample = check_safe(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        else:
            if not silent:
                print("skipping check_safe at", region, "since counterexample", model[1])
            counterexample = model[1]

    ## Resolving the result
    if example is True:
        space.remove_white(region)
        if progress:
            # progress(False, (((get_rectangle_volume(region) / space.get_volume()) / globals()["init_white_rectangles"]) / (coverage - globals()["init_coverage"])))
            progress(False,
                     (get_rectangle_volume(region) / space.get_volume()) / (coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} is unsafe", "red"))
        return
    elif counterexample is True:
        space.remove_white(region)
        if progress:
            progress(False,
                     (get_rectangle_volume(region) / space.get_volume()) / (coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()}  is safe", "green"))
        return
    else:  ## Unknown
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region,
                  colored(f"{space.get_coverage()} {(example, counterexample)} is unknown", "grey"))

    ## Parse example
    example_points = parse_model_values(str(example), solver)

    ## Parse counterexample
    counterexample_points = parse_model_values(str(counterexample), solver)

    ## Find maximum dimension and split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Halt if max recursion reached
    if recursion_depth == 0:
        return

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return f"timeout reached here with coverage: {space.get_coverage()}"

        ## Initialisation of example and counterexample
    model_low = [9, 9]
    model_high = [9, 9]

    ## Assign example and counterexample to children
    if example is False:
        model_low[0] = None
        model_high[0] = None
    else:
        if example_points[index] > low + (high - low) / 2:  ## skipped converting example point to float
            model_low[0] = None
            model_high[0] = example
        else:
            model_low[0] = example
            model_high[0] = None
        ## Overwrite if equal
        if example_points[index] == low + (high - low) / 2:  ## skipped converting example point to float
            model_low[0] = None
            model_high[0] = None

    if counterexample is False:
        model_low[1] = None
        model_high[1] = None
    else:
        if counterexample_points[index] > low + (high - low) / 2:  ## skipped converting example point to float
            model_low[1] = None
            model_high[1] = counterexample
        else:
            model_low[1] = counterexample
            model_high[1] = None

        ## Overwrite if equal
        if counterexample_points[index] == low + (high - low) / 2:  ## skipped converting example point to float
            model_low[1] = None
            model_high[1] = None

    ## Check if the que created (if alg1 used before it is not)
    try:
        type(globals()["que"])
    except KeyError:
        globals()["que"] = Queue()

    ## Add calls to the Queue
    # private_check_deeper_queue_checking_both(region, constraints, recursion_depth, epsilon, coverage, silent, model=None, solver="z3", delta=0.01, debug: bool = False, progress=False):
    globals()["que"].enqueue([copy.deepcopy(foo), constraints, recursion_depth - 1, epsilon, coverage, silent,
                              model_low, solver, delta, debug, progress, timeout])
    globals()["que"].enqueue([copy.deepcopy(foo2), constraints, recursion_depth - 1, epsilon, coverage, silent,
                              model_high, solver, delta, debug, progress, timeout])

    ## Execute the Queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_queue_checking_both(*globals()["que"].dequeue())


def color_margins(greater, smaller):
    """ Colors outside of the smaller region in the greater region as previously unsat

    Args:
        greater (list of intervals): region in which the smaller region is located
        smaller (list of intervals): smaller region which is not to be colored
    """
    # rectangles_sat.append(Rectangle((low_x,low_y), width, height, fc='g'))
    # print("greater ",greater)
    # print("smaller ",smaller)
    if greater is None or smaller is None:
        return

    ## If 1 dimensional coloring
    if len(smaller) == 1:
        ## Color 2 regions, to the left, to the right
        ## To the left
        globals()["rectangles_unsat_added"].append(
            Rectangle((greater[0][0], 0), smaller[0][0] - greater[0][0], 1, fc='r'))
        ## To the right
        globals()["rectangles_unsat_added"].append(
            Rectangle((smaller[0][1], 0), greater[0][1] - smaller[0][1], 1, fc='r'))

    ## Else 2 dimensional coloring
    elif len(smaller) == 2:
        ## Color 4 regions, to the left, to the right, below, and above
        ## TODO
        globals()["rectangles_unsat_added"].append(
            Rectangle((greater[0][0], 0), smaller[0][0] - greater[0][0], 1, fc='r'))
        ## TODO
        globals()["rectangles_unsat_added"].append(
            Rectangle((smaller[0][1], 0), greater[0][1] - smaller[0][1], 1, fc='r'))
        ## TODO
        globals()["rectangles_unsat_added"].append(
            Rectangle((smaller[0][0], 0), smaller[0][1] - smaller[0][0], smaller[1][0], fc='r'))
        ## TODO
        globals()["rectangles_unsat_added"].append(
            Rectangle((smaller[0][0], smaller[1][1]), smaller[0][1] - smaller[0][0], 1 - smaller[1][0], fc='r'))
    else:
        print("Error, trying to color more than 2 dimensional hyperrectangle")

    # re=[]
    # re.append(Rectangle((greater[0][0],0),smaller[0][0]-greater[0][0] , 1, fc='r'))
    # re.append(Rectangle((smaller[0][1],0),greater[0][1]-smaller[0][1] , 1, fc='r'))
    # re.append(Rectangle((smaller[0][0],0),smaller[0][1]-smaller[0][0] , smaller[1][0], fc='r'))
    # re.append(Rectangle((smaller[0][0],smaller[1][1]),smaller[0][1]-smaller[0][0] ,1- smaller[1][0], fc='r'))

    # fig = plt.figure()
    # pic = fig.add_subplot(111, aspect='equal')
    # pic.set_xlabel('p')
    # pic.set_ylabel('q')
    # pic.set_title("red = unsafe region, green = safe region, white = in between")
    # pc = PatchCollection(re,facecolor='r', alpha=0.5)
    # pic.add_collection(pc)


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
        if not silent:
            # print("white: ",globals()["hyper_rectangles_white"])
            print("check_deeper(", new_tresh, [constraints[i]], ")")
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

        if not silent:
            print("Computed hull of nonred region is:", new_tresh)
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

    if not silent:
        print("Checking interval in", region, "current time is ", datetime.datetime.now())

    if called:
        if not silent:
            print("CALLED")
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
            if debug:
                print(f"property {constraints.index(prop) + 1} ϵ {eval(prop)}, which is not in the interval {interval}")
            return False
        else:
            if debug:
                print(f'property {constraints.index(prop) + 1} ϵ {eval(prop)} {colored(" is safe", "green")}')

        i = i + 1

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

    if not silent:
        print("Checking interval_out", region, "current time is ", datetime.datetime.now())

    if called:
        if not silent:
            print("CALLED")
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
            if debug:
                print(
                    f"property {constraints.index(prop) + 1} ϵ {eval(prop)}, which is not outside of interval {interval}")
        else:
            space.add_red(region)
            if debug:
                print(f'property {constraints.index(prop) + 1} ϵ {eval(prop)} {colored(" is unsafe", "red")}')
            return True
        i = i + 1
    return False


def private_check_deeper_interval(region, constraints, intervals, recursion_depth, epsilon, coverage, silent,
                                  debug: bool = False, progress=False, timeout=0):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        coverage (float): coverage threshold to stop computation
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        progress (function or False): function(update_to, update_by) to update progress
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    if debug:
        silent = False

    ## TODO check consistency
    # print(region,prop,n,epsilon,coverage,silent)
    # print("check equal", globals()["non_white_area"],non_white_area)
    # print("check equal", globals()["whole_area"],whole_area)

    space = globals()["space"]

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return

    ## TODO
    # if presampled:
    #    while globals()["que"].size() > 0:
    #        private_check_deeper_interval(*globals()["que"].dequeue())
    #    return

    ## Stop if the given hyperrectangle is to small
    if get_rectangle_volume(region) < epsilon:
        if len(region) > 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"hyperrectangle {region} too small, skipped", "grey"))
            return f"hyperrectangle {region} too small, skipped"
        elif len(region) == 2:
            if not silent:
                print("depth:", recursion_depth, colored(f"rectangle {region} too small, skipped", "grey"))
            return f"rectangle {region} too small, skipped"
        else:
            if not silent:
                print("depth:", recursion_depth, colored(f"interval {region} too small, skipped", "grey"))
            return f"interval {region} too small, skipped"

    ## Stop if the the current coverage is above the given thresholds
    space_coverage = space.get_coverage()
    if space_coverage >= coverage:
        globals()["que"] = Queue()

        print(colored(f"coverage {space_coverage} is above the threshold", "blue"))
        return f"coverage {space_coverage} is above the threshold"

    ## Resolve the result
    # print("gonna check region: ", region)
    if check_interval_out(region, constraints, intervals, silent=silent, called=False, debug=debug) is True:
        result = "unsafe"
        if progress:
            ## Fixing overflow of progress when refinement continuous for "flat" refinement
            if globals()["flat_refinement"]:
                ## Proportion of que lenght and initial number of white rectangles
                progress(1 - globals()["que"].size() / globals()["numb_of_white_rectangles"])
            else:
                progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                            coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "red"))
            print()
    elif check_interval_in(region, constraints, intervals, silent=silent, called=False, debug=debug) is True:
        result = "safe"
        if progress:
            ## Fixing overflow of progress when refinement continuous for "flat" refinement
            if globals()["flat_refinement"]:
                ## Proportion of que lenght and initial number of white rectangles
                progress(1 - globals()["que"].size() / globals()["numb_of_white_rectangles"])
            else:
                progress(False, (2 ** (-(globals()["init_recursion_depth"] - recursion_depth))) / (
                            coverage - globals()["init_coverage"]))
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "green"))
            print()
    else:
        result = "unknown"
        if not silent:
            print("depth, hyper-rectangle, current_coverage, result")
            print(recursion_depth, region, colored(f"{space.get_coverage()} {result} \n", "grey"))
            print()

    if result == "safe" or result == "unsafe":
        # print("removing region:", region)
        space.remove_white(region)

    if result == "safe" or result == "unsafe":
        return

    ## Find maximum interval and split
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]
    foo = copy.deepcopy(region)
    foo[index] = (low, low + (high - low) / 2)
    foo2 = copy.deepcopy(region)
    foo2[index] = (low + (high - low) / 2, high)
    space.remove_white(region)
    space.add_white(foo)
    space.add_white(foo2)

    ## Halt if max recursion reached
    if recursion_depth == 0:
        return

    ## Halt if timeout
    if time() - globals()["start_time"] >= timeout > 0:
        if not silent:
            print(f"timeout reached here with coverage: {space.get_coverage()}")
        return

    ## Check if the que created (if alg1 used before it is not)
    try:
        type(globals()["que"])
    except KeyError:
        globals()["que"] = Queue()

    ## Add calls to the Queue
    # private_check_deeper_interval(region, constraints, intervals, recursion_depth, epsilon, coverage, silent, debug: bool = False, progress=False):
    globals()["que"].enqueue([copy.deepcopy(foo), constraints, intervals, recursion_depth - 1, epsilon, coverage,
                              silent, debug, progress, timeout])
    globals()["que"].enqueue([copy.deepcopy(foo2), constraints, intervals, recursion_depth - 1, epsilon, coverage,
                              silent, debug, progress, timeout])

    ## Execute the queue
    # print(globals()["que"].printQueue())
    while globals()["que"].size() > 0:
        private_check_deeper_interval(*globals()["que"].dequeue())
