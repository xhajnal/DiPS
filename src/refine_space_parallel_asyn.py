import _queue
import datetime
import socket
from platform import system
from time import strftime, localtime, time
from collections.abc import Iterable
import copy
from mpmath import mpi
from termcolor import colored
from z3 import Real, Int, Bool, BitVec, Solver, set_param, Z3Exception, unsat, unknown, Not, Or, sat

import multiprocessing
from multiprocessing import Process

## My imports
import refine_space
from common.convert import decouple_constraints, constraints_to_ineq
from common.solver_parser import pass_models_to_sons
from common.space_stuff import split_by_longest_dimension, split_by_samples
from common.model_stuff import find_param
from common.rectangle import My_Rectangle
from refine_space import private_presample
from sample_space import sample_region
from space import RefinedSpace


if "wind" not in system().lower():
    try:
        from dreal import logical_and, logical_or, logical_not, Variable, CheckSatisfiability
        gIsDrealAvailable = True
    except ImportError as err:
        print(f"Error while loading dreal {err}")
        gIsDrealAvailable = False

## TODO check whether I need fixing z3

## Try to run z3
try:
    p = Real('p')
except NameError:
    raise Exception("z3 not loaded properly")

## Global variables
global glob_parameters
global glob_constraints
global glob_functions
global glob_intervals
global glob_solver

global glob_delta

global glob_sample_size


def check_deeper_async(region, constraints, recursion_depth, epsilon, coverage, silent, version=4, sample_size=False,
                       sample_guided=False, debug=False, save=False, title="", where=False, show_space=True, solver="z3",
                       delta=0.001, gui=False, iterative=False, parallel=True, timeout=0, single_call_timeout=0):
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
        sample_guided (bool): flag to run sampling-guided refinement
        debug (bool): if True extensive print will be used
        save (bool): if True output is stored
        title (string):: text to be added in Figure titles
        where (tuple/list): output matplotlib sources to output created figure
        show_space (bool): if show_space the refined space will be visualised
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        gui (bool or Callable): called from the graphical user interface
        iterative (bool) : iterative approach, deprecated
        parallel (int): number of threads to use in parallel, when True (#cores - 1) is used
        timeout (int): timeout of refinement in seconds (set 0 for no timeout)
        single_call_timeout (int): timeout of a single SMT/interval arithmetics call in seconds (set 0 for no timeout)
    """
    global glob_parameters
    global glob_constraints
    global glob_intervals

    ## TODO infer max_max_recursion, when it crashes if maxrecursion is higher
    max_max_recursion = 12
    # if recursion_depth > max_max_recursion:
    #     recursion_depth = max_max_recursion

    ## INTERNAL SETTINGS AND VARIABLES
    times_it_timed_out = 0

    if parallel is True:
        pool_size = multiprocessing.cpu_count() - 2
    elif parallel > 1:
        pool_size = min(parallel, multiprocessing.cpu_count() - 2)
    elif parallel == 1:
        pool_size = 1

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
        epsilon = max(epsilon, space.get_volume() / 2 ** (recursion_depth + 1))

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
    refine_space.space = space
    refine_space.numb_of_white_rectangles = numb_of_white_rectangles
    if sample_size:
        private_presample(region, constraints, sample_size, where, show_space, gui, save, silent, debug)

    # NORMAL REFINEMENT - WITHOUT/AFTER PRESAMPLING
    ## Parsing version/algorithm
    ## Initialise the parameters
    if version <= 4:
        index = 0
        if solver == "z3":
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
        elif solver == "dreal":
            for param in space.params:
                globals()[param] = Variable(param)

    ## Converting constraints to inequalities
    elif version == 5:
        if isinstance(constraints[0], list):
            egg = constraints
        else:
            egg = constraints_to_ineq(constraints, silent=silent, debug=debug)
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

        ## Parallel version
        glob_parameters = parameters
        global glob_silent
        global glob_debug
        global glob_single_call_timeout
        glob_silent = silent
        glob_debug = debug
        glob_single_call_timeout = single_call_timeout

        # rectangles_to_be_refined = list(map(lambda x: x.region, space.rectangles_unknown[key]))

        rectangles_to_be_refined = space.get_flat_white()

        if version == 5:
            global glob_functions
            glob_functions = egg[0]
            glob_intervals = egg[1]
            glob_constraints = constraints
        else:
            glob_constraints = constraints
            global glob_solver
            glob_solver = solver
        if solver == "dreal":
            global glob_delta
            glob_delta = delta

        global glob_sample_size
        if sample_guided is True:
            glob_sample_size = 2  ## Selected by the best performance on hsb models
        else:
            glob_sample_size = sample_guided

        ## PARALLEL SAMPLING-GUIDED REFINEMENT
        ## TODO probably delete following two lines
        import warnings
        warnings.filterwarnings("ignore")
        # rectangles_to_be_refined = [rectangles_to_be_refined[0]]
        print(colored(f"rectangles_to_be_refined before, {rectangles_to_be_refined}", "cyan")) if not silent else None
        # print(pool_size)

        manager = multiprocessing.Manager()
        inq = manager.Queue()
        helpq = manager.Queue()
        outq = manager.Queue()

        for x in rectangles_to_be_refined:
            inq.put(x)
            helpq.put("t")
        # print("about to create workers")
        if version == 5:
            workers = [Worker(str(name), inq, helpq, outq, [glob_functions, glob_intervals], version, solver, delta, silent, debug, sample_guided) for name in range(parallel)]
        else:
            workers = [Worker(str(name), inq, helpq, outq, constraints, version, solver, delta, silent, debug, sample_guided) for name in range(parallel)]
        # print("about to create workers")
        for worker in workers:
            worker.start()
        # print("about to create workers")

        try:
            while True:
                # Waiting for workers to finish
                print(f"Help q size: {helpq.qsize()}") if debug else None
                print("Waiting for workers. Out queue size {}".format(outq.qsize())) if debug else None
                if space.get_coverage() > coverage:
                    for worker in workers:
                        worker.terminate()
                    helpq = manager.Queue()
                    raise Exception("halt")

                if helpq.qsize() == 0 and outq.qsize() == 0:
                    print(colored("BREAKING", "green"))
                    break

                # sleep(1)  ## TODO delete this line

                result = outq.get()

                space = parse_result(space, result, sample_guided, version, debug)
                # # print("RESULT", result)
                # white = result[0]
                # space.remove_white(white)
                # # print(f"region checked: {white}")
                # item = result[1]
                # # print(f"region checking result: {item}")
                # if item is True:
                #     space.add_green(white)
                # elif item is False:
                #     space.add_red(white)
                # else:
                #     for rectangle in item:
                #         # print("rectangle", rectangle)
                #         if version == 2 or version == 5:
                #             space.add_white(My_Rectangle(rectangle, is_white=True, model=(None, None)))
                #         else:
                #             space.add_white(My_Rectangle(rectangle[0], is_white=True, model=(rectangle[1][0], rectangle[1][1])))
                # print(colored(f"Current coverage is {space.get_coverage()}, current time is {datetime.datetime.now()}", "blue")) if debug else None
        except Exception as err:
            # print(str(err))
            try:
                while True:
                    result = outq.get(block=False)
                    space = parse_result(space, result, sample_guided, version, debug)
                    # # print("RESULT", result)
                    # white = result[0]
                    # space.remove_white(white)
                    # # print(f"region checked: {white}")
                    # item = result[1]
                    # # print(f"region checking result: {item}")
                    # if item is True:
                    #     space.add_green(white)
                    # elif item is False:
                    #     space.add_red(white)
                    # else:
                    #     for rectangle in item:
                    #         # print("rectangle", rectangle)
                    #         if version == 2 or version == 5:
                    #             space.add_white(My_Rectangle(rectangle, is_white=True, model=(None, None)))
                    #         else:
                    #             space.add_white(My_Rectangle(rectangle[0], is_white=True, model=(rectangle[1][0], rectangle[1][1])))
                    # print(colored(f"Current coverage is {space.get_coverage()}, current time is {datetime.datetime.now()}", "blue")) if debug else None
            except _queue.Empty:
                break

        # Clean up
        for worker in workers:
            worker.terminate()

        print(colored(f"Current coverage is {space.get_coverage()}, current time is {datetime.datetime.now()}", "blue")) if not silent else None
        print(space.nice_print()) if debug else None

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
                                     show_all=not gui, is_presampled=False, is_sampling_guided=sample_guided,
                                     is_parallel_refinement=parallel)
        else:
            space_shown = [False]
    else:  ## TODO THIS IS A HOTFIX
        if show_space:
            space_shown = space.show(green=True, red=True, sat_samples=gui and len(space.params) <= 2,
                                     unsat_samples=gui and len(space.params) <= 2, save=save, where=where,
                                     show_all=not gui, is_presampled=True, is_sampling_guided=sample_guided,
                                     is_parallel_refinement=parallel)
        else:
            space_shown = [False]

    if single_call_timeout:
        import sys
        sys.stdout.write(colored(f"\n This refinement timed out {times_it_timed_out} times ", "yellow"))
    if not silent:
        print(colored(f"Result coverage is: {space.get_coverage()}", "blue"))
        print(colored(f"Parallel refinement took {round(space.time_last_refinement, 4)} seconds", "yellow"))
        print(colored(f"Check calls took {round(space.time_check, 4)} seconds, {round(100 * space.time_check / space.time_last_refinement, 2)}% of refinement", "yellow"))
        print(colored(f"SMT calls took {round(space.time_smt, 4)} seconds, {round(100 * space.time_smt / space.time_check, 2) if space.time_check > 0 else None}% of checks, {round(100 * space.time_smt / space.time_last_refinement, 2)}% of refinement", "yellow"))
    if where:
        if space_shown[0] is None:
            return space, space_shown[1]
        else:
            return space
    else:
        return space


def parse_result(space, result, sample_guided, version, debug):
    # print("RESULT", result)
    white = result[0]
    space.remove_white(white)
    # print(f"region checked: {white}")
    item = result[1]
    # print(f"region checking result: {item}")
    if item is True:
        space.add_green(white)
        return space
    elif item is False:
        space.add_red(white)
        return space
    else:
        if sample_guided:
            # print("parsing result of sample-guided result")
            for rectangle in item:
                # print("rectangle", rectangle)
                if version == 2 or version == 5:
                    space.add_white(My_Rectangle(rectangle, is_white=True, model=(None, None)))
                else:
                    space.add_white(My_Rectangle(rectangle[0], is_white=True, model=(rectangle[1][0], rectangle[1][1])))
            print(colored(f"Current coverage is {space.get_coverage()}, current time is {datetime.datetime.now()}", "blue")) if debug else None
            return space
        else:
            # print("parsing result of not sample-guided result")
            if item[2] is not None:
                # print(colored(f"{item[0]}, {item[1]}", "red"))
                space.add_white(My_Rectangle(item[0], is_white=True, model=item[2]))
                space.add_white(My_Rectangle(item[1], is_white=True, model=item[3]))
            else:
                # print(colored(f"{item[0]}, {item[1]}", "red"))
                space.add_white(item[0])
                space.add_white(item[1])
            return space


def check_unsafe_parallel(region, constraints, silent=False, solver="z3", delta=0.001, debug=False, timeout=0):
    """ Check if the given region is unsafe or not using z3 or dreal.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of functions (rational functions in the case of Markov Chains)
        silent (bool): if silent printed output is set to minimum
        solver (string): specified solver, allowed: z3, dreal
        delta (number): used for delta solving using dreal
        debug (bool): if True extensive print will be used
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    global glob_parameters

    ## Initialisation
    if debug:
        silent = False

    print(f"Checking unsafe {region} using {('dreal', 'z3')[solver == 'z3']} solver, current time is {datetime.datetime.now()}") if not silent else None

    if solver == "z3":  ## avoiding collision name
        del delta
    elif solver == "dreal":
        if not gIsDrealAvailable:
            raise Exception("Dreal is not properly loaded, install it or use z3 instead, please.")

    ## Choosing solver
    if solver == "z3":
        set_param(max_lines=1, max_width=1000000)
        s = Solver()

        # if timeout > 0:
        #     print("z3 single call timeout set to", timeout)
        #     s.set("timeout", timeout)

        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        for param in glob_parameters:
            if debug:
                print(f"globals()[param] {globals()[param]}")
                print(f"region[{j}] {region[j]}")
            s.add(globals()[param] > region[j][0])
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

        # start_time = time()
        # timeout_decorator.timeout(timeout)
        check = s.check()
        # print(colored(f"This check unsafe took {time() - start_time} seconds", "yellow"))
        # if time()-start_time > timeout + 0.3:
        #     print(colored(f"This check unsafe took {time() - start_time} seconds, which is {time()-start_time-timeout} seconds longer than timeout {timeout}, while the region is {check}", "red"))

        ## If there is no example of satisfaction, hence all unsat, hence unsafe, hence red
        if check == unsat:
            print(f'The region {region} is {colored("unsafe", "red")}') if not silent else None
            return True
        elif check == unknown:
            return False
        ## Else there is an example for satisfaction, hence not all unsat
        else:
            print(f"Counterexample of unsafety: {s.model()}") if debug else None
            return s.model()

    elif solver == "dreal":
        if delta is None:
            delta = 0.001
        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        f_sat = None
        for param in glob_parameters:
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

        # if timeout > 0:
        #     timeout_decorator.timeout(timeout)
        result = CheckSatisfiability(f_sat, delta)

        if result is not None:
            print(f"Counterexample of unsafety: {result}") if debug else None
            return result
        else:
            print(f'The region {region} is {colored("unsafe", "red")}') if not silent else None
            return True


def check_safe_parallel(region, constraints, silent=False, solver="z3", delta=0.001, debug=False, timeout=0):
    """ Check if the given region is safe or not using z3 or dreal.

    It means whether for all parametrisations in **region** every property(prop) is evaluated within the given
    **interval**, otherwise it is not safe and counterexample is returned.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        silent (bool): if silent printed output is set to minimum
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        debug (bool): if True extensive print will be used
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    global glob_parameters
    ## Initialisation
    if debug:
        silent = False

    print(f"Checking safe {region} using {('dreal', 'z3')[solver == 'z3']} solver, current time is {datetime.datetime.now()}") if not silent else None

    if solver == "z3":  ## avoiding collision name
        del delta
    elif solver == "dreal":
        if not gIsDrealAvailable:
            raise Exception("Dreal is not properly loaded, install it or use z3 instead, please.")

    ## Choosing solver
    if solver == "z3":
        set_param(max_lines=1, max_width=1000000)
        ## Initialisation of z3 solver
        s = Solver()

        # if timeout > 0:
        #     print("single SMT call timeout", timeout)
        #     s.set("timeout", timeout)

        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        for param in glob_parameters:
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

        # start_time = time()

        # timeout_decorator.timeout(timeout)
        check = s.check()
        # if time()-start_time > timeout + 0.3:
        #     print(colored(f"This check safe took {time() - start_time} seconds, which is {time()-start_time-timeout} seconds longer than timeout {timeout}, while the region is {check}", "red"))
        #     print(formula)
        #     for param in glob_parameters:
        #         print(globals()[param])
        #         print(region[j][0])
        #         print(region[j][1])

        ## If there is an example of falsification
        if check == sat:
            print(f"Counterexample of safety: {s.model()}") if debug else None
            return s.model()
        elif check == unknown:
            return False
        ## Else there is no example of falsification, hence all sat, hence safe, hence green
        else:
            print(f"The region {region} is " + colored("is safe", "green")) if not silent else None
            return True

    elif solver == "dreal":
        if delta is None:
            delta = 0.001
        ## Adding regional restrictions to solver (hyperrectangle boundaries)
        j = 0
        f_sat = None
        for param in glob_parameters:
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

        # if timeout > 0:
        #     timeout_decorator.timeout(timeout)
        result = CheckSatisfiability(f_sat, delta)

        if result is None:
            print(f"The region {region} is " + colored("is safe", "green")) if not silent else None
            return True
        else:
            print(f"Counterexample of safety: {result}") if debug else None
            return result


def private_check_deeper_parallel(region, constraints=None, solver=None, delta=None, silent=None, debug=None, single_call_timeout=None):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        single_call_timeout (int): timeout of a single SMT/interval arithmetics call in seconds (set 0 for no timeout)
    """

    if constraints is None:
        constraints = glob_constraints
    if solver is None:
        solver = glob_solver
    if delta is None and solver == "dreal":
        delta = glob_delta
    if silent is None:
        silent = glob_silent
    if debug is None:
        debug = glob_debug
    if single_call_timeout is None:
        single_call_timeout = glob_single_call_timeout

    # print("region", region)

    # print("single_call_timeout", single_call_timeout) if not silent else None
    if check_unsafe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug, timeout=single_call_timeout) is True:
        return False
    elif check_safe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug, timeout=single_call_timeout) is True:
        return True

    ## Find index of maximum dimension to be split
    print(f'The region {region} is {colored("unknown", "yellow")}') if not silent else None
    rectangle_low, rectangle_high, index, threshold = split_by_longest_dimension(region)

    return rectangle_low, rectangle_high, None, None


def private_check_deeper_parallel_sampled(region, constraints=None, solver=None, delta=None, silent=None, debug=None, single_call_timeout=None):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        single_call_timeout (int): timeout of a single SMT/interval arithmetics call in seconds (set 0 for no timeout)
    """
    if constraints is None:
        constraints = glob_constraints
    if solver is None:
        solver = glob_solver
    if delta is None and solver == "dreal":
        delta = glob_delta
    if silent is None:
        silent = glob_silent
    if debug is None:
        debug = glob_debug

    print("region", region) if debug else None
    # print("single_call_timeout", single_call_timeout) if not silent else None
    # print("glob_sample_size", glob_sample_size)
    sat_list, unsat_list = sample_region(region, glob_parameters, constraints, sample_size=glob_sample_size, boundaries=region,
                                         compress=True, silent=True, save=False, debug=debug, progress=False,
                                         quantitative=False, parallel=False, save_memory=False, stop_on_unknown=True)
    if sat_list == [] or unsat_list == []:  ## all unsat or all sat
        if unsat_list:  ## some unsat samples
            print(f"Skipping check_safe {region} as some sat unsat samples found") if not silent else None
            if check_unsafe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
                return False

        if sat_list:  ## some sat samples
            print(f"Skipping check_unsafe {region} as some sat unsat samples found") if not silent else None
            if check_safe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
                return True
    else:
        return split_by_samples(region, sat_list, unsat_list, glob_sample_size, debug=debug)

    ## Find index of maximum dimension to be split
    rectangle_low, rectangle_high, index, threshold = split_by_longest_dimension(region)

    print("rectangle_low", rectangle_low) if debug else None
    print("rectangle_high", rectangle_high) if debug else None

    return [rectangle_low, rectangle_high]


def private_check_deeper_checking_parallel(region, constraints=None, model=None, solver=None, delta=None, silent=None, debug=None, single_call_timeout=None):
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
        single_call_timeout (int): timeout of a single SMT/interval arithmetics call in seconds (set 0 for no timeout)
    """
    if constraints is None:
        constraints = glob_constraints
    if solver is None:
        solver = glob_solver
    if delta is None and solver == "dreal":
        delta = glob_delta
    if silent is None:
        silent = glob_silent
    if debug is None:
        debug = glob_debug

    ## Resolving if the region safe/unsafe/unknown
    if model is None:
        example = check_unsafe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    elif model[0] is None:
        example = check_unsafe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    else:
        print("Skipping check_unsafe at", region, "since example", model[0]) if not silent else None
        example = model[0]

    ## Resolving the result
    if example is True:
        return False
    elif check_safe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug) is True:
        return True

    ## Find index of maximum dimension to be split
    rectangle_low, rectangle_high, index, threshold = split_by_longest_dimension(region)

    ## Compute passing example and counterexample to sons
    model_low, model_high = pass_models_to_sons(example, False, index, threshold, solver)

    return rectangle_low, rectangle_high, model_low, model_high


def private_check_deeper_checking_both_parallel(region, constraints=None, model=None, solver=None, delta=None, silent=None, debug=None, single_call_timeout=None):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints (list of strings): list of properties
        model (pair of example, counterexample): of the satisfaction in the given region
        solver (string):: specified solver, allowed: z3, dreal
        delta (number):: used for delta solving using dreal
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        single_call_timeout (int): timeout of a single SMT/interval arithmetics call in seconds (set 0 for no timeout)
    """
    if constraints is None:
        constraints = glob_constraints
    if solver is None:
        solver = glob_solver
    if delta is None and solver == "dreal":
        delta = glob_delta
    if silent is None:
        silent = glob_silent
    if debug is None:
        debug = glob_debug

    ## Resolving if the region safe/unsafe/unknown
    ## If both example and counterexample are None
    if model is None:
        example = check_unsafe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        # print("example", example)
        if example is True:
            counterexample = None
        else:
            counterexample = check_safe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    ## Elif the example is None
    elif model[0] is None:
        example = check_unsafe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug)
    else:
        print("Skipping check_unsafe at", region, "since example", model[0]) if not silent else None
        example = model[0]
    ## If counterexample is not None
    if model is not None:
        if model[1] is None:
            counterexample = check_safe_parallel(region, constraints, silent, solver=solver, delta=delta, debug=debug)
        else:
            print("Skipping check_safe at", region, "since counterexample", model[1]) if not silent else None
            counterexample = model[1]

    ## Resolving the result
    if example is True:
        return False
    elif counterexample is True:
        return True

    ## Find index of maximum dimension to be split
    rectangle_low, rectangle_high, index, threshold = split_by_longest_dimension(region)

    ## Compute passing example and counterexample to sons
    model_low, model_high = pass_models_to_sons(example, counterexample, index, threshold, solver)

    return rectangle_low, rectangle_high, model_low, model_high


def private_check_deeper_interval_parallel(region, functions=None, intervals=None, silent=None, debug=None):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        functions  (list of strings): array of properties
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    if functions is None:
        functions = glob_functions
    if intervals is None:
        intervals = glob_intervals
    if silent is None:
        silent = glob_silent
    if debug is None:
        debug = glob_debug

    ## Resolve the result
    if check_interval_out_parallel(region, functions, intervals, silent=silent, debug=debug) is True:
        return False
    elif check_interval_in_parallel(region, functions, intervals, silent=silent, debug=debug) is True:
        return True

    ## Find index of maximum dimension to be split
    rectangle_low, rectangle_high, index, threshold = split_by_longest_dimension(region)

    return rectangle_low, rectangle_high, None, None


def private_check_deeper_interval_parallel_sampled(region, functions=None, intervals=None, silent=None, debug=None):
    """ Refining the parameter space into safe and unsafe regions

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        functions  (list of strings): array of properties
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    if functions is None:
        functions = glob_functions
    constraints = glob_constraints
    if intervals is None:
        intervals = glob_intervals
    if silent is None:
        silent = glob_silent
    if debug is None:
        debug = glob_debug

    print("region", region) if debug else None
    # print("glob_sample_size", glob_sample_size)
    sat_list, unsat_list = sample_region(region, glob_parameters, constraints, sample_size=glob_sample_size,
                                         boundaries=region, compress=True, silent=True, save=False, debug=debug, progress=False,
                                         quantitative=False, parallel=False, save_memory=False, stop_on_unknown=True)
    if sat_list == [] or unsat_list == []:  ## all unsat or all sat
        # if unsat_list != []:  ## some un sat samples
        if unsat_list:
            if check_interval_out_parallel(region, functions, intervals, silent=silent, debug=debug) is True:
                return False
        if sat_list:
            if check_interval_in_parallel(region, functions, intervals, silent=silent, debug=debug) is True:
                return True
    else:
        return split_by_samples(region, sat_list, unsat_list, glob_sample_size, debug=debug)

    ## Find index of maximum dimension to be split
    rectangle_low, rectangle_high, index, threshold = split_by_longest_dimension(region)

    return [rectangle_low, rectangle_high]


def check_interval_in_parallel(region, constraints, intervals, silent=False, debug=False, timeout=0):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): array of properties
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    global glob_parameters
    if debug:
        silent = False

    print("Checking interval in", region, "current time is ", datetime.datetime.now()) if not silent else None

    ## Assign each parameter its interval
    for index, param in enumerate(glob_parameters):
        locals()[param] = mpi(region[index][0], region[index][1])

    ## Check that all prop are in its interval
    for index, prop in enumerate(constraints):
        ## TODO THIS CAN BE OPTIMISED
        try:
            interval = mpi(float(intervals[index].start), float(intervals[index].end))
        except AttributeError:
            interval = mpi(float(intervals[index][0]), float(intervals[index][1]))

        if not eval(prop) in interval:
            print(f"property {constraints.index(prop) + 1} ϵ {eval(prop)}, which is not in the interval {interval}") if debug else None
            return False
        else:
            print(f'property {constraints.index(prop) + 1} ϵ {eval(prop)} {colored(" is safe", "green")}') if debug else None

    return True


def check_interval_out_parallel(region, constraints, intervals, silent=False, debug=False, timeout=0):
    """ Check if the given region is unsafe or not.

    It means whether there exists a parametrisation in **region** every property(prop) is evaluated within the given
    **interval** (called a model in SMT), otherwise it is unsafe.

    Args:
        region (list of pairs of numbers): list of intervals, low and high bound, defining the parameter space to be refined
        constraints  (list of strings): of functions
        intervals (list of pairs/ sympy.Intervals): array of interval to constrain constraints
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        timeout (int): timeout in seconds (set 0 for no timeout)
    """
    global glob_parameters
    if debug:
        silent = False

    print("Checking interval_out", region, "current time is ", datetime.datetime.now()) if not silent else None

    ## Assign each parameter its interval
    for index, param in enumerate(glob_parameters):
        try:
            globals()[param] = mpi(region[index][0], region[index][1])
        except:
            print(f"Error occurred while region: {region}, with param {globals()[param]} of interval {mpi(region[index][0], region[index][1])}")

    ## Check that all prop are in its interval
    for index, prop in enumerate(constraints):
        # print("prop", prop)
        # print("eval prop", eval(prop))
        prop_eval = eval(prop)

        ## TODO THIS CAN BE OPTIMISED
        try:
            interval = mpi(float(intervals[index].start), float(intervals[index].end))
        except AttributeError:
            interval = mpi(float(intervals[index][0]), float(intervals[index][1]))

        ## If there exists an intersection (neither of these interval is greater in all points)
        try:
            if not (prop_eval > interval or prop_eval < interval):
                print(f"property {constraints.index(prop) + 1} ϵ {eval(prop)}, which is not outside of interval {interval}") if debug else None
            else:
                print(f'property {constraints.index(prop) + 1} ϵ {eval(prop)} {colored(" is unsafe", "red")}') if debug else None
                return True
        except TypeError as err:
            print(err)
            print(prop_eval)
            print(interval)
            raise err

    return False


class Worker(Process):
    """
    Simple worker.
    """
    def __init__(self, name, in_queue, help_queue, out_queue, constraints, version=2, solver="z3", delta=0.01, silent=False, debug=False, sample_guided=False):
        super(Worker, self).__init__()
        self.name = name
        self.in_queue = in_queue
        self.help_queue = help_queue
        self.out_queue = out_queue
        if version == 5:
            self.functions = constraints[0]
            self.intervals = constraints[1]
        else:
            self.constraints = constraints
        self.version = version
        self.solver = solver
        self.delta = delta
        self.silent = silent
        self.debug = debug
        self.sample_guided = sample_guided

    def run(self):
        while True:
            # Grab work
            work = self.in_queue.get()
            print("{} got {}".format(self.name, work)) if self.debug else None
            # region = copy.copy(work)

            if self.sample_guided:
                # Refine
                if self.version == 2:
                    print(f"Selecting biggest rectangles method with {('dreal', 'z3')[solver == 'z3']} solver") if not self.silent else None
                    spam = private_check_deeper_parallel_sampled(work, self.constraints, solver=self.solver, delta=self.delta, silent=self.silent, debug=self.debug)
                elif self.version == 3:
                    print(colored(f"Selecting biggest rectangles method with sampling and passing examples with {('dreal', 'z3')[solver == 'z3']} solver", "green")) if not self.silent else None
                    raise NotImplementedError("So far only alg 2 and 5 are implemented for parallel runs.")
                    spam = private_check_deeper_checking_parallel_sampled(work, self.constraints, solver=self.solver, delta=self.delta, silent=self.silent, debug=self.debug)
                elif self.version == 4:
                    print(colored(f"Selecting biggest rectangles method with sampling and passing examples and counterexamples with {('dreal', 'z3')[solver == 'z3']} solver", "green")) if not self.silent else None
                    raise NotImplementedError("So far only alg 2 and 5 are implemented for parallel runs.")
                    spam = private_check_deeper_checking_both_parallel_sampled(work, self.constraints, solver=self.solver, delta=self.delta, silent=self.silent, debug=self.debug)
                elif self.version == 5:
                    print(colored(f"Selecting biggest rectangles method with sampling and interval arithmetics", "green")) if not self.silent else None
                    spam = private_check_deeper_interval_parallel_sampled(work, self.functions, self.intervals, silent=self.silent, debug=self.debug)
            else:
                if self.version == 2:
                    print(f"Selecting biggest rectangles method with {('dreal', 'z3')[solver == 'z3']} solver") if not self.silent else None
                    spam = private_check_deeper_parallel(work, self.constraints, solver=self.solver, delta=self.delta, silent=self.silent, debug=self.debug)
                elif self.version == 3:
                    print(colored(f"Selecting biggest rectangles method with sampling and passing examples with {('dreal', 'z3')[solver == 'z3']} solver", "green")) if not self.silent else None
                    raise NotImplementedError("So far only alg 2 and 5 are implemented for parallel runs.")
                    spam = private_check_deeper_checking_parallel(work, self.constraints, solver=self.solver, delta=self.delta, silent=self.silent, debug=self.debug)
                elif self.version == 4:
                    print(colored(
                        f"Selecting biggest rectangles method with sampling and passing examples and counterexamples with {('dreal', 'z3')[solver == 'z3']} solver",
                        "green")) if not self.silent else None
                    raise NotImplementedError("So far only alg 2 and 5 are implemented for parallel runs.")
                    spam = private_check_deeper_checking_both_parallel(work, self.constraints, solver=self.solver, delta=self.delta, silent=self.silent, debug=self.debug)
                elif self.version == 5:
                    print(colored(f"Selecting biggest rectangles method with sampling and interval arithmetics",
                                  "green")) if not self.silent else None
                    spam = private_check_deeper_interval_parallel(work, self.functions, self.intervals, silent=self.silent, debug=self.debug)

            # Put result into out_queue
            print(f"refine result: {spam}") if self.debug else None
            self.out_queue.put([work, spam])

            ## Put results in help_queue
            if spam is True or spam is False:
                pass
            else:
                for rectangle in spam:
                    # self.out_queue.put(rectangle)
                    if rectangle is not None:
                        self.in_queue.put(rectangle)
                    self.help_queue.put("t")

            print("{} finished {}".format(self.name, work)) if self.debug else None
            ## Mark job done by decreasing size of help queue
            self.help_queue.get()
