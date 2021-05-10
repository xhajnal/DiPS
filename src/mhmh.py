import itertools
from copy import copy
from fractions import Fraction
from math import floor
from time import time
from termcolor import colored

## My imports
from common.convert import ineq_to_constraints, round_sig
from common.mathematics import create_intervals_hsb
from refine_space import check_deeper
from refine_space_parallel import check_deeper_parallel
from common.config import load_config
from common.space_stuff import get_rectangle_volume
from metropolis_hastings import init_mh, HastingsResults
from space import RefinedSpace

config = load_config()
# results_dir = spam["results"]
results_dir = config["results"]
refinement_results = config["refinement_results"]
refine_timeout = config["refine_timeout"]
z3_path = config["z3_path"]
tmp_dir = config["tmp"]
del config


def initialise_mhmh(params, parameter_domains, functions, constraints, data, sample_size, mh_sampling_iterations,
                    eps=0, sd=0.15, theta_init=False, is_probability=None, where_mh=False, where_ref=False,
                    progress=False, burn_in=False, bins=20, mh_timeout=False, debug=False, metadata=True,
                    draw_plot=False, save=False, silent=True, recursion_depth=10, epsilon=0.001, delta=0.001,
                    coverage=0.95, version=4, solver="z3", gui=False, parallel=True, ref_timeout=0, mh_result=None):
    """ Initialisation method for MHMH - space refinement with prior splitting based on MH

    Args:
        params (list of strings): parameter names
        parameter_domains (list of tuples): domains of parameters
        functions (list of strings): expressions to be evaluated and compared with data
        constraints  (list of strings): array of (in)equalities
        data (list of floats): measurement values
        sample_size (int): total number of observations in data
        mh_sampling_iterations (int): number of iterations/steps in searching in space
        eps (number): very small value used as probability of non-feasible values in prior - not used now
        sd (float): variation of walker in parameter space
        theta_init (list of floats): initial parameter point
        is_probability (bool): flag whether functions represent probabilities or not (None for unknown)
        where_mh (tuple/list or False): Metropolis-Hastings output matplotlib sources to output created figure, if False a new will be created
        where_ref (tuple/list or False): Refinement output matplotlib sources to output created figure, if False a new will be created
        progress (Tkinter element or False): function processing progress
        burn_in (number): fraction or count of how many samples will be trimmed from beginning
        bins (int): number of segments per dimension in the output plot
        mh_timeout (int): timeout of Metropolis-Hastings in seconds (0 for no timeout)
        debug (bool): if True extensive print will be used
        metadata (bool): if True metadata will be plotted
        draw_plot (Callable): function showing intermediate plots
        save (bool): if True output is stored
        silent (bool): if silent printed output is set to minimum
        recursion_depth (Int): max number of recursions to do
        epsilon (float): minimal size of rectangle to be checked
        delta (number):: used for delta solving using dreal
        coverage (float): coverage threshold to stop computation
        version (Int): version of the algorithm to be used for refinement
        solver (string):: specified solver, allowed: z3, dreal
        gui (bool or Callable): called from the graphical user interface
        parallel (int): number of threads to use in parallel for refinement, when True (#cores - 1) is used
        ref_timeout (int): timeout of refinement in seconds (0 for no timeout)
        mh_result (HastingsResults or None): (optional) list of accepted points - if not None, this list of accepted points will be used as result of MH
    """
    if mh_result is None:
        ## Run MH
        mh_result = init_mh(params, parameter_domains, functions, data, sample_size, mh_sampling_iterations, eps=eps,
                            sd=sd, theta_init=theta_init, is_probability=is_probability, where=where_mh, progress=progress,
                            burn_in=burn_in, bins=bins, timeout=mh_timeout, silent=silent, debug=debug, metadata=metadata,
                            draw_plot=draw_plot)
    if mh_result is False:
        raise Exception("MMHM cannot continue as Metropolis Hasting resulted in no accepted point, consider using more iterations.")

    assert isinstance(mh_result, HastingsResults)
    time_mhmh_took = mh_result.time_it_took
    transformation_started = time()

    ## Create bins
    intervals = []
    for interval in parameter_domains:
        add_this = []
        for index in range(bins):
            add_this.append([Fraction(index, bins-1)*(interval[1] - interval[0]) + interval[0], Fraction(index+1, bins-1)*(interval[1] - interval[0]) + interval[0]])
        intervals.append(add_this)
    del add_this

    if debug:
        print(intervals)

    ## My TEST INTERVALS
    # intervals = [[[0, 1], [1, 2]], [[3, 4], [4, 5]], [[6, 7], [7, 8], [8, 9]]]
    # intervals = [[[0, 1], [1, 2]], [[3, 4], [4, 5], [6, 7]]]

    ## Count the possible hyperrectangles
    picks = 1
    for item in intervals:
        picks = picks * len(item)
    # print("number of picks", picks)

    ## Initialise my_list - create a list of empty lists
    rectangularised_space = [[]]*picks

    ##
    number = 1
    for index, interval in enumerate(intervals):
        # print("number", number)
        for pick_index in range(picks):
            # print("   pick_index", pick_index)
            # print(" will append index", pick_index // number % len(interval), "of interval", interval, " : ", interval[pick_index // number % len(interval)])
            rectangularised_space[pick_index] = [*rectangularised_space[pick_index], list(interval[pick_index // number % len(interval)])]   ## TODO changed "tuple" to "list"
            # my_list[pick_index].append(interval[pick_index % number])
            # print("   my_list now", my_list)
        number = number * len(interval)

    if debug:
        print("Rectangularised Space based on MH bins", rectangularised_space)

    ## Convert rectangularised space into dictionary to count points in each hyperrectangle
    ## This is dictionary rectangle -> number of accepted points inside
    my_dictionary = {}
    for item in rectangularised_space:
        item = tuple(map(lambda x: tuple(x), item))  ## TODO swapped back to tuples to do dictionary stuff
        my_dictionary[tuple(item)] = 0

    ## Count points in each hyperrectangle
    for point in mh_result.get_all_accepted():
        # print(point)
        point = point[:-1]
        ## print(point)
        indices = []
        for index in range(len(point)):
            interval_length = Fraction(str(parameter_domains[index][1] - parameter_domains[index][0]))
            relative_position = (point[index] - parameter_domains[index][0]) / interval_length
            relative_interval = relative_position * (bins-1)

            indices.append((Fraction(interval_length, bins-1) * floor(relative_interval) + parameter_domains[index][0],
                            Fraction(interval_length, bins-1) * (floor(relative_interval)+1) + parameter_domains[index][0]))
        indices = tuple(indices)
        ## print("indices", indices)
        try:
            my_dictionary[indices] = my_dictionary[indices] + 1
        except KeyError as err:
            print(indices)
            raise err

    ## Compute size of bins:
    rect_size = get_rectangle_volume(rectangularised_space[0])

    ## Parse the bins
    if debug:
        print("Dictionary (hype)rectangle -> number of accepted points within the rectangle: \n", my_dictionary)

    ## Flip da dictionary: number of accepted points -> list of rectangles
    my_new_dictionary = {}
    for key, value in my_dictionary.items():
        if value in my_new_dictionary.keys():
            my_new_dictionary[value] = [*my_new_dictionary[value], list(map(lambda x: list(x), key))]
        else:
            my_new_dictionary[value] = [list(map(lambda x: list(x), key))]

    del my_dictionary
    if debug:
        print(my_new_dictionary)

    ## Compute expected number of points per bin
    expected_values = mh_sampling_iterations / picks
    print("We expect to see ", expected_values, "per (hype)rectangle") if not silent else None

    ## Initialise and add only rectangles to be refined
    space = RefinedSpace(parameter_domains, params)
    space.rectangles_unknown = {}
    del rectangularised_space

    ## Fill the Queue
    if debug:
        print(f"Choosing rectangles to refine:")
    picked = 0     ## number of picked rectangles
    sizes = sorted(copy(list(my_new_dictionary.keys())))  ## list of numbers of accepted points per rectangle
    while picked < 1/2*picks:  ## TODO choose a threshold of how many rectangles to refine
        ## Choose left or right side
        if abs(expected_values - sizes[0]) > abs(expected_values - sizes[-1]):
            items = my_new_dictionary[sizes[0]]
            if debug:
                print(f"Picking highest value {sizes[0]} of rectangle {items} over right value {sizes[-1]}")
            for item in items:
                item = [list(x) for x in item]
                space.add_white(item)
            picked = picked + len(items)
            del my_new_dictionary[sizes[0]]  ## remove picked size
            sizes = sizes[1:]  ## remove picked size

        else:
            items = my_new_dictionary[sizes[-1]]
            if debug:
                print(f"Picking highest value {sizes[-1]} of rectangles {items} over right value {sizes[0]}")
            for item in items:
                item = [list(x) for x in item]
                space.add_white(item)
            picked = picked + len(items)
            del my_new_dictionary[sizes[-1]]  ## remove picked size
            sizes = sizes[:-1]  ## remove picked size

    sizes = list(reversed(sizes))
    for index, size in enumerate(sizes):
        if size < 0.25*expected_values:
            items = my_new_dictionary[size]
            if debug:
                print(f"Picking a value below 1/4 of expected number accepted points {size} of rectangle {items}")
            for item in items:
                item = [list(x) for x in item]
                space.add_white(item)
            picked = picked + len(items)
            del my_new_dictionary[size]  ## remove picked size
            sizes = sizes[1:]  ## remove picked size
        else:
            sizes = list(reversed(sizes))
            break

    if debug:
        print()
        print("Selected rectangles to be refined:", space.get_white())
        print("we added", len(space.get_white()[rect_size]), "rectangles")
        print()

    ## Optimize memory
    # del sizes
    # del my_new_dictionary

    ## Compute time it took
    transformation_took = time() - transformation_started
    time_mhmh_took = time_mhmh_took + transformation_took

    ## Run first refinement
    ref_start_time = time()
    if parallel:
        check_deeper_parallel(space, constraints, recursion_depth=0, epsilon=0, coverage=1,
                              silent=silent, version=version, sample_size=False, debug=debug, save=save, title="",
                              where=None, show_space=False, solver=solver, delta=delta, gui=False, iterative=False,
                              parallel=parallel, timeout=ref_timeout)
    else:
        check_deeper(space, constraints, recursion_depth=0, epsilon=0, coverage=1,
                     silent=silent, version=version, sample_size=False, debug=debug, save=save, title="",
                     where=None, show_space=False, solver=solver, delta=delta, gui=False, iterative=False,
                     timeout=ref_timeout)

    ## Run second refinement
    ## TODO add the rest of rectangles
    rectangle_to_be_added = list(itertools.chain.from_iterable(my_new_dictionary.values()))
    for item in rectangle_to_be_added:
        space.add_white(item)

    if debug:
        print("we added", len(rectangle_to_be_added), "more rectangles")

    epsilon = max(epsilon, space.get_volume() / 2 ** (recursion_depth + 1))
    globals()['init_recursion_depth'] = recursion_depth
    if parallel:
        check_deeper_parallel(space, constraints, recursion_depth=recursion_depth, epsilon=epsilon, coverage=coverage,
                              silent=silent, version=version, sample_size=False, debug=debug, save=save, title="",
                              where=None, show_space=False, solver=solver, delta=delta, gui=False, iterative=False,
                              parallel=parallel, timeout=ref_timeout)
    else:
        check_deeper(space, constraints, recursion_depth=recursion_depth, epsilon=epsilon, coverage=coverage,
                     silent=silent, version=version, sample_size=False, debug=debug, save=save, title="",
                     where=None, show_space=False, solver=solver, delta=delta, gui=False, iterative=False,
                     timeout=ref_timeout)

    del sizes
    del my_new_dictionary

    ## Finish time business
    time_refinement_took = time() - ref_start_time
    time_mhmh_took = time_mhmh_took + time_refinement_took

    ## Refinement Visualisation
    print(colored(f"Refinement of MHMH using alg {version} with {solver} solver took {round(time_refinement_took, 2)} seconds", "yellow")) if not silent else None

    space.title = f"using max_recursion_depth:{globals()['init_recursion_depth']}, min_rec_size:{epsilon}, achieved_coverage:{space.get_coverage()}, alg{version}, {solver}"
    if where_ref is not None:
        space_shown = space.show(green=True, red=True, sat_samples=False, unsat_samples=False, save=save, where=where_ref,
                                 show_all=not gui, is_mhmh=True, is_parallel_refinement=parallel)
    print(colored(f"The whole MHMH took {round(time_mhmh_took, 2)} seconds", "yellow")) if not silent else None

    return space, mh_result


if __name__ == '__main__':
    ### GENERAL SETTING FOR TESTS
    depth = 10
    epsilon = 0
    coverage = 0.95
    show_matadata = False
    bins = 11

    silent = True
    debug = False
    gui = False
    version = 2  ## in [2,5]
    solver = "z3"  ## in ["z3", "dreal"] if version in [2,3,4] else irrelevant

    ## TEST 1 - comment following tests to run this one
    ## Trivial example of two different approaches: MHMH and standard refinement
    print(colored("Trivial example of two different approaches: MHMH and standard refinement", "blue"))
    params = ["p", "q"]
    parameter_intervals = [(0, 1), (0, 1)]
    f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
    constraints = ["p**2-2*p+1 > 0.1", "p**2-2*p+1 < 0.8", "2*q*p**2-2*p**2-2*q*p+2*p > 0.1", "2*q*p**2-2*p**2-2*q*p+2*p < 0.7", "(-2)*q*p**2+p**2+2*q*p > 0.3", "(-2)*q*p**2+p**2+2*q*p < 0.69"]

    ## TEST 2 - comment following tests to run this one
    data = [0.04, 0.02, 0.94]
    intervals = create_intervals_hsb(float(0.95), int(100), data)
    constraints = ineq_to_constraints(f, intervals, decoupled=True)

    timeout = 3600

    ## TEST 3
    constraints = ['((p+(-1))**2)/(1) >= 0.0', '((p+(-1))**2)/(1) <= 0.0834072934108425',
                   '(2*((p)*(p+(-1))*(q+(-1))))/(1) >= 0.0', '(2*((p)*(p+(-1))*(q+(-1))))/(1) <= 0.0524394957835608',
                   '(-1*((p)*(2*p*q+(-1)*p+(-2)*q)))/(1) >= 0.888453434338595',
                   '(-1*((p)*(2*p*q+(-1)*p+(-2)*q)))/(1) <= 0.991546565661405']
    print(constraints)

    ### RUN TEST
    ## MHMH sequential
    space, mh_result = initialise_mhmh(params, parameter_intervals, data=data, functions=f, sample_size=100,
                                       mh_sampling_iterations=10000, eps=0, silent=silent, debug=debug,
                                       bins=bins, is_probability=True, constraints=constraints,
                                       metadata=show_matadata, recursion_depth=depth, epsilon=epsilon,
                                       coverage=coverage, version=version, solver=solver, gui=False,
                                       where_ref=False, parallel=False, mh_timeout=timeout)
    print(f"Sequential MHMH took {round_sig(space.time_refinement, 4)} seconds")
    print()

    ## MHMH parallel
    space2, mh_result2 = initialise_mhmh(params, parameter_intervals, data=data, functions=f, sample_size=100,
                                         mh_sampling_iterations=10000, eps=0, silent=silent, debug=debug, bins=bins,
                                         is_probability=True, constraints=constraints, metadata=show_matadata,
                                         recursion_depth=depth, epsilon=epsilon, coverage=coverage, version=version,
                                         solver=solver, gui=False, where_ref=False, parallel=True, mh_timeout=timeout)
    print(f"Parallel MHMH took {round_sig(space2.time_refinement, 4)} seconds")
    print()

    ## Parallel refinement for comparison
    space3 = check_deeper_parallel(parameter_intervals, constraints, -1, epsilon, coverage, silent, version, sample_size=False,
                                   debug=False, save=False, title="", where=False, show_space=True, solver=solver, delta=0.001,
                                   gui=False, iterative=False, timeout=timeout)
    print(f"Parallel refinement took {round_sig(space3.time_refinement, 4)} seconds")
    print()

    ## Normal refinement for comparison
    space4 = check_deeper(parameter_intervals, constraints, -1, epsilon, coverage, silent, version, sample_size=False,
                          debug=False, save=False, title="", where=False, show_space=True, solver=solver, delta=0.001,
                          gui=False, iterative=False, timeout=timeout)

    print(f"Sequential refinement took {round_sig(space4.time_refinement, 4)} seconds")
    print()
