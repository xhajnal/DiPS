import math
import sys
import os
from copy import copy
from datetime import datetime
from os.path import basename, isfile
from pathlib import Path

from sympy import factor
from time import time, sleep

from refine_space import check_deeper
from refine_space_parallel import check_deeper_parallel
from common.convert import ineq_to_constraints, round_sig
from common.files import pickle_load
from common.mathematics import create_intervals_hsb
from load import load_data, get_f
from common.model_stuff import parse_params_from_model
from mc import call_storm, call_prism_files
from mc_informed import general_create_data_informed_properties
from metropolis_hastings import init_mh, HastingsResults
from mhmh import initialise_mhmh
from refine_space_parallel_asyn import check_deeper_async
from space import RefinedSpace
from optimize import *
from common.config import load_config

## PATHS
spam = load_config()
data_dir = spam["data"]
model_path = spam["models"]
property_path = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
refinement_results = spam["refinement_results"]
tmp = spam["tmp"]

sys.setrecursionlimit(4000000)

## SET OF RUNS
run_prism_benchmark = False
run_semisyn = True
shallow = True  ## only a single setting will be chosen from a set within the benchmark

## SET METHODS
run_optimise = False
run_sampling = False
run_refine = True
run_mh = False
run_lifting = False
run_presampled_refine = False
run_mhmh = False

## GLOBAL SETTINGS
model_checker = "storm"  ## choose from "prism", "storm", "both"
precision = 4  ## number of decimal points to show
repetitions = 20  ## number of runs for each setting

global debug
globals()['debug'] = False
global silent
globals()['silent'] = True
global factorise
globals()['factorise'] = True

## SAMPLING SETTING
grid_size = spam["grid_size"]  # 25

## INTERVALS SETTINGS
C = spam["confidence_level"]  # 0.95

## REFINEMENT SETTINGS
max_depth = spam["max_depth"]  # -1
coverage = spam["coverage"]  # 0.95
epsilon = 0
refine_timeout = spam["refine_timeout"]  # 3600

## PRESAMPLED REFINEMENT SETTINGS
ref_sample_size = 21

## MH SETTINGS
mh_bins = 11  ## number of bins per dimension
iterations = 5000  # spam["mh_iterations"]  # 500000
mh_timeout = spam["mh_timeout"]  # 3600
mh_where = None  ## skip showing figures
mh_gui = False  ## skip showing progress
mh_metadata = False  ## skip showing metadata

## MHMH SETTINGS
mhmh_bins = 11  ## number of bins per dimension
mhmh_iterations = 1000
mhmh_timeout = 3600
mhmh_where = None  ## skip showing figures
mhmh_gui = False  ## skip showing progress
mhmh_metadata = False  ## skip showing metadata
mhmh_run_in_parallel = True

del spam


def load_functions(path, factorise=False, debug=False, source="any"):
    """ Loads rational functions from a file (path)

    Args
        path (Path or string): path to file
        factorise (bool): flag whether to factorise rational functions
        debug (bool): if True extensive print will be used
        source (string): which model checker result to load - "prism", "storm", if anything else, first we try to load storm, then prism is used
    """
    try:
        if source == "prism":
            raise FileNotFoundError()
        functions = pickle_load(os.path.join(storm_results, path))
        print(colored(f"Loaded function file: {path}.p using Storm file", "blue"))
    except FileNotFoundError:
        try:
            if source == "prism":
                raise FileNotFoundError()
            functions = get_f(os.path.join(storm_results, path + ".txt"), "storm", factorize=False)
            print(colored(f"Loaded function file: {path}.txt using Storm file", "blue"))
        except FileNotFoundError:
            try:
                if source == "storm":
                    raise FileNotFoundError()
                functions = pickle_load(os.path.join(prism_results, path))
                print(colored(f"Loaded function file: {path}.p using PRISM file", "blue"))
            except FileNotFoundError:
                if source == "storm":
                    raise FileNotFoundError()
                functions = get_f(os.path.join(prism_results, path + ".txt"), "prism", factorize=True)
                print(colored(f"Loaded function file: {path}.txt using PRISM file", "blue"))

    if factorise:
        print(colored(f"Factorising function file: {path}", "blue"))
        functions = list(map(factor, functions))
        functions = list(map(str, functions))

    if debug:
        # print("functions", functions)
        print("functions[0]", functions[0])
    else:
        print("functions[0]", functions[0])
    return functions


def compute_functions(model_file, property_file, output_path=False, parameter_domains=False, silent=False, debug=False):
    """ Runs parametric model checking of a model (model_file) and property (property_file).
        With given model parameters domains (parameter_domains)
        It saves output, rational functions, to a file (output_path).

    Args
        model_file (string or Path): model file path
        property_file (string or Path): property file path
        output_path (string or Path): output, rational functions, file path
        parameter_domains (list of pairs): list of intervals to be used for respective parameter (default all intervals are from 0 to 1)
        silent (bool): if True output will be printed
        debug (bool): if True extensive print will be used
    """
    constants, parameters = parse_params_from_model(model_file, silent=silent, debug=debug)
    if parameter_domains is False:
        parameter_domains = []
        for param in parameters:
            parameter_domains.append([0, 1])

    if output_path is False:
        ## Using default paths
        prism_output_path = prism_results
        storm_output_path = storm_results
    elif os.path.isabs(output_path):
        prism_output_path = output_path
        storm_output_path = output_path
    else:
        prism_output_path = os.path.join(prism_results, output_path)
        storm_output_path = os.path.join(storm_results, output_path)

    if not os.path.isdir(prism_output_path):
        if Path(prism_output_path).suffix != ".txt":
            prism_output_path = os.path.splitext(prism_output_path)[0] + ".txt"

    if not os.path.isdir(storm_output_path):
        if Path(storm_output_path).suffix != ".txt":
            storm_output_path = os.path.splitext(storm_output_path)[0] + ".txt"

    if model_checker == "prism" or model_checker == "both":
        call_prism_files(model_file, [], param_intervals=parameter_domains, seq=False, no_prob_checks=False,
                         memory="", model_path="", properties_path=property_path,
                         property_file=property_file, output_path=prism_output_path,
                         gui=False, silent=silent)
    if model_checker == "storm" or model_checker == "both":
        try:
            os.remove(storm_output_path)
        except Exception as err:
            print(err)
        call_storm(model_file=model_file, params=[], param_intervals=parameter_domains,
                   property_file=property_file, storm_output_file=storm_output_path,
                   time=True, silent=silent)


def repeat_sampling(space, constraints, sample_size, boundaries=None, silent=False, save=False, debug=False,
                    progress=False, quantitative=False, parallel=True, save_memory=False, repetitions=40, show_space=False):
    """ Runs space sampling


        Args:
            constraints  (list of strings): array of properties
            sample_size (int): number of samples in dimension
            boundaries (list of intervals): subspace to sample, default is whole space
            silent (bool): if silent printed output is set to minimum
            save (bool): if True output is pickled
            debug (bool): if True extensive print will be used
            progress (Tkinter element): progress bar
            quantitative (bool): if True return how far is the point from satisfying / not satisfying the constraints
            parallel (Bool): flag to run this in parallel mode
            save_memory (Bool): if True saves only sat samples
    """
    avrg_time = 0
    for run in range(repetitions):
        sampling = space.grid_sample(constraints, sample_size, boundaries=boundaries, silent=silent, save=save,
                                     debug=debug, progress=progress, quantitative=quantitative, parallel=parallel,
                                     save_memory=save_memory)
        if debug or show_space:
            space.show(sat_samples=True, unsat_samples=True)

        avrg_time += space.time_last_sampling
        sleep(1)

    avrg_time = avrg_time/repetitions
    if repetitions > 1:
        print(colored(f"Average time of {repetitions} runs is {round_sig(avrg_time)}", "yellow"))

    else:
        print(colored(f"Sampling took {round_sig(avrg_time)}", "yellow"))

    return avrg_time, sampling


def repeat_refine(text, parameters, parameter_domains, constraints, timeout=0, single_call_timeout=0, silent=True, debug=False, alg=4,
                  solver="z3", sample_size=False, sample_guided=False, repetitions=repetitions, where=None, parallel=True, is_async=False):
    """ Runs space refinement for multiple times

    Args
        text (string): text to be printed
        parameters (list of strings): list of model parameters
        parameter_domains (list of pairs): list of intervals to be used for respective parameter
        constraints  (list of strings): array of properties
        timeout (int): timeout in seconds (set 0 for no timeout)
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
        alg (Int): version of the algorithm to be used
        solver (string): specified solver, allowed: z3, dreal
        sample_size (Int or bool): number of samples in dimension used for presampling, False for no presampling
        sample_guided (bool): flag to run sampling-guided refinement
        repetitions (Int): number of runs per setting
        parallel (bool): flag whether to run in parallel mode
        is_async (bool): flag whether to run asynchronous calls when in parallel mode instead of map
    """
    print(colored(f"Refining, {text}", "green"))
    print(f"max_depth: {max_depth}, coverage: {coverage}, epsilon: {epsilon}, alg: {colored(alg, 'green')}, {'solver: ' + colored(solver, 'green') + ', ' if alg<5 else ''}{'with ' + colored(str(single_call_timeout), 'green')+ 's' if single_call_timeout>0 else 'without'} single SMT call timeout, current time is: {datetime.now()}")

    avrg_time, avrg_check_time, avrg_smt_time = 0, 0, 0

    if debug or where is False:
        show_space = True
        where = False
    else:
        show_space = False
        where = None

    for run in range(repetitions):
        space = RefinedSpace(parameter_domains, parameters)
        if parallel:
            try:
                if is_async:
                    spam = check_deeper_async(space, constraints, max_depth, epsilon=epsilon, coverage=coverage,
                                              silent=silent, version=alg, sample_size=sample_size, where=where if run == 0 else None,
                                              sample_guided=sample_guided,  debug=debug, save=False, solver=solver,
                                              delta=0.01, gui=False, show_space=show_space if run == 0 else False, iterative=False,
                                              parallel=parallel, timeout=timeout, single_call_timeout=single_call_timeout)
                else:
                    spam = check_deeper_parallel(space, constraints, max_depth, epsilon=epsilon, coverage=coverage,
                                                 silent=silent, version=alg, sample_size=sample_size, where=where if run == 0 else None,
                                                 sample_guided=sample_guided,  debug=debug, save=False, solver=solver,
                                                 delta=0.01, gui=False, show_space=show_space if run == 0 else False, iterative=False,
                                                 parallel=parallel, timeout=timeout, single_call_timeout=single_call_timeout)
                print("coverage reached", spam.get_coverage()) if not silent else None
            except NotImplementedError as err:
                print(colored("skipping this, not implemented", "blue"))
                print(err)
        else:
            if sample_guided:
                print(colored("Sampling-guided sequential refinement not implemented", "blue"))
                return
            if is_async:
                print(colored("Asynch sequential refinement not implemented", "blue"))
                return
            spam = check_deeper(space, constraints, max_depth, epsilon=epsilon, coverage=coverage,
                                silent=silent, version=alg, sample_size=sample_size, debug=debug, save=False, where=where if run == 0 else None,
                                solver=solver, delta=0.01, gui=False, show_space=show_space if run == 0 else False, iterative=False, timeout=timeout)
            print("coverage reached", spam.get_coverage()) if not silent else None
        if debug:
            print("refined space", spam.nice_print())

        sys.stdout.write(f"{run + 1}/{repetitions} ({round_sig(space.time_last_refinement)} s), ")

        if avrg_time/(run + 1) > timeout > 0:
            print()
            print(colored(f"Timeout reached, run number {run+1} with time {space.time_last_refinement}", "red"))
            avrg_time = 99999999999999999
            break

        avrg_time += space.time_last_refinement
        try:
            avrg_check_time += space.time_check
            avrg_smt_time += space.time_smt
        except:
            pass

    print()
    avrg_time = avrg_time/repetitions
    if repetitions > 1:
        print(colored(f"Average time of {repetitions} runs is {round_sig(avrg_time)}", "yellow"))

    else:
        print(colored(f"Refinement took {round_sig(avrg_time)}", "yellow"))

    try:
        avrg_check_time = avrg_check_time / repetitions
        avrg_smt_time = avrg_smt_time / repetitions
        print(colored(f"Average of check calls took {round_sig(avrg_check_time)} seconds, {round_sig(100 * avrg_check_time / avrg_time, 2)}% of refinement", "yellow"))
        print(colored(f"Average of SMT calls took {round_sig(avrg_smt_time, 8)} seconds, {round_sig(100 * avrg_smt_time / avrg_check_time, 2) if space.time_check > 0 else None}% of checks, {round_sig(100 * avrg_smt_time / avrg_time, 2)}% of refinement", "yellow"))
    except:
        pass

    return avrg_time, space


def repeat_mhmh(text, parameters, parameter_domains, data, functions, sample_size, mh_sampling_iterations, eps, silent,
                debug, bins, metadata, constraints, recursion_depth, epsilon, coverage, version, solver, gui, where,
                is_probability, repetitions=repetitions, parallel=mhmh_run_in_parallel, theta_init=False):

    ## TODO add info on what is running
    print(colored(f"{text}", "green"))
    print("max_depth", max_depth, "coverage", coverage, "epsilon", epsilon, "alg", colored(version, "green"), "solver", colored(solver, "green"), "current time is: ", datetime.now())

    avrg_whole_time, avrg_mh_time, avrg_refine_time = 0, 0, 0
    for run in range(repetitions):
        start_time = time()
        space, mh_result = initialise_mhmh(parameters, parameter_domains, functions=functions,
                                           constraints=constraints, data=data, sample_size=sample_size,
                                           mh_sampling_iterations=mh_sampling_iterations, theta_init=theta_init,
                                           eps=eps, is_probability=is_probability, where=where, bins=bins,
                                           mh_timeout=mhmh_timeout, silent=silent, debug=debug,
                                           metadata=metadata, recursion_depth=recursion_depth,
                                           epsilon=epsilon, coverage=coverage, version=version,
                                           solver=solver, gui=gui, parallel=parallel, ref_timeout=refine_timeout)
        end_time = time() - start_time

        sys.stdout.write(f"{run + 1}/{repetitions}, ({round_sig(end_time)} s of {round_sig(mh_result.time_it_took)} s MH, {round_sig(space.time_refinement)} s refine) ")

        if avrg_whole_time/(run + 1) > mhmh_timeout > 0:
            print()
            print(colored(f"Timeout reached,  run number {run+1} with time {end_time}, MH {mh_result.time_it_took}, refinement{space.time_refinement}", "red"))
            avrg_whole_time = 99999999999999999
            break

        avrg_mh_time += mh_result.time_it_took
        avrg_refine_time += space.time_refinement
        avrg_whole_time += end_time


    print()
    avrg_mh_time = avrg_mh_time / repetitions
    avrg_refine_time = avrg_refine_time / repetitions
    avrg_whole_time = avrg_whole_time/repetitions

    if repetitions > 1:
        print(colored(f"Average time of {repetitions} runs is {round_sig(avrg_whole_time)}, while MH took {round_sig(avrg_mh_time)} and refine {round_sig(avrg_refine_time)}", "yellow"))
        print(colored(f"{round_sig(avrg_mh_time)} / {round_sig(avrg_refine_time)} / {round_sig(avrg_whole_time)}", "yellow"))
    else:
        print(colored(f"MHMH took {round_sig(avrg_whole_time)}", "yellow"))

    try:
        return (avrg_whole_time, avrg_mh_time, avrg_refine_time), space, mh_result
    except:
        return avrg_whole_time


if __name__ == '__main__':
    ## PRISM BENCHMARKS
    test_cases = ["zeroconf"]  ## ["crowds", "brp", "nand", "Knuth", "zeroconf"]
    for test_case in test_cases:
        ## Skip PRISM BENCHMARKS?
        if not run_prism_benchmark:
            break
        if test_case == "crowds":
            if shallow:
                settings = [[3, 5]]
            else:
                settings = [[3, 5], [5, 5], [10, 5], [15, 5], [20, 5]]
            is_probability = False

        if test_case == "brp":
            if shallow:
                settings = [[16, 2]]
            else:
                settings = [[16, 2], [128, 2], [128, 5], [256, 2], [256, 5]]
            is_probability = False

        if test_case == "nand":
            if shallow:
                settings = [[10, 1]]
            else:
                settings = [[10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5]]
            is_probability = False

        if test_case == "Knuth":
            settings = ["", "_2_params", "_3_params"]  ## ["", "_2_params", "_3_params"]
            is_probability = False

        if test_case == "zeroconf":
            settings = [10]
            is_probability = False

        for constants in settings:
            if test_case == "Knuth":
                model_name = f"{test_case}/parametric_die{constants}"
                property_name = "BSCCs"
                property_file = os.path.join(property_path, test_case, f"parametric_die_{property_name}.pctl")
            elif test_case == "zeroconf":
                model_name = f"{test_case}/{test_case}-{constants}"
                property_name = "reach_BSCCs"
                property_file = os.path.join(property_path, test_case, f"{test_case}-{property_name}.pctl")
            else:
                model_name = f"{test_case}/{test_case}_{constants[0]}-{constants[1]}"
                property_file = os.path.join(property_path, test_case, f"{test_case}.pctl")
                property_name = ""
            model_file = os.path.join(model_path, f"{model_name}.pm")
            consts, parameters = parse_params_from_model(model_file, silent=True)

            if debug:
                print("parameters", parameters)

            ## SETUP PARAMETER AND THEIR DOMAINS
            parameter_domains = []
            for item in parameters:
                parameter_domains.append([0, 1])
            if debug:
                print("parameter_domains", parameter_domains)

            if property_name:
                function_file_name = f"{model_name}_{property_name}"
            else:
                function_file_name = f"{model_name}"

            ## LOAD FUNCTIONS
            if run_optimise or run_sampling or run_refine or run_mh or run_mhmh or run_presampled_refine:
                try:
                    functions = load_functions(function_file_name, factorise=True, debug=debug)
                except FileNotFoundError as err:
                    ## compute functions
                    try:
                        os.mkdir(os.path.join(prism_results, test_case))
                        os.mkdir(os.path.join(storm_results, test_case))
                    except Exception as err:
                        print(err)
                        pass
                    compute_functions(model_file, property_file, output_path=function_file_name, debug=debug)
                    functions = load_functions(function_file_name, factorise=True, debug=debug)
            else:
                functions = []
            assert isinstance(functions, list)

            ## COPY-PASTED GENERATED DATASETS
            if shallow:
                data_sets = [[0.5]]
            else:
                data_sets = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
            if test_case == "Knuth":
                if shallow:
                    data_sets = [[1 / 6] * 6, [0.208, 0.081, 0.1, 0.254, 0.261, 0.096]]  ## [[1 / 6] * 6, [0.208, 0.081, 0.1, 0.254, 0.261, 0.096]]
                else:
                    data_sets = [[1/6]*6, [0.006, 0.011, 0.068, 0.008, 0.097, 0.81], [0.208, 0.081, 0.1, 0.254, 0.261, 0.096]]  ## [1/6]*6, [0.006, 0.011, 0.068, 0.008, 0.097, 0.81], [0.208, 0.081, 0.1, 0.254, 0.261, 0.096]
            if test_case == "zeroconf":
                data_sets = [[0.684, 0.316]]

            for data_set in data_sets:
                if debug:
                    print("data_set", data_set)
                    print(type(data_set[0]))

                ## COMPUTE INTERVALS
                n_samples = [100, 1500, 3500]

                i = 0
                intervals = []
                for i in range(len(n_samples)):
                    intervals.append(create_intervals_hsb(float(C), int(n_samples[i]), data_set))
                    if debug:
                        print(f"Intervals, confidence level {C}, n_samples {n_samples[i]}: {intervals[i]}")

                ## OPTIMIZE PARAMS
                if run_optimise:
                    start_time = time()
                    result_1 = optimize(functions, parameters, parameter_domains, data_set)
                    print(colored(
                        f"Optimisation, data {data_set}, \nIt took {round_sig(time() - start_time, precision)} seconds", "yellow"))
                    if debug:
                        print(result_1)

                if run_sampling or run_refine or run_mhmh or run_presampled_refine:
                    constraints = []
                    for i in range(len(n_samples)):
                        constraints.append(ineq_to_constraints(functions, intervals[i], decoupled=True))
                        if debug:
                            print(f"Constraints with {n_samples[i]} samples :{constraints[i]}")
                else:
                    constraints = []

                print(constraints) if debug else None

                ## SAMPLE SPACE
                for i in range(len(n_samples)):
                    if not run_sampling:
                        break
                    print(colored(f"Sampling, dataset {data_set}, grid size {grid_size}, {n_samples[i]} samples"))
                    space = RefinedSpace(parameter_domains, parameters)
                    repeat_sampling(space, constraints[i], grid_size, silent=silent, save=False, debug=debug,
                                    quantitative=False, parallel=True, repetitions=40)

                ## REFINE SPACE
                for i in range(len(n_samples)):
                    if not run_refine:
                        break
                    text = f"dataset {data_set}, {n_samples[i]} samples"
                    if not shallow:
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=2, solver="z3")
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=2, solver="dreal")

                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=3, solver="z3")
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=3, solver="dreal")
                    
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=4, solver="z3")
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=4, solver="dreal")

                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=5)
                    
                ## PRESAMPLED REFINEMENT
                for i in range(len(n_samples)):
                    if not run_presampled_refine:
                        break
                    text = f"Presampled, dataset {data_set}, {n_samples[i]} samples"
                    if not shallow:
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=2, solver="z3", sample_size=ref_sample_size)
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=2, solver="dreal", sample_size=ref_sample_size)

                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=3, solver="z3", sample_size=ref_sample_size)
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=3, solver="dreal", sample_size=ref_sample_size)
                        
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=4, solver="z3", sample_size=ref_sample_size)
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=4, solver="dreal", sample_size=ref_sample_size)

                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent, debug, alg=5, sample_size=ref_sample_size)
                    
                ## METROPOLIS-HASTINGS
                # space = RefinedSpace(parameter_domains, parameters)
                ## TODO more indices of n_samples
                for i in range(len(n_samples)):
                    if not run_mh:
                        break
                    start_time = time()
                    mh_results = init_mh(parameters, parameter_domains, functions, data_set, n_samples[i], iterations, 0,
                                         silent=silent, debug=debug, is_probability=is_probability, where=True,
                                         metadata=False, timeout=mh_timeout)
                    assert isinstance(mh_results, HastingsResults)
                    print(colored(
                        f"this was MH, {iterations} iterations, dataset {data_set}, # of samples, {n_samples[i]} took {round_sig(time() - start_time, precision)} seconds", "yellow"))
                    if debug:
                        print("# of accepted points", len(mh_results.accepted))
                        if mh_results.last_iter > 0:
                            print("current iteration", mh_results.last_iter)
                            iter_time = (time() - start_time) * (iterations / mh_results.last_iter)
                            print("time it would take to finish", iter_time)
                        print()

                ## MHMH
                for i in range(len(n_samples)):
                    if not run_mhmh:
                        break
                    start_time = time()
                    text = f"{'Parallel ' if mhmh_run_in_parallel else''}MHMH, dataset {data_set}, {n_samples[i]} samples"
                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples[i], mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                                debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints[i],
                                recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=4, solver="z3",
                                gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions)

                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples[i], mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                                debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints[i],
                                recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=4, solver="dreal",
                                gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions)

                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples[i], mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent,
                                debug=debug, bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints[i],
                                recursion_depth=max_depth, epsilon=epsilon, coverage=coverage, version=5, solver=None,
                                gui=mhmh_gui, where=mhmh_where, is_probability=True, repetitions=repetitions)

                ## LIFTING WITH STORM HERE
                storm_parameter_domains = copy(parameter_domains)
                storm_parameter_domains = list(map(lambda x: (x[0] + 0.000001, x[1] - 0.000001), storm_parameter_domains))
                for i in range(len(n_samples)):
                    if not run_lifting:
                        break
                    spam = general_create_data_informed_properties(property_file, intervals[i], silent=False)
                    conf = str(C).replace(".", ",")
                    data_sett = ",".join(list(map(lambda x: str(x).replace(".", ","), data_set)))
                    save_informed_property_file = os.path.join(results_dir, "data-informed_properties", str(basename(property_file).split(".")[0]) + f"{n_samples[i]}_samples_{conf}_confidence" + ".pctl")
                    try:
                        os.mkdir(os.path.join(results_dir, "data-informed_properties"))
                    except FileExistsError as err:
                        pass

                    with open(save_informed_property_file, "w") as f:
                        for item in spam:
                            f.write(item + "\n")

                    try:
                        os.mkdir(os.path.join(refinement_results, test_case))
                    except FileExistsError as err:
                        pass

                    call_storm(model_file=model_file, params=parameters, param_intervals=storm_parameter_domains,
                               property_file=save_informed_property_file, storm_output_file=os.path.join(refinement_results, f"{model_name}_refined_{data_sett}_data_set_{n_samples[i]}_samples_{conf}_confidence.txt"),
                               time=True, silent=False)

    ## Semisynchronous models
    for multiparam in [False, True]:                                                ## [False, True]
        for population_size in [2, 3, 5, 10, 15]:                                ## [2, 3, 4, 5, 10, 15]
            for data_index, data_dir_subfolder in enumerate(["data", "data_1"]):    ## ["data", "data_1"]
                ## SKIP Semisynchronous models
                if not run_semisyn:
                    break
                if shallow and data_index > 0:
                    continue

                if multiparam:
                    params = "multiparam"
                    prefix = "multiparam_"
                else:
                    params = "2-param"
                    prefix = ""

                if population_size == 2 and multiparam is True:
                    continue

                if population_size == 4 and multiparam is False:
                    continue

                ## LOAD MODELS AND PROPERTIES
                try:
                    test_case = "bee"
                    # model_name = f"semisynchronous/{prefix}semisynchronous_{population_size}"
                    model_name = f"{prefix}semisynchronous_{population_size}_bees"
                    model_file = os.path.join(model_path, test_case, model_name + ".pm")
                    parameters = parse_params_from_model(model_file, silent=True)[1]
                except FileNotFoundError as err:
                    print(colored(f"model {model_name} not found, skipping this test", "red"))
                    if debug:
                        print(err)
                    break
                property_file = os.path.join(property_path, test_case, f"prop_{population_size}_bees.pctl")
                if not isfile(property_file):
                    with open(property_file, "r") as file:
                        pass

                ## SETUP PARAMETERS AND THEIR DOMAINS
                if debug:
                    print("parameters", parameters)
                parameter_domains = []
                for item in parameters:
                    parameter_domains.append([0, 1])
                if debug:
                    print("parameter_domains", parameter_domains)

                ## LOAD FUNCTIONS
                if run_optimise or run_sampling or run_refine or run_mh or run_mhmh or run_presampled_refine:
                    try:
                        functions = load_functions(f"{test_case}/{model_name}", debug=debug)
                    except FileNotFoundError as err:
                        ## Compute functions
                        compute_functions(model_file, property_file, output_path=f"{test_case}/{model_name}", debug=debug)
                        functions = load_functions(f"{test_case}/{model_name}", debug=debug)

                ## LOAD DATA
                data_set = load_data(os.path.join(data_dir, f"{data_dir_subfolder}/{params}/data_n={population_size}.csv"), debug=debug)
                if debug:
                    print("data_set", data_set)
                    print(type(data_set[0]))
                # data_set_2 = load_data(os.path.join(data_dir, f"data_1/data_n={population_size}.csv"))
                # print("data_set_2", data_set_2)

                ## COMPUTE INTERVALS
                if shallow:
                    n_samples = [100]
                else:
                    n_samples = [100, 1500, 3500]  ## [100, 1500, 3500]

                ## TODO more indices of n_samples
                i = 0
                intervals = []
                for i in range(len(n_samples)):
                    intervals.append(create_intervals_hsb(float(C), int(n_samples[i]), data_set))
                    if debug:
                        print(f"Intervals, confidence level {C}, n_samples {n_samples[i]}: {intervals[i]}")

                ## OPTIMIZE PARAMS
                if run_optimise:
                    start_time = time()
                    result_1 = optimize(functions, parameters, parameter_domains, data_set)
                    print(colored(f"Optimisation, {'multiparam' if bool(multiparam) else '2-param'}, {population_size} bees, dataset {data_index+1}", "green"))
                    print(colored(f"Optimisation took {round_sig(time() - start_time, precision)} seconds", "yellow"))
                    if debug:
                        print(result_1)

                if run_sampling or run_refine or run_mhmh or run_presampled_refine:
                    constraints = []
                    for i in range(len(n_samples)):
                        constraints.append(ineq_to_constraints(functions, intervals[i], decoupled=True))
                        if debug:
                            print(f"constraints with {n_samples[i]} samples :{constraints[i]}")

                    print(constraints) if debug else None

                ## SAMPLE SPACE
                for i in range(len(n_samples)):
                    if not run_sampling:
                        break
                    print(colored(f"Sampling, {'multiparam' if bool(multiparam) else '2-param'} , {population_size} bees, grid size {grid_size}, dataset {data_index+1}, {n_samples[i]} samples"))
                    space = RefinedSpace(parameter_domains, parameters)
                    repeat_sampling(space, constraints[i], grid_size, silent=silent, save=False, debug=debug,
                                    quantitative=False, parallel=True, repetitions=40)

                ## REFINE SPACE
                for i in range(len(n_samples)):
                    if not run_refine:
                        break
                    text = f"{'multiparam' if bool(multiparam) else '2-param'} , {population_size} bees, dataset {data_index + 1}, {n_samples[i]} samples "
                    if not shallow:
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="z3", alg=2)
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="dreal", alg=2)

                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="z3", alg=3)
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="dreal", alg=3)
                        
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="z3", alg=4)
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="dreal", alg=4)

                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, alg=5)

                ## METROPOLIS-HASTINGS
                # space = RefinedSpace(parameter_domains, parameters)
                ## TODO more indices of n_samples
                for i in range(len(n_samples)):
                    if not run_mh:
                        break
                    start_time = time()
                    mh_results = init_mh(parameters, parameter_domains, functions, data_set, n_samples[i], iterations, 0,
                                         silent=silent, debug=debug, is_probability=True, where=True, metadata=False,
                                         timeout=mh_timeout, parallel=3)
                    assert isinstance(mh_results, HastingsResults)
                    print(colored(f"this was MH, {iterations} iterations, {'multiparam' if bool(multiparam) else '2-param'} , {population_size} bees, dataset {data_index+1}, {n_samples[i]} samples took {round_sig(time() - start_time, precision)} seconds", "yellow"))
                    if debug:
                        print("# of accepted points", len(mh_results.accepted))
                        if mh_results.last_iter > 0:
                            print("current iteration", mh_results.last_iter)
                            iter_time = (time() - start_time) * (iterations / mh_results.last_iter)
                            print("time it would take to finish", iter_time)
                        print()

                ## PRESAMPLED REFINEMENT
                for i in range(len(n_samples)):
                    if not run_presampled_refine:
                        break
                    text = f"{'presampled multiparam' if bool(multiparam) else '2-param'} , {population_size} bees, dataset {data_index + 1}, {n_samples[i]} samples "
                    if not shallow:
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="z3", alg=2, sample_size=ref_sample_size)
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="dreal", alg=2, sample_size=ref_sample_size)

                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="z3", alg=3, sample_size=ref_sample_size)
                        repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="dreal", alg=3, sample_size=ref_sample_size)

                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="z3", alg=4, sample_size=ref_sample_size)
                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, solver="dreal", alg=4, sample_size=ref_sample_size)

                    repeat_refine(text, parameters, parameter_domains, constraints, refine_timeout, silent=silent, debug=debug, alg=5, sample_size=ref_sample_size)

                ## MHMH
                for i in range(len(n_samples)):
                    if not run_mhmh:
                        break
                    start_time = time()
                    text = f"{'Parallel ' if mhmh_run_in_parallel else''}MHMH, dataset {data_set}, {n_samples[i]} samples"

                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples[i], mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent, debug=debug,
                                bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints[i], recursion_depth=max_depth,
                                epsilon=epsilon, coverage=coverage, version=2, solver="z3", gui=mhmh_gui, where=mhmh_where,
                                is_probability=True, repetitions=repetitions, parallel=mhmh_run_in_parallel)

                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples[i], mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent, debug=debug,
                                bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints[i], recursion_depth=max_depth,
                                epsilon=epsilon, coverage=coverage, version=2, solver="dreal", gui=mhmh_gui, where=mhmh_where,
                                is_probability=True, repetitions=repetitions, parallel=mhmh_run_in_parallel)

                    repeat_mhmh(text, parameters, parameter_domains, data=data_set, functions=functions,
                                sample_size=n_samples[i], mh_sampling_iterations=mhmh_iterations, eps=0, silent=silent, debug=debug,
                                bins=mhmh_bins, metadata=mhmh_metadata, constraints=constraints[i], recursion_depth=max_depth,
                                epsilon=epsilon, coverage=coverage, version=5, solver=None, gui=mhmh_gui, where=mhmh_where,
                                is_probability=True, repetitions=repetitions, parallel=mhmh_run_in_parallel)
                    print()

