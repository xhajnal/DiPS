import time
import os
import socket
import numpy

## Importing my code
from termcolor import colored

from mc import call_prism
from load import find_param, parse_params_from_model
from common.config import load_config

spam = load_config()
model_path = spam["models"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
tmp = spam["tmp"]
del spam


def generate_all_data_two_param(agents_quantities, dic_fun, p_v=None, q_v=None):
    """ Generates data for all agents_quantities in the current as .csv files

    Args:
        agents_quantities (list of ints): list of population sized to be used
        dic_fun (dict): dictionary population size to list of rational functions
        p_v (float/None): value of the first parameter - p
        q_v (float/None): value of the second parameter - q
    """
    # create distribution data, by Tanja, 14.1.2019
    # edited, by xhajnal, 18.1.2019, still name 'a' is not defined
    # edited, by xhajnal, 26.1.2019, a set as [] but does not work
    # totally rewritten to not paste empty line, doc added, by xhajnal, 03.02.2019
    import random

    if p_v is None:
        p_v = random.uniform(0., 1.)
        p_v = round(p_v, 2)
    if q_v is None:
        q_v = random.uniform(0., 1.)
        q_v = round(q_v, 2)
    a = []

    # create the files called "data_n=N.csv" where the first line is
    # "n=5, p_v--q1_v--q2_v--q3_v--q4_v in the multiparam. case, and, e.g.
    # 0.03 -- 0.45 -- 0.002 -- 0.3 (for N=5, there are 4 entries)

    for N in agents_quantities:
        with open('data_n=' + str(N) + ".csv", "w") as file:
            file.write('n=' + str(N) + ', p_v=' + str(p_v) + ', q_v=' + str(q_v) + "\n")
            second_line = ""

            for polynome in dic_fun[N]:
                parameters = set()
                if len(parameters) < N:
                    parameters.update(find_param(polynome))
                parameters = sorted(list(parameters))
                parameter_value = [p_v, q_v]

                for param in range(len(parameters)):
                    a.append(parameter_value[param])
                    globals()[parameters[param]] = parameter_value[param]

                x = eval(polynome)
                x = round(x, 2)
                second_line = second_line + str(x) + ","

            file.write(second_line[:-1])


def generate_experiments_and_data(model_types, n_samples, populations, dimension_sample_size,
                                  sim_length=False, modular_param_space=None, folder=False, silent=False, debug=False):
    """ Generate experiment data for given settings

    Args:
        model_types (list of strings): list of model types
        n_samples (list of ints): list of sample sizes
        populations (list of ints): list of agent populations
        dimension_sample_size (int): number of samples of in each parameter dimension to be used
        sim_length (int): length of the simulation
        modular_param_space (numpy array): parameter space to be used
        folder (str or path): folder where to search for models
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    max_sample = max(n_samples)
    start_time = time.time()

    i = 1
    experiments = {}
    data = {}
    for model_type in model_types:
        if not silent:
            print("model_type: ", model_type)
        if "synchronous" in model_type and not sim_length:
            sim_length = 2
        data[model_type] = {}
        experiments[model_type] = {}
        for population_size in populations:
            if not silent:
                print("population size: ", population_size)
            if "semisynchronous" in model_type and not sim_length:
                sim_length = 2 * population_size
            if "asynchronous" in model_type and not sim_length:
                sim_length = 2 * population_size
            if folder is False:
                model = os.path.join(model_path, (model_type + str(population_size) + ".pm"))
            else:
                model = os.path.join(folder, (model_type + str(population_size) + ".pm"))
            # A bad way how to deal with model without N
            if isinstance(population_size, str):
                population_size = 0
            if not silent:
                print("model: ", model)

            consts, parameters = parse_params_from_model(model, silent=silent)
            if not silent:
                print("parameters: ", parameters)

            ## Modulate parameter space
            if modular_param_space is not None:
                param_space = modular_param_space
            else:
                param_space = numpy.random.random((len(parameters), dimension_sample_size))

            if not silent:
                print("parameter space: ")
                print(param_space)

            experiments[model_type][population_size] = {}
            data[model_type][population_size] = {}
            for n_sample in n_samples:
                experiments[model_type][population_size][n_sample] = {}
                data[model_type][population_size][n_sample] = {}

            # print(len(param_space[0]))
            for column in range(len(param_space[0])):
                column_values = []
                for value in param_space[:, column]:
                    column_values.append(value)
                column_values = tuple(column_values)
                if not silent:
                    print("parametrisation: ", column_values)
                for n_sample in n_samples:
                    experiments[model_type][population_size][n_sample][column_values] = []
                    data[model_type][population_size][n_sample][column_values] = []
                # file = open("path_{}_{}_{}_{}_{}.txt".format(model_type,N,max_sample,v_p,v_q),"w+")
                # file.close()
                for sample in range(1, max_sample + 1):
                    ## Dummy path file for prism output
                    parameter_values = ""
                    prism_parameter_values = ""

                    for value in range(len(parameters)):
                        parameter_values = parameter_values + "_" + str(column_values[value])
                        prism_parameter_values = prism_parameter_values + str(parameters[value]) + "=" + str(
                            column_values[value]) + ","
                    prism_parameter_values = prism_parameter_values[:-1]
                    # print(parameter_values)
                    # print(prism_parameter_values)

                    ## More profound path name here
                    ## path_file = f"path_{model_type}{N}_{max_sample}_{parameter_values}.txt"
                    path_file = os.path.join(tmp, "dump_file_{}_{}_{}_{}.txt".format(model_type, population_size, max_sample, str(time.time()).replace(".", "")))
                    # print(path_file)

                    ## Here is the PRISM called
                    if not silent:
                        print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
                    call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
                               silent=True, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
                   
                    ## Parse the dump file
                    # print("curr dir:", os.getcwd())
                    with open(path_file, "rt") as file:
                        last_line = file.readlines()[-1]

                    ## Append the experiment
                    state = sum(list(map(lambda x: int(x), last_line.split(" ")[2:-1])))

                    ## If some error occurred
                    ## A bad way how to deal with files without N
                    if population_size != 0:
                        if state > population_size or debug or "2" in last_line.split(" ")[2:-1]:
                            print(last_line[:-1])
                            print("state: ", state)
                            print()
                        else:  ## If no error remove the file
                            os.remove(path_file)
                        for n_sample in n_samples:
                            if sample <= n_sample:
                                experiments[model_type][population_size][n_sample][column_values].append(state)
                for n_sample in n_samples:
                    ## A bad way how to deal with files without N
                    for i in range(population_size + 1):
                        data[model_type][population_size][n_sample][column_values].append(len(list(
                            filter(lambda x: x == i, experiments[model_type][population_size][n_sample][column_values]))) / n_sample)
                print("states: ", experiments[model_type][population_size][max_sample][column_values])

    print(colored(f"  It took {socket.gethostname()} {time.time() - start_time} seconds to run", "yellow"))
    return experiments, data


def generate_experiments(model_types, n_samples, populations, dimension_sample_size,
                         sim_length=False, modular_param_space=None, silent=False, debug=False):
    """ Generate experiment data for given settings

    Args:
        model_types (list of strings): list of model types
        n_samples (list of ints): list of sample sizes
        populations (list of ints): list of agent populations
        dimension_sample_size (int): number of samples of in each parameter dimension to be used
        sim_length (Int): length of the simulation
        modular_param_space (numpy array): parameter space to be used
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    return generate_experiments_and_data(model_types, n_samples, populations, dimension_sample_size,
                                         sim_length=sim_length, modular_param_space=modular_param_space, silent=silent, debug=debug)[0]


def generate_data(model_types, n_samples, populations, dimension_sample_size,
                  sim_length=False, modular_param_space=None, silent=False, debug=False):
    """ Generate experiment data for given settings

    Args:
        model_types (list of strings): list of model types
        n_samples (list of ints): list of sample sizes
        populations (list of ints): list of agent populations
        dimension_sample_size (int): number of samples of in each parameter dimension to be used
        sim_length (Int): length of the simulation
        modular_param_space (numpy array): parameter space to be used
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    return generate_experiments_and_data(model_types, n_samples, populations, dimension_sample_size,
                                         sim_length=sim_length, modular_param_space=modular_param_space, silent=silent, debug=debug)[1]
