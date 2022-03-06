import sys
from time import time
import os
import socket
import numpy

## Importing my code
from termcolor import colored

from common.convert import parse_numbers
from mc import call_prism
from common.model_stuff import parse_params_from_model, find_param
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
        with open(f"data_n={N}.csv", "w") as file:
            file.write(f"n={N}, p_v={p_v}, q_v={q_v}\n")
            second_line = ""

            for polynomial in dic_fun[N]:
                parameters = set()
                if len(parameters) < N:
                    parameters.update(find_param(polynomial))
                parameters = sorted(list(parameters))
                parameter_value = [p_v, q_v]

                for param in range(len(parameters)):
                    a.append(parameter_value[param])
                    globals()[parameters[param]] = parameter_value[param]

                x = eval(polynomial)
                x = round(x, 2)
                second_line = f"{second_line}{x},"

            file.write(second_line[:-1])


def generate_experiments_and_data(model_types, n_samples, populations, dimension_sample_size, sim_length=False,
                                  modular_param_space=None, input_folder=False, output_folder=False, silent=False,
                                  debug=False):
    """ Generate experiment data for given settings

    Args:
        model_types (list of strings): list of model types
        n_samples (list of ints): list of sample sizes
        populations (list of ints): list of agent populations
        dimension_sample_size (int): number of samples of in each parameter dimension to be used
        sim_length (int): length of the simulation
        modular_param_space (numpy array): parameter space to be used
        input_folder (str or path): folder where to search for models
        output_folder (str or path): folder to dump PRISM output
        silent (bool): if silent printed output is set to minimum
        debug (bool): if True extensive print will be used
    """
    max_sample = max(n_samples)
    start_time = time.time()

    if output_folder is False:
        output_folder = tmp

    i = 1
    experiments = {}
    data = {}
    for model_type in model_types:
        if not silent:
            print("model_type: ", model_type)
        data[model_type] = {}
        experiments[model_type] = {}
        for population_size in populations:
            if not silent:
                print("population size: ", population_size)
            if "semisyn" in model_type and not sim_length:
                sim_length = 3 * population_size
            elif "syn" in model_type and not sim_length:
                sim_length = 2 * population_size
            elif "asyn" in model_type and not sim_length:
                sim_length = 4 * population_size

            if input_folder is not False:
                model_path = input_folder
            model = os.path.join(model_path, (model_type + "_" + str(population_size) + ".pm"))
            if not os.path.isfile(model):
                model = os.path.join(model_path, (str(population_size) + "_" + model_type + ".pm"))
            if not os.path.isfile(model):
                model = os.path.join(model_path, (model_type + "_" + str(population_size) + ".pm"))
            if not os.path.isfile(model):
                model = os.path.join(model_path, (str(population_size) + "_bees_" + model_type + ".pm"))
            if not os.path.isfile(model):
                model = os.path.join(model_path, (model_type + "_" + str(population_size) + "_bees.pm"))
            if not os.path.isfile(model):
                model = os.path.join(model_path, (str(population_size) + "bees_" + model_type + ".pm"))
            if not os.path.isfile(model):
                model = os.path.join(model_path, (model_type + "_" + str(population_size) + "bees.pm"))
            if not os.path.isfile(model):
                raise Exception("Model file not found", model)

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
                    path_file = os.path.join(output_folder,
                                             "dump_file_{}_{}_{}_{}.txt".format(model_type, population_size, max_sample,
                                                                                str(time.time()).replace(".", "")))
                    # print(path_file)

                    ## Here is the PRISM called
                    if not silent:
                        print(
                            f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
                    call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
                               silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))

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
                            filter(lambda x: x == i,
                                   experiments[model_type][population_size][n_sample][column_values]))) / n_sample)
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
                                         sim_length=sim_length, modular_param_space=modular_param_space, silent=silent,
                                         debug=debug)[0]


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
                                         sim_length=sim_length, modular_param_space=modular_param_space, silent=silent,
                                         debug=debug)[1]


if __name__ == '__main__':
    import numpy as np

    ### Honeybee
    use_case = "Honeybee"
    print(colored(f"Use_case {use_case}", "yellow"))
    model_types = ["semisynchronous"]
    populations = [2, 3, 4, 5, 10, 15]
    # populations = [3]
    n_samples = [3500]
    # n_samples = [5, 4]
    dimension_sample_size = 1
    silent = True
    debug = False

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # for model_type in model_types:
    #     ## MULTIPARAM
    #     param_space = np.array([[0.81], [0.92], [0.92], [0.92]])
    #     population = 4
    #     multiparam = "multiparam"
    #     Experiments_multiparam, Data_multiparam = generate_experiments_and_data([f"{multiparam}_{model_type}"], n_samples, [population], dimension_sample_size, input_folder=os.path.join(model_path, "old_bees"), modular_param_space=param_space, silent=silent)
    #     print(Data_multiparam)

    ### Knuth Die
    use_case = "Knuth Die"
    print(colored(f"Use_case {use_case}", "yellow"))
    ## Settings
    model = "Knuth/parametric_die.pm"
    prism_parameter_values = "p=0.1"
    n_samples = 1000
    silent = True
    sim_length = 100

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # path_file = "brokolica.txt"
    # values = {}
    # for i in range(n_samples):
    #     # print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
    #     call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
    #                silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
    #     with open(os.path.join(os.getcwd(), "results/prism_results", path_file)) as file:
    #         lines = file.readlines()
    #         # print(lines)
    #         last_line = lines[-1]
    #         # print()
    #         # print("last_line", last_line)
    #         value = parse_numbers(last_line)[-1]
    #         # print()
    #         # print("value", value)
    #         if value not in values.keys():
    #             values[value] = 1
    #         else:
    #             values[value] = values[value] + 1
    # print(values)
    # # Remove auxiliary simulation file
    # os.remove(os.path.join(os.getcwd(), "results/prism_results", path_file))
    data = [6, 11, 68, 8, 97, 810]
    data = [item / n_samples for item in data]
    print("Obtained data", data)

    ### Knuth Die with 3params, author Tatjana Petrov
    print(colored(f"Use_case {use_case}", "yellow"))
    ## Settings
    model = "Knuth/parametric_die_3_params.pm"
    prism_parameter_values = "p1=0.4,p2=0.7,p3=0.5"
    n_samples = 1000
    silent = True
    sim_length = 100

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # path_file = "zemiak.txt"
    # values = {}
    # for i in range(n_samples):
    #     # print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
    #     call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
    #                silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
    #     with open(os.path.join(os.getcwd(), "results/prism_results", path_file)) as file:
    #         lines = file.readlines()
    #         # print(lines)
    #         last_line = lines[-1]
    #         # print()
    #         # print("last_line", last_line)
    #         value = parse_numbers(last_line)[-1]
    #         # print()
    #         # print("value", value)
    #         if value not in values.keys():
    #             values[value] = 1
    #         else:
    #             values[value] = values[value] + 1
    # print(values)
    # # Remove auxiliary simulation file
    # os.remove(os.path.join(os.getcwd(), "results/prism_results", path_file))
    data = [208, 81, 100, 254, 261, 96]

    # Get expected value
    observations = []
    for index, item in enumerate(data):
        observations.extend([index + 1] * item)
    print("observations:", observations)
    print(len(observations), "observations")
    import scipy.stats as st

    a = st.t.interval(0.95, len(observations) - 1, loc=np.mean(observations), scale=st.sem(observations))
    print("mean", np.mean(observations))
    print("confidence intervals for expected roll", a)

    # data = [item/n_samples for item in data]
    print("Obtained data", data)

    ## 20 bees
    use_case = "20 bees"
    print(colored(f"Use_case {use_case}", "yellow"))
    model = "../20_synchronous.pm"
    prism_parameter_values = "r_00=0.1,r_01=0.103,r_02=0.106,r_03=0.109,r_04=0.112,r_05=0.115,r_06=0.118,r_07=0.121,r_08=0.124,r_09=0.127,r_10=0.13,r_11=0.133,r_12=0.136,r_13=0.139,r_14=0.142,r_15=0.145,r_16=0.148,r_17=0.151,r_18=0.154,r_19=0.157"
    n_samples = 3000
    silent = True
    sim_length = 1000

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # path_file = "skorica.txt"
    # values = {}
    # for i in range(n_samples):
    #     sys.stdout.write(f"{i+1},")
    #     # print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
    #     call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
    #                silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
    #     with open(os.path.join(os.getcwd(), "results/prism_results", path_file)) as file:
    #         lines = file.readlines()
    #         # for line in lines:
    #         #     print(line)
    #         last_line = lines[-1]
    #         # print()
    #         # print("last_line", last_line)
    #         line = last_line.split(" ")
    #         line = line[2:-1]
    #         value = sum(list(map(lambda x: int(x), line)))
    #         # print()
    #         # print("value", value)
    #         if value not in values.keys():
    #             values[value] = 1
    #         else:
    #             values[value] = values[value] + 1
    # print()
    # print(values)
    # # Remove auxiliary simulation file
    # os.remove(os.path.join(os.getcwd(), "results/prism_results", path_file))

    data = [363 / n_samples, 720 / n_samples, 827 / n_samples, 577 / n_samples, 304 / n_samples, 146 / n_samples,
            41 / n_samples, 16 / n_samples, 6 / n_samples, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ## TODO uncomment following lines to rerun
    import matplotlib.pyplot as plt
    import numpy as np

    x = range(21)
    y = data

    fig, ax = plt.subplots()
    # ax.yaxis.set_major_formatter(formatter)
    plt.xticks(x)
    plt.bar(x, y)
    # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))

    ## TODO uncomment following line to show the plot
    # plt.show()
    print("Obtained data", data)

    ### Zeroconf
    use_case = "Zeroconf"
    print(colored(f"Use_case {use_case}", "yellow"))
    ## Settings
    model = "zeroconf/zeroconf-10.pm"
    prism_parameter_values = "p=0.8,q=0.8"
    n_samples = 1000
    silent = True
    sim_length = 10000000

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # path_file = "krupica.txt"
    # values = {}
    # for i in range(n_samples):
    #     # print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
    #     call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
    #                silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
    #     with open(os.path.join(os.getcwd(), "results/prism_results", path_file)) as file:
    #         lines = file.readlines()
    #         # print(lines)
    #         last_line = lines[-1]
    #         # print()
    #         # print("last_line", last_line)
    #         value = parse_numbers(last_line)[-1]
    #         # print()
    #         # print("value", value)
    #         if value not in values.keys():
    #             values[value] = 1
    #         else:
    #             values[value] = values[value] + 1
    # print(values)
    # # Remove auxiliary simulation file
    # os.remove(os.path.join(os.getcwd(), "results/prism_results", path_file))
    data = [684, 316]
    data = [item / n_samples for item in data]
    print("Obtained data", data)

    ### Huy's Zeroconf
    use_case = "Huy's Zeroconf"

    print(colored(f"Use_case {use_case}", "yellow"))
    ## Settings
    model = "zeroconf/Huys/Huys_Zeroconf_4.pm"
    prism_parameter_values = "p=0.105547,q=0.449658"
    n_samples = 10000
    silent = True
    sim_length = 10000000

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # start_time = time()
    # path_file = "Huiho_krupica.txt"
    # values = {}
    # for i in range(n_samples):
    #     # print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
    #     call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
    #                silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
    #     with open(os.path.join(os.getcwd(), "results/prism_results", path_file)) as file:
    #         lines = file.readlines()
    #         # print(lines)
    #         last_line = lines[-1]
    #         # print()
    #         # print("last_line", last_line)
    #         value = parse_numbers(last_line)[-1]
    #         # print()
    #         # print("value", value)
    #         if value not in values.keys():
    #             values[value] = 1
    #         else:
    #             values[value] = values[value] + 1
    # print(f"It took {time() - start_time} seconds")
    # print(values)
    # # Remove auxiliary simulation file
    # os.remove(os.path.join(os.getcwd(), "results/prism_results", path_file))

    data = [0.0001, 0.9999]
    data = [item / n_samples for item in data]
    print("Obtained data", data)

    ### SIR
    use_case = "Huy's SIR"

    print(colored(f"Use_case {use_case}", "yellow"))
    ## Settings
    model = "SIR/sir_5_1_0.pm"
    prism_parameter_values = "alpha=0.034055,beta=0.087735"
    n_samples = 10000
    silent = True
    sim_length = 10000000

    ## TODO uncomment following lines to rerun
    # print("running", use_case)
    # start_time = time()
    # path_file = "Huiho_psenicaa.txt"
    # values = {}
    # for i in range(n_samples):
    #     # print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}")
    #     call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_length)} {path_file}",
    #                silent=silent, prism_output_path=os.path.join(os.getcwd(), "results/prism_results"))
    #     with open(os.path.join(os.getcwd(), "results/prism_results", path_file)) as file:
    #         lines = file.readlines()
    #         # print(lines)
    #         last_line = lines[-1]
    #         # print()
    #         # print("last_line", last_line)
    #         value = parse_numbers(last_line)[-1]
    #         # print()
    #         # print("value", value)
    #         if value not in values.keys():
    #             values[value] = 1
    #         else:
    #             values[value] = values[value] + 1
    # print(f"It took {time() - start_time} seconds")
    # print(values)
    # # Remove auxiliary simulation file
    # # os.remove(os.path.join(os.getcwd(), "results/prism_results", path_file))

    # data = [0.1098, 0.1377, 0.1296, 0.1312, 0.1466, 0.3451]  ## in order from 6 to 1 ## wrong old model
    exact_probs = [0.2746887629856223, 0.1292460318901584, 0.08158486526511617, 0.07315190727423651, 0.10128347308717214, 0.34004495949769376]
    data = [0.2721, 0.1316, 0.0871, 0.0719, 0.1021, 0.3352]  ## in order from 6 to 1
    data = [item / n_samples for item in data]
    print("Exact probabilities computed with PRISM", exact_probs)
    print("Obtained data", data)

    ### Honeybee
    use_case = "Honeybees extended"
    print(colored(f"Use_case {use_case}", "yellow"))
    model_types = ["semisynchronous"]
    populations = [2, 3, 4, 5, 10, 15]
    # populations = [3]
    n_samples = [3500, 1500, 100]
    # n_samples = [5, 4]
    dimension_sample_size = 1
    silent = True
    debug = False

    ## TODO uncomment following lines to rerun
    # for model_type in model_types:
    #     for multiparam in ["2-param", "multiparam"]:
    #         for population in populations:
    #             ## 2-PARAM
    #             if multiparam == "2-param":
    #                 ## Data sets
    #                 two_param_data_sets = [np.array([[0.81], [0.92]]), np.array([[0.53], [0.13]])]
    #                 for param_space in two_param_data_sets:
    #                     print(colored(f"model_type {model_type}, # of params {multiparam}, population size {population}, parameter space {param_space}", "blue"))
    #                     Experiments_two_param, Data_two_param = generate_experiments_and_data(model_types, n_samples, [population], dimension_sample_size, input_folder=os.path.join(model_path, model_type, multiparam), modular_param_space=param_space, silent=silent)
    #                     if debug:
    #                         print(Experiments_two_param)
    #                     print(Data_two_param)
    #
    #             ## MULTIPARAM
    #             elif multiparam == "multiparam":
    #                 ## Data set 1
    #                 if population == 2:
    #                     param_space = np.array([[0.19], [0.76]])
    #                 if population == 3:
    #                     param_space = np.array([[0.19], [0.76], [0.76]])
    #                 elif population == 4:
    #                     param_space = np.array([[0.19], [0], [0.76], [0.76]])
    #                 elif population == 5:
    #                     param_space = np.array([[0.19], [0], [0.76], [0.76], [0.76]])
    #                 elif population == 10:
    #                     param_space = np.array([[0.19], [0], [0], [0], [0], [0.76], [0.76], [0.76], [0.76], [0.76], [0.76]])
    #                 elif population == 15:
    #                     param_space = np.array([[0.19], [0], [0], [0], [0], [0], [0], [0.76], [0.76], [0.76], [0.76], [0.76], [0.76], [0.76], [0.76], [0.76]])
    #                 print(colored(f"model_type {model_type}, # of params {multiparam}, population size {population}, parameter space {param_space}", "blue"))
    #                 Experiments_multiparam, Data_multiparam = generate_experiments_and_data([f"{multiparam}_{model_type}"], n_samples, [population], dimension_sample_size, input_folder=os.path.join(model_path, model_type, multiparam), modular_param_space=param_space, silent=silent)
    #                 if debug:
    #                     print(Experiments_multiparam)
    #                 print(Data_multiparam)
    #
    #                 ## Data set 2
    #                 if population == 2:
    #                     param_space = np.array([[0.19], [0.86]])
    #                 if population == 3:
    #                     param_space = np.array([[0.19], [0.86], [0.86]])
    #                 elif population == 4:
    #                     param_space = np.array([[0.19], [0], [0.86], [0.86]])
    #                 elif population == 5:
    #                     param_space = np.array([[0.19], [0], [0.86], [0.86], [0.86]])
    #                 elif population == 10:
    #                     param_space = np.array([[0.19], [0], [0], [0], [0], [0.86], [0.86], [0.86], [0.86], [0.86], [0.86]])
    #                 elif population == 15:
    #                     param_space = np.array([[0.19], [0], [0], [0], [0], [0], [0], [0.86], [0.86], [0.86], [0.86], [0.86], [0.86], [0.86], [0.86], [0.86]])
    #                 print(colored(f"model_type {model_type}, # of params {multiparam}, population size {population}, parameter space {param_space}", "blue"))
    #                 Experiments_multiparam, Data_multiparam = generate_experiments_and_data([f"{multiparam}_{model_type}"], n_samples, [population], dimension_sample_size, input_folder=os.path.join(model_path, model_type, multiparam), modular_param_space=param_space, silent=silent)
    #                 if debug:
    #                     print(Experiments_multiparam)
    #                 print(Data_multiparam)

