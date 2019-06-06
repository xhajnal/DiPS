import sys
import time
import os
import socket
import configparser

config = configparser.ConfigParser()
print(os.getcwd())

workspace = os.path.dirname(__file__)
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)

config.read(os.path.join(workspace, "../config.ini"))

prism_results = config.get("paths", "prism_results")
if not os.path.exists(prism_results):
    os.makedirs(prism_results)

sys.path.append(workspace)
from mc_prism import call_prism
from load import find_param

os.chdir(cwd)


def generate_all_data_twoparam(agents_quantities, dic_fun, p_v=None, q_v=None):
    """ Generates data for all agents_quantities in the current as .csv files

    Args
    ----
    agents_quantities: (list of ints) list of population sized to be used
    dic_fun: (dictionary) dictionary population size to list of rational functions
    p_v: (float/None) value of the first parameter - p
    q_v: (float/None) value of the second parameter - q
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
        file = open('data_n=' + str(N) + ".csv", "w")
        file.write('n=' + str(N) + ', p_v=' + str(p_v) + ', q_v=' + str(q_v) + "\n")
        secondline = ""

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
            secondline = secondline + str(x) + ","

        file.write(secondline[:-1])
        file.close()


def generate_experiments_and_data(model_types, multiparam, n_samples, populations, dimension_sample_size,
                                modular_param_space=None, silent=False):
    """Generate experiment data for given settings

    Args
    ------
    model_types: (list of strings) list of model types
    multiparam: (Bool) yes if multiparam should be used
    n_samples: (list of ints) list of sample sizes
    populations: (list of ints) list of agent populations
    dimension_sample_size: (list of ints) number of samples of in each paramter dimension to be used
    modular_param_space: (numpy array) parameter space to be used
    silent: (Bool): if silent printed output is set to minimum
    """
    max_sample = max(n_samples)
    start_time = time.time()

    i = 1
    Experiments = {}
    Data = {}
    for model_type in model_types:
        if multiparam:
            model_type = "multiparam_" + model_type
        if not silent:
            print("model_type: ", model_type)
        if "synchronous" in model_type:
            sim_lenght = 2
        Data[model_type] = {}
        Experiments[model_type] = {}
        for N in populations:
            if not silent:
                print("population size: ", N)
            if "semisynchronous" in model_type:
                sim_lenght = 2 * N
            if "asynchronous" in model_type:
                sim_lenght = 2 * N
            parameters = ["p"]
            if multiparam:
                for agents in range(1, N):
                    parameters.append("q" + str(agents))
            else:
                parameters.append("q")
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

            model = model_type + str(N) + ".pm"

            Experiments[model_type][N] = {}
            Data[model_type][N] = {}
            for n_sample in n_samples:
                Experiments[model_type][N][n_sample] = {}
                Data[model_type][N][n_sample] = {}

            # print(len(param_space[0]))
            for column in range(len(param_space[0])):
                column_values = []
                for value in param_space[:, column]:
                    column_values.append(value)
                column_values = tuple(column_values)
                if not silent:
                    print("parametrisation: ", column_values)
                for n_sample in n_samples:
                    Experiments[model_type][N][n_sample][column_values] = []
                    Data[model_type][N][n_sample][column_values] = []
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
                    path_file = "dump_file_{}_{}_{}_{}.txt".format(model_type, N, max_sample, str(time.time()).replace(".", ""))
                    # print(path_file)

                    ## Here is the PRISM called
                    if not silent:
                        print(f"calling: \n {model} -const {prism_parameter_values} -simpath {str(sim_lenght)} {path_file}")
                    call_prism(f"{model} -const {prism_parameter_values} -simpath {str(sim_lenght)} {path_file}",
                               silent=True, prism_output_path=cwd, std_output_path=None)
                   
                    ## Parse the dump file
                    # print("curr dir:", os.getcwd())
                    with open(path_file, "rt") as file:
                        last_line = file.readlines()[-1]

                    ## Append the experiment
                    state = sum(list(map(lambda x: int(x), last_line.split(" ")[2:-1])))

                    ## If some error occurred
                    if state > N or not silent or "2" in last_line.split(" ")[2:-1]:
                        print(last_line[:-1])
                        print("state: ", state)
                        print()
                    else:  ## If no error remove the file
                        os.remove(path_file)
                    for n_sample in n_samples:
                        if sample <= n_sample:
                            Experiments[model_type][N][n_sample][column_values].append(state)
                for n_sample in n_samples:
                    for i in range(N + 1):
                        Data[model_type][N][n_sample][column_values].append(len(list(
                            filter(lambda x: x == i, Experiments[model_type][N][n_sample][column_values]))) / n_sample)
                print("states: ", Experiments[model_type][N][max_sample][column_values])

    print(f"  It took {socket.gethostname()} {time.time() - start_time} seconds to run")
    return Experiments, Data


def generate_experiments(model_types, multiparam, n_samples, populations, dimension_sample_size,
                         modular_param_space=None):
    """Generate experiment data for given settings

    Args
    ------
    model_types: (list of strings) list of model types
    multiparam: (Bool) yes if multiparam should be used
    n_samples: (list of ints) list of sample sizes
    populations: (list of ints) list of agent populations
    dimension_sample_size: (list of ints) number of samples of in each paramter dimension to be used
    modular_param_space: (numpy array) parameter space to be used
    """
    return generate_experiments_and_data(model_types, multiparam, n_samples, populations, dimension_sample_size,
                                       modular_param_space)[0]


def generate_data(model_types, multiparam, n_samples, populations, dimension_sample_size, modular_param_space=None):
    """Generate experiment data for given settings

    Args
    ------
    model_types: (list of strings) list of model types
    multiparam: (Bool) yes if multiparam should be used
    n_samples: (list of ints) list of sample sizes
    populations: (list of ints) list of agent populations
    dimension_sample_size: (list of ints) number of samples of in each paramter dimension to be used
    modular_param_space: (numpy array) parameter space to be used
    """
    return generate_experiments_and_data(model_types, multiparam, n_samples, populations, dimension_sample_size,
                                       modular_param_space)[1]


if __name__ == "__main__":
    multiparam = True
    model_types = ["synchronous_parallel_"]
    n_samples = [3, 2]
    populations = [10]
    dimension_sample_size = 5
    Debug, Debug2 = generate_experiments_and_data(model_types, multiparam, n_samples, populations, 4, None, True)
    print(Debug)

    p_values = [0.028502714675268215, 0.45223461506339047, 0.8732745414252937, 0.6855555397734584, 0.13075717833714784]
    q_values = [0.5057623641293089, 0.29577906622244676, 0.8440550299528644, 0.8108008054929994, 0.03259111103419188]
    import numpy
    default_2dim_param_space = numpy.zeros((2, 5))
    default_2dim_param_space[0] = p_values
    default_2dim_param_space[1] = q_values

    Debug, Debug2 = generate_experiments_and_data(["synchronous_parallel_"], False, [3, 2], [2], 4, default_2dim_param_space, False)
    print(Debug["synchronous_parallel_"][2][3][(0.45223461506339047, 0.29577906622244676)])
    print(Debug2["synchronous_parallel_"][2][3][(0.45223461506339047, 0.29577906622244676)])

