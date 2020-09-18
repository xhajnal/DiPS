import os
import sys
from os.path import isfile
from pathlib import Path
from sympy import Interval

## Importing my code
from common.mathematics import margin
from common.config import load_config

spam = load_config()
model_folder = spam["models"]
properties_folder = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
del spam


def general_create_data_informed_properties(prop_file, intervals, output_file=False, silent: bool = False):
    """ Creates data informed property file from regular "profile" and intervals
    Args:
        prop_file (File/String):  regular prop file which contains lines in the form P=? (...)
        intervals (list of pairs of numbers): list of intervals to assign for the properties
        output_file (File/String): output prop file, if False or not given data_informed_properties as a list of strings is returned
        silent (bool): if silent command line output is set to minimum
    """
    if isinstance(prop_file, str):
        prop_file = Path(prop_file)
        if not isfile(prop_file):
            raise Exception(f"{prop_file} is not a file.")
    data_informed_properties = []
    i = 0
    with open(prop_file, "r") as file:
        for line in file:
            if line.startswith("P=?"):
                prefix = "P"
                line = line.split("P=?")[1]
            elif line.startswith("R") and "=?" in line:
                prefix = line.split("=?")[0]
                line = line.split("=?")[1]
            else:
                continue
            try:
                if isinstance(intervals[i], Interval):
                    data_informed_properties.append(f"{prefix}>{str(intervals[i].inf)} {line}")
                    data_informed_properties.append(f"{prefix}<{str(intervals[i].sup)} {line}")
                else:
                    data_informed_properties.append(f"{prefix}>{str(intervals[i][0])} {line}")
                    data_informed_properties.append(f"{prefix}<{str(intervals[i][1])} {line}")
                i = i + 1
            ## Checking sizes of properties and intervals
            except IndexError:
                print("data_informed_properties", data_informed_properties)
                raise Exception(f"Number of properties is larger than number of intervals {len(intervals)}")

    if not silent:
        print("data_informed_properties", data_informed_properties)
    ## Checking sizes of properties and intervals
    if len(intervals) is not i:
        raise Exception(f"Number of properties {i} is not corresponding to number of intervals {len(intervals)}")

    if not output_file:
        ## Getting rid of EOL
        data_informed_properties = list(map(lambda x: x[:-1], data_informed_properties))
        return data_informed_properties
    else:
        if isinstance(prop_file, str):
            output_file = Path(output_file)
            if not isfile(output_file):
                raise Exception(f"{output_file} is not a file.")
        with open(output_file, "w") as file:
            for line in data_informed_properties:
                file.write(line)


def create_data_informed_properties(population, data, alpha, n_samples, multiparam, seq):
    """ Creates property file of reaching each BSCC of the model of *N* agents as prop_<N>.pctl file.
    For more information see the HSB19 paper..
    
    Args:
        population (int): number of agents
        data: map of data
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        multiparam (bool): if True multiparam model is used
        seq (bool): if seq the property will be written for sequential usage
    """

    if multiparam:
        model = "_multiparam"
    else:
        model = ""

    if seq:
        conjunction = "\n"
        seq = "_seq"
    else:
        conjunction = " & "
        seq = ""

    with open(os.path.join(properties_folder, "prop{}_{}_{}_{}{}.pctl".format(model, population, alpha, n_samples, seq)), "w") as file:
        print(os.path.join(properties_folder, "prop{}_{}_{}_{}{}.pctl".format(model, population, alpha, n_samples, seq)))

        for i in range(len(data[population])):
            if data[population][i] - margin(alpha, n_samples, data[population][i]) > 0:
                if i > 0:
                    file.write("P>{} [ F (a0=1)".format(data[population][i] - margin(alpha, n_samples, data[population][i])))
                else:
                    # print(("P>{} [ F (a0=0)".format(data[N][i]-margin(alpha,n_samples,data[N][i]))))
                    # print(alpha,n_samples,data[N][i])
                    file.write("P>{} [ F (a0=0)".format(data[population][i] - margin(alpha, n_samples, data[population][i])))

                for j in range(1, population):
                    file.write("&(a" + str(j) + "=" + str(1 if j < i else 0) + ")")
                file.write("]{}".format(conjunction))
            if data[population][i] + margin(alpha, n_samples, data[population][i]) < 1:
                if i > 0:
                    file.write("P<{} [ F (a0=1)".format(data[population][i] + margin(alpha, n_samples, data[population][i])))
                else:
                    file.write("P<{} [ F (a0=0)".format(data[population][i] + margin(alpha, n_samples, data[population][i])))

                for j in range(1, population):
                    file.write("&(a" + str(j) + "=" + str(1 if j < i else 0) + ")")
                file.write("]{}".format(conjunction))
        if seq != "_seq":
            file.write(" true ")


def call_data_informed_prism(population, parameters, data, alpha, n_samples, multiparam, seq):
    """
    Creates data informed properties.
    
    Args:
        population (int): number of agents
        data: map of data
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        multiparam (bool): if True multiparam model is used
        parameters (string):  set of parameters
        seq (bool): if seq the property will be written for sequential usage
    """

    if multiparam:
        model = "multiparam_semisynchronous_parallel"
        prop = "_multiparam"
    else:
        model = "semisynchronous_parallel"
        prop = ""

    if seq:
        print("start=$SECONDS")
        j = 1

        for i in range(len(data[population])):
            # print(data[N][i], "margin: (", data[N][i]-margin(alpha, n_samples, data[N][i]),",",data[N][i]+margin(alpha, n_samples, data[N][i]),")")
            if data[population][i] - margin(alpha, n_samples, data[population][i]) > 0:
                sys.stdout.write('prism {}/{}_{}.pm '.format(model_folder, model, population))
                sys.stdout.write('{}/prop{}_{}_{}_{}_seq.pctl '.format(properties_folder, prop, population, alpha, n_samples))
                sys.stdout.write('-property {}'.format(j))
                j = j + 1
                sys.stdout.write(' -param "')
                sys.stdout.write('{}=0:1'.format(parameters[0]))
                for param in parameters[1:]:
                    sys.stdout.write(',{}=0:1'.format(param))
                sys.stdout.write('" >> {}/{}_{}_{}_{}_seq.txt 2>&1'.format(prism_results, model, population, alpha, n_samples))

                print()
                print()

            if data[population][i] + margin(alpha, n_samples, data[population][i]) < 1:
                sys.stdout.write('prism {}/{}_{}.pm '.format(model_folder, model, population))
                sys.stdout.write('{}/prop{}_{}_{}_{}_seq.pctl '.format(properties_folder, prop, population, alpha, n_samples))
                sys.stdout.write('-property {}'.format(j))
                j = j + 1
                sys.stdout.write(' -param "')
                sys.stdout.write('{}=0:1'.format(parameters[0]))
                for param in parameters[1:]:
                    sys.stdout.write(',{}=0:1'.format(param))
                sys.stdout.write('" >> {}/{}_{}_{}_{}_seq.txt 2>&1'.format(prism_results, model, population, alpha, n_samples))

                print()
                print()
            print("---")
            print()
        print("end=$SECONDS")
        print('echo "It took: $((end-start)) seconds." >> {}/{}_{}_{}_{}_seq.txt 2>&1'.format(prism_results, model, population,
                                                                                              alpha, n_samples))

    else:
        sys.stdout.write('(time prism {}/{}_{}.pm '.format(model_folder, model, population))
        sys.stdout.write('{}/prop{}_{}_{}_{}.pctl '.format(properties_folder, prop, population, alpha, n_samples))
        sys.stdout.write('-param {}=0:1'.format(parameters[0]))
        for param in parameters[1:]:
            sys.stdout.write(',{}=0:1'.format(param))
        sys.stdout.write(') > {}/{}_{}_{}_{}.txt 2>&1'.format(prism_results, model, population, alpha, n_samples))


def call_storm(population, parameters, data, alpha, n_samples, multiparam):
    """
    Returns command to call storm with given model and data informed properties
    
    Args:
        population (int): number of agents
        data (dict): map of data
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        multiparam (bool): if True multiparam model is used
        parameters (list of string): list of parameters
    """

    storm_models = model_folder.replace("\\\\", "/").replace("\\", "/").split("/")[-1]
    storm_output = storm_results.replace("\\\\", "/").replace("\\", "/").split("/")[-1]

    if multiparam:
        model = "multiparam_asynchronous"

    else:
        model = "asynchronous"

    print("start=$SECONDS")

    suffix = str(population)
    for i in range(len(data[population])):
        # print(data[N][i], "margin: (", data[N][i]-margin(alpha, n_samples, data[N][i]),",",data[N][i]+margin(alpha, n_samples, data[N][i]),")")

        if data[population][i] - margin(alpha, n_samples, data[population][i]) > 0:
            suffix = "{}-low".format(i)
            sys.stdout.write('./storm-pars --prism /{}/{}_{}.pm --prop "P>{}'.format(storm_models, model, population, data[population][i] - margin(alpha, n_samples, data[population][i])))
            if i > 0:
                sys.stdout.write("[ F (a0=1)")
            else:
                sys.stdout.write("[ F (a0=0)")

            for j in range(1, population):
                sys.stdout.write("&(a" + str(j) + "=" + str(1 if j < i else 0) + ")")
            sys.stdout.write(']"')
            sys.stdout.write(' --region "')
            sys.stdout.write('0.01<={}<=0.99'.format(parameters[0]))
            for param in parameters[1:]:
                sys.stdout.write(',0.01<={}<=0.99'.format(param))
            sys.stdout.write(
                '" --refine --printfullresult >> /{}/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(storm_output, model, population, alpha, n_samples, suffix))
            print()
            print()

        if data[population][i] + margin(alpha, n_samples, data[population][i]) < 1:
            suffix = "{}-high".format(i)
            sys.stdout.write('./storm-pars --prism /{}/{}_{}.pm --prop "P<{}'.format(storm_models, model, population, data[population][i] + margin(alpha, n_samples, data[population][i])))
            if i > 0:
                sys.stdout.write("[ F (a0=1)")
            else:
                sys.stdout.write("[ F (a0=0)")

            for j in range(1, population):
                sys.stdout.write("&(a" + str(j) + "=" + str(1 if j < i else 0) + ")")
            sys.stdout.write(']"')
            sys.stdout.write(' --region "')
            sys.stdout.write('0.01<={}<=0.99'.format(parameters[0]))
            for param in parameters[1:]:
                sys.stdout.write(',0.01<={}<=0.99'.format(param))
            sys.stdout.write(
                '" --refine --printfullresult >> /{}/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(storm_output, model, population, alpha, n_samples, suffix))

            print()
            print()
        print("---")
        print()
    print("end=$SECONDS")
    print('echo "It took: $((end-start)) seconds." >> /{}/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(storm_output, model, population, alpha, n_samples, suffix))
    print()
