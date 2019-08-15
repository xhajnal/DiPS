import configparser
import os
import sys

workspace = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(workspace)

sys.path.append(os.path.join(workspace, '../src/'))
from load import margin

config = configparser.ConfigParser()
# print(os.getcwd())
workspace = os.path.dirname(__file__)
# print("workspace",workspace)
config.read(os.path.join(workspace, "../config.ini"))
# os.chdir(config.get("paths", "properties"))

properties_folder = config.get("paths", "properties")
if not os.path.exists(properties_folder):
    raise OSError("Directory does not exist: " + str(properties_folder))

prism_results = config.get("paths", "prism_results")
if not os.path.exists(prism_results):
    os.makedirs(prism_results)

storm_results = config.get("paths", "storm_results")
if not os.path.exists(storm_results):
    os.makedirs(storm_results)

model_folder = config.get("paths", "models")
if not os.path.exists(model_folder):
    raise OSError("Directory does not exist: " + str(model_folder))

os.chdir(cwd)


def create_data_informed_properties(population, data, alpha, n_samples, multiparam, seq):
    """ Creates property file of reaching each BSCC of the model of *N* agents as prop_<N>.pctl file.
    For more information see paper.
    
    Args
    ----------
    population: (int) number of agents
    data: map of data    
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    multiparam: (Bool) if True multiparam model is used
    seq: (Bool) if seq the property will be written for sequential usage
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

    file = open(os.path.join(properties_folder, "prop{}_{}_{}_{}{}.pctl".format(model, population, alpha, n_samples, seq)), "w")
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
    if seq is not "_seq":
        file.write(" true ")
    file.close()


def call_data_informed_prism(population, parameters, data, alpha, n_samples, multiparam, seq):
    """
    Creates data informed properties.
    
    Args
    ----------
    population: (int) number of agents
    data: map of data    
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    multiparam: (Bool) if True multiparam model is used
    parameters: (string)  set of parameters
    seq: (Bool) if seq the property will be written for sequential usage
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
    
    Args
    ----------
    population: (int) number of agents
    data: map of data    
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    multiparam: (Bool) if True multiparam model is used
    parameters: (list of string) list of parameters
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
            sys.stdout.write('./storm-pars --prism /{}/{}_{}.pm --prop "P>{}'.format(storm_models, model, population,
                                                                                     data[population][i] - margin(alpha,
                                                                                                                  n_samples,
                                                                                                                  data[population][i])))
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
                '" --refine --printfullresult >> /{}/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(storm_output, model, population, alpha,
                                                                                         n_samples, suffix))

            print()
            print()

        if data[population][i] + margin(alpha, n_samples, data[population][i]) < 1:
            suffix = "{}-high".format(i)
            sys.stdout.write('./storm-pars --prism /{}/{}_{}.pm --prop "P<{}'.format(storm_models, model, population,
                                                                                     data[population][i] + margin(alpha,
                                                                                                                  n_samples,
                                                                                                                  data[population][i])))
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
                '" --refine --printfullresult >> /{}/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(storm_output, model, population, alpha,
                                                                                         n_samples, suffix))

            print()
            print()
        print("---")
        print()
    print("end=$SECONDS")
    print('echo "It took: $((end-start)) seconds." >> /{}/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(storm_output, model, population,
                                                                                              alpha, n_samples, suffix))
    print()
