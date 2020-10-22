import glob
import os
import re
import socket
import subprocess
from platform import system
from time import time, strftime, localtime
from pathlib import Path
import psutil
from termcolor import colored

## Importing my code
from load import parse_params_from_model
from common.files import write_to_file
from common.config import load_config

spam = load_config()
prism_path = spam["prism_path"]
model_path = spam["models"]
properties_path = spam["properties"]
results_dir = spam["results"]
prism_results = spam["prism_results"]
storm_results = spam["storm_results"]
del spam

if "prism" not in os.environ["PATH"]:
    print("prism was probably not in PATH, adding it there")
    if "wind" in system().lower():
        os.environ["PATH"] = os.environ["PATH"] + ";" + prism_path
    else:
        os.environ["PATH"] = os.environ["PATH"] + ":" + prism_path


def set_java_heap_win(size):
    """  Changing the java heap size for the PRISM on Windows

    Args:
        size (string): sets maximum memory, see https://www.prismmodelchecker.org/manual/ConfiguringPRISM/OtherOptions

    Returns:
        previous value of memory
    """
    previous_size = -5
    output = ""

    with open(os.path.join(str(prism_path), "prism.bat"), 'r') as input_file:
        # print(input_file)
        for line in input_file:
            output = output + str(line)
            if line.startswith('java'):
                # print("line", line)
                previous_size = re.findall(r'-Xmx.+[g|G|m|M] -X', line)
                # print("previous_size: ", previous_size)
                previous_size = previous_size[0][4:-3]
                # print("previous_size: ", previous_size)

    a = str(f'-Xmx{str(size)} -X')
    # print(a)
    output = re.sub(r'-Xmx.+[g|G|m|M] -X', a, output)
    # print("output: ", output)

    with open(os.path.join(str(prism_path), "prism.bat"), 'w') as input_file:
        input_file.write(output)

    if previous_size == -5:
        print("Error occurred while reading the prism.bat file")
        return

    return previous_size


## TODO rewrite this without the paths, just files
def call_prism(args, seq=False, silent: bool = False, model_path=model_path, properties_path=properties_path,
               prism_output_path=prism_results, std_output_path=prism_results, std_output_file=False):
    """  Solves problem of calling prism from another directory.

    Args:
        args (string): args for executing prism
        seq (bool): if true it will take properties one by one and append the results (helps to deal with memory)
        silent (bool): if silent command line output is set to minimum
        model_path (string): path to load  models from
        properties_path (string): path to load properties from
        prism_output_path (string): path to save the files inside the command
        std_output_path (string): path to save the results of the command
        std_output_file (string): file name to save the output
    """
    # print("prism_results", prism_results)
    # print("std_output_path", std_output_path)
    # print("std_output_file", std_output_file)

    if std_output_path is not None:
        output_file_path = Path(args.split()[0]).stem
        # print("output_file_path", output_file_path)
        if not std_output_file:
            # print("if")
            output_file_path = os.path.join(std_output_path, Path(output_file_path + ".txt"))
        else:
            # print("else")
            output_file_path = os.path.join(prism_results, Path(str(std_output_file)))
            # print("new output_file_path", output_file_path)
    else:
        output_file_path = ""

    # print("output_file_path", output_file_path)

    # os.chdir(config.get("mandatory_paths","cwd"))
    curr_dir = os.getcwd()
    os.chdir(prism_path)
    # print(os.getcwd())
    prism_args = []

    try:
        # print(args.split(" "))

        args = args.split(" ")
        # print(args)
        for arg in args:
            # print(arg)
            # print(re.compile('\.[a-z]').search(arg))
            if re.compile(r'\.pm').search(arg) is not None:
                model_file_path = os.path.join(model_path, arg)
                # print(model_file)
                if not os.path.isfile(model_file_path):
                    print(f"{colored('model file', 'red')} {model_file_path} {colored('not found -- skipped', 'red')}")
                    return 404, f"model file  {model_file_path} not found -- skipped"
                prism_args.append(model_file_path)
            elif re.compile(r'\.pctl').search(arg) is not None:
                property_file_path = os.path.join(properties_path, arg)
                # print(property_file)
                if not os.path.isfile(property_file_path):
                    print(f"{colored('property file', 'red')} {property_file_path} {colored('not found -- skipped', 'red')}")
                    return 404, f"property file {property_file_path} not found -- skipped"
                prism_args.append(property_file_path)
            elif re.compile(r'\.txt').search(arg) is not None:
                print("prism_output_path", prism_output_path)
                if not os.path.isabs(prism_output_path):
                    prism_output_path = os.path.join(Path(prism_results), Path(prism_output_path))

                if not os.path.isdir(prism_output_path):
                    if not silent:
                        print(f"{colored('The path', 'red')} {prism_output_path} {colored('not found, this may cause trouble', 'red')}")

                prism_output_file_path = os.path.join(prism_output_path, arg)
                print("prism_output_file_path", prism_output_file_path)
                prism_args.append(prism_output_file_path)
            else:
                prism_args.append(arg)
        # print(prism_args)
        # prism_args.append(" ".join(args.split(" ")[-2:]))
        # print(prism_args)

        # print(system().lower())
        if "wind" in system().lower():
            args = ["prism.bat"]
        else:
            args = ["prism"]
        args.extend(prism_args)

        ## forwarding error output to the file
        # args.append("2>&1")

        if seq:
            if os.path.isfile(os.path.join(std_output_path, output_file_path)):
                os.remove(os.path.join(std_output_path, output_file_path))
            with open(property_file_path, 'r') as property_file:
                args.append("-property")
                args.append("")
                property_file = property_file.readlines()
                for i in range(1, len(property_file) + 1):
                    args[-1] = str(i)
                    if not silent:
                        print("calling \"", " ".join(args), "\"")
                    output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")

                    ## Check for errors
                    if 'OutOfMemoryError' in output:
                        if "-javamaxmem" not in args:
                            memory = round(psutil.virtual_memory()[0] / 1024 / 1024 / 1024)  ## total memory converted to GB
                            print(colored(f"A memory error occurred while seq, max memory increased to {memory}GB", "red"))
                            args[-2] = "-javamaxmem"
                            args[-1] = f"{memory}g"
                            args.append("-property")
                            args.append(str(i))
                            if "wind" in system().lower():
                                previous_memory = set_java_heap_win(f"{memory}g")
                            if not silent:
                                print("calling \"", " ".join(args), "\"")
                            output = subprocess.run(args, stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT).stdout.decode("utf-8")
                            if 'OutOfMemoryError' in output:
                                write_to_file(output_file_path, output, silent, append=True)
                                print(colored(f"A memory error occurred while seq even after increasing the memory, close some programs and try again", "red"))
                                if "wind" in system().lower():
                                    set_java_heap_win(previous_memory)
                                return "memory_fail", "A memory error occurred while seq even after increasing the memory, close some programs and try again"
                        else:
                            write_to_file(output_file_path, output, silent, append=True)
                            print(colored(f"A memory error occurred while seq with given amount of memory", "red"))
                            ## Changing the memory setting back
                            if "wind" in system().lower():
                                set_java_heap_win(previous_memory)
                            return "memory", "A memory error occurred while seq with given amount of memory"

                    write_to_file(output_file_path, output, silent, append=True)

                    ## Check for errors
                    ## 'OutOfMemoryError', "Cannot allocate memory", 'Type error', 'Syntax error', 'NullPointerException', 'use -noprobchecks'
                    output = output.split("\n")
                    for item in output:
                        # print(item)
                        if 'use -noprobchecks' in item:
                            print(colored(f"Outgoing transitions checksum error occurred", "red"))
                            return "noprobchecks", item.strip()
                        if ("error" in item.lower()) or ("Cannot allocate memory" in item) or ('exception' in item.lower()):
                            spam = item.strip()
                            print(colored(spam, "red"))
                            return "error", spam

        else:
            if not silent:
                print("calling \"", " ".join(args))
            output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")
            write_to_file(output_file_path, output, silent, append=False)

            ## Check for errors
            ## 'OutOfMemoryError', "Cannot allocate memory", 'Type error', 'Syntax error', 'NullPointerException', 'use -noprobchecks'
            output = output.split("\n")
            for item in output:
                # print(item)
                if 'use -noprobchecks' in item:
                    print(colored(f"Outgoing transitions checksum error occurred", "red"))
                    return "noprobchecks", item.strip()
                if ("error" in item.lower()) or ("Cannot allocate memory" in item) or ('exception' in item.lower()):
                    spam = item.strip()
                    print(colored(spam, "red"))
                    return "error", spam
        return 0, ""
    finally:
        os.chdir(curr_dir)


## TODO rewrite this without the paths, just files
def call_prism_files(model_prefix, agents_quantities, param_intervals=False, seq=False, no_prob_checks=False, memory="",
                     model_path=model_path, properties_path=properties_path, property_file=False, output_path=prism_results, gui=False, silent=False):
    """  Calls prism for each file matching the prefix

    Args:
        model_prefix (string): file prefix to be matched
        agents_quantities (list of ints): pop_sizes to be used
        param_intervals (list of pairs or False): list of intervals to be used for respective parameter (default all intervals are from 0 to 1)
        seq (bool): if true it will take properties one by one and append the results (helps to deal with memory)
        no_prob_checks (bool or string): True if no noprobchecks option is to be used for prism
        model_path (string): path to load  models from
        properties_path (string): path to load properties from
        property_file (string): file name of single property files to be used for all models
        output_path (string): path for the output
        memory (string or int): sets maximum memory in GB, see https://www.prismmodelchecker.org/manual/ConfiguringPRISM/OtherOptions
        gui (function or False): callback function to be used
        silent (bool): if True the output is put to minimum
    """
    # print("model_path ", model_path)
    # print("model_prefix ", model_prefix)
    # os.chdir(config.get("mandatory_paths","cwd"))
    if no_prob_checks:
        no_prob_checks = '-noprobchecks '
    else:
        no_prob_checks = ""

    if memory == "":
        memory = ""
    elif "javamaxmem" not in str(memory):
        memory = f'-javamaxmem {memory}g '

    if not agents_quantities:
        # print("I was here")
        agents_quantities = [""]

    for N in sorted(agents_quantities):
        # print("glob.glob(os.path.join(model_path, model_prefix + str(N) + .pm))", glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm")))
        # print("glob.glob(os.path.join(model_path, model_prefix))", glob.glob(os.path.join(model_path, model_prefix)))
        # print("model_prefix", model_prefix)
        if "." in model_prefix:
            files = glob.glob(os.path.join(model_path, model_prefix))
        else:
            files = glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm"))
        if not silent:
            print("input files: ", files)
        if not files:
            print(colored("No model files for N="+str(N)+" found", "red"))
            if gui:
                gui(1, "Parameter synthesis", "No model files found.")
        for file in files:
            file = Path(file)
            start_time = time()
            # print("{} seq={}{} >> {}".format(file, seq, noprobchecks, str(prism_results)))

            ## Parsing the parameters from the files
            model_consts, model_parameters = parse_params_from_model(file, silent)
            params = ""
            i = 0
            for param in model_parameters:
                if param_intervals:
                    params = f"{params}{param}={param_intervals[i][0]}:{param_intervals[i][1]},"
                else:
                    params = f"{params}{param}=0:1,"
                i = i + 1
            ## Getting rid of the last ,
            if params:
                params = params[:-1]

            ## OLD parameters
            # if multiparam:
            #     params = ""
            #     for i in range(1, N):
            #         params = "{},q{}=0:1".format(q, i)
            #         # q=q+",q"+str(i)"=0:1"
            # else:
            #     params = ",q=0:1"
            # error = call_prism("{} prop_{}.pctl {}{}-param p=0:1{}".format(file, N, memory, noprobchecks, params),
            #                   seq=seq,
            #                   model_path=model_path, properties_path=properties_path, std_output_path=output_path)

            # print("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q))

            ## Calling the PRISM using our function

            if not property_file:
                error = call_prism("{} prop_{}.pctl {}{}-param {}".format(file, N, memory, no_prob_checks, params),
                                   seq=seq, model_path=model_path, properties_path=properties_path,
                                   std_output_path=output_path)
            else:
                # print("output_path", output_path)
                # print("file", file.stem)
                error = call_prism("{} {} {}{}-param {}".format(file, property_file, memory, no_prob_checks, params),
                                   seq=seq, model_path=model_path, properties_path=properties_path, std_output_path=output_path,
                                   std_output_file="{}_{}.txt".format(str(file.stem).split(".")[0], str(Path(property_file).stem).split(".")[0]),
                                   silent=silent)

            # print(f"  Return code is: {error}")
            if not silent:
                print(colored(f"  It took {socket.gethostname()}, {time() - start_time} seconds to run", "yellow"))

            ## Check for missing files
            if error[0] == 404:
                print(colored(error[1], "red"))
                if gui:
                    gui(2, "Parameter synthesis", error[1])
                continue

            ## Check if memory problem has occurred
            if error[0] == "memory":
                if not seq:
                    ## A memory occurred while not seq, trying seq now
                    seq = True
                    ## Remove the file because appending would no overwrite the file
                    os.remove(os.path.join(output_path, "{}.txt".format(file.stem)))
                    print(colored("A memory error occurred. Running prop by prob now", "red"))
                    if gui:
                        gui(3, "Parameter synthesis", "A memory error occurred. Running prop by prob now")
                else:
                    ## A memory occurred while seq
                    ## Remove the file because appending would not overwrite the file
                    os.remove(os.path.join(output_path, "{}.txt".format(file.stem)))
                    memory = round(psutil.virtual_memory()[0] / 1024 / 1024 / 1024)  ## total memory converted to GB
                    if "wind" in system().lower():
                        previous_memory = set_java_heap_win(f"{memory}g")
                    print(colored(f"A memory error occurred while seq, max memory increased to {memory}GB", "red"))
                    if gui:
                        gui(3, "Parameter synthesis", f"A memory error occurred while seq, max memory increased to {memory}GB")

            if error[0] == "memory_fail":
                ## An error occurred even when seq and max memory, no reason to continue
                if gui:
                    gui(1, "Parameter synthesis", f"An error occurred even when seq and max memory")
                break

            ## Check if there was problem with sum of probabilities
            if error[0] == "noprobchecks":
                if no_prob_checks == "":
                    print(colored("Outgoing transitions checksum error occurred. Running with noprobchecks option", "red"))
                    no_prob_checks = '-noprobchecks '
                    if gui:
                        gui(3, "Parameter synthesis", "Outgoing transitions checksum error occurred. Running with noprobchecks option")
                else:
                    print(colored("This is embarrassing, but Outgoing transitions checksum error occurred while noprobchecks option", "red"))
                    if gui:
                        gui(2, "Parameter synthesis", "This is embarrassing, but Outgoing transitions checksum error occurred while noprobchecks option")

            ## Check for other errors
            if error[0] == "error":
                ## Check for NullPointerException
                if "NullPointerException" in error[1]:
                    if seq:
                        # print(colored(error[1], "red"))
                        print(colored("Sorry, I do not know to to fix this, please try it manually", "red"))
                        print()
                        if gui:
                            gui(1, "Parameter synthesis", "Sorry, I do not know to to fix this, please try it manually")
                        break
                    else:
                        print(colored("Trying to fix the null pointer exception by running prop by prop", "red"))
                        if gui:
                            gui(3, "Parameter synthesis", "Trying to fix the null pointer exception by running prop by prop")
                        seq = True
                        ## Remove the file because appending would no overwrite the file
                        os.remove(os.path.join(output_path, "{}.txt".format(file.stem)))
                elif ('OutOfMemoryError' in error[1]) or ("Cannot allocate memory" in error[1]):
                    if not seq:
                        seq = True
                    else:
                        print(colored(f"A memory error occurred while seq, close some programs and try again with more memory.", "red"))
                        if gui:
                            gui(2, "Parameter synthesis", f"A memory error occurred while seq, close some programs and try again with more memory.")
                elif "Type error" in error[1]:
                    print(colored("A type error occurred, please check input files or manual", "red"))
                    if gui:
                        gui(2, "Parameter synthesis", "A type error occurred, please check input files or manual")
                    continue
                elif "Syntax error" in error[1]:
                    print(colored("A syntax error occurred, please check input files or manual.", "red"))
                    if gui:
                        gui(2, "Parameter synthesis", "A syntax error occurred, please check input files or manual.")
                    continue
                else:
                    print("Unrecognised error occurred:")
                    print(colored(error[1], "red"))
                    if gui:
                        gui(1, "Parameter synthesis", f"Unrecognised error occurred: \n {error[1]}")
                    continue

            if error[0] != 0:
                ## If an error occurred call this function for this file again
                print()
                # print("seq",seq)
                # print("noprobchecks", noprobchecks)
                call_prism_files(model_prefix, [N], seq=seq, no_prob_checks=no_prob_checks, memory=memory, model_path=model_path,
                                 properties_path=properties_path, property_file=property_file, output_path=prism_results)
            print()

    ## Setting the previous memory on windows
    if "wind" in system().lower():
        try:
            set_java_heap_win(previous_memory)
        except UnboundLocalError:
            pass


## TODO rewrite this without the paths, just files
def call_storm(args, silent: bool = False, model_path=model_path, properties_path=properties_path,
               output_folder=storm_results, storm_file_name="tmp.cmd", time=False):
    """  Prints calls for storm model checking.

    Args:
        args (string): args for executing storm
        silent (bool): if silent command line output is set to minimum
        model_path (string): path to load  models from
        properties_path (string): path to load properties from
        output_folder (string): folder to save results
        storm_file_name (string): file name to save the output
        time (bool): if True time measurement is added
    """

    print(colored(f"arguments {args}", "blue"))

    command_file_path = os.path.join(output_folder, storm_file_name)  ## Path to .cmd file
    if ".cmd" in storm_file_name:
        storm_file_path = os.path.join(output_folder, storm_file_name.replace(".cmd", ".txt"))  ## Path to .txt file
    else:
        storm_file_path = os.path.join(output_folder, f"{storm_file_name}.txt")  ## Path to .txt file

    print(f"{command_file_path} {colored('found', 'blue')}")

    storm_args = []
    args = args.split(" ")

    with open(command_file_path, "a+") as command_file_path:
        # print(args)
        for arg in args:
            # print(arg)
            # print(re.compile('\.[a-z]').search(arg))
            if re.compile(r'\.pm').search(arg) is not None:
                model_file_path = os.path.join(model_path, arg)
                # print(model_file)
                if not os.path.isfile(model_file_path):
                    command_file_path.write(f"file {model_file_path} not found -- skipped \n")
                    print(f"{colored('file', 'red')} {model_file_path} {colored('not found -- skipped', 'red')}")
                    return 404
                print(f"{model_file_path} {colored('found', 'blue')}")
                storm_args.append(f"/DiPS/{os.path.relpath(model_file_path, os.path.join(model_path,'..'))}")
            elif re.compile(r'\.pctl').search(arg) is not None:
                property_file_path = os.path.join(properties_path, arg)
                # print(property_file)
                if not os.path.isfile(property_file_path):
                    command_file_path.write(f"file {property_file_path} not found -- skipped \n")
                    print(f"{colored('file', 'red')} {property_file_path} {colored('not found -- skipped', 'red')}")
                    return 404
                # storm_args.append(property_file_path)
                storm_args.append("my_super_cool_string")
            elif re.compile(r'\.txt').search(arg) is not None:
                storm_file_path = os.path.join(properties_path, arg)
                if not os.path.isabs(storm_file_path):
                    storm_file_path = os.path.join(Path(storm_results), Path(storm_file_path))
                command_file_path.write(f"storm_output_path {storm_file_path} \n")
                print("storm_output_path", storm_file_path)
            else:
                storm_args.append(arg)

        args = ["./storm-pars --prism "]
        args.extend(storm_args)
        if time:
            args.append(")")
        if storm_file_path != "":
            args.append(">>")
            print(colored(storm_file_path, "blue"))
            args.append(f"/DiPS/{os.path.relpath(storm_file_path, os.path.join(model_path, '..'))}")
            args.append("2>&1 \n")

        if time:
            spam = "(time "
        else:
            spam = ""
        for arg in args:
            spam = spam + " " + arg
        if time:
            spam = spam + " "

        with open(property_file_path, "r") as file:
            content = file.readlines()
            for line in content:
                # print(colored(line, "blue"))
                line = line.replace('"', '\\"')

                output = spam.replace("my_super_cool_string", f"--prop \"{line[:-1]}\"")
                command_file_path.write(output)
                print(output)

    return True
    # output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")
    # write_to_file(std_output_path, output_file_path, output, silent, append=False)


## TODO rewrite this without the paths, just files
def call_storm_files(model_prefix, agents_quantities, param_intervals=False, model_path=model_path, properties_path=properties_path, property_file=False, command_output_file=False, output_path=storm_results, time=False):
    """  Calls storm for each file matching the prefix

    Args:
        model_prefix: file prefix to be matched
        agents_quantities (list of ints or False): pop_sizes to be used
        model_path (string): path to load  models from
        param_intervals (list of pairs): list of intervals to be used for respective parameter (default all intervals are from 0 to 1)
        properties_path (string): path to load properties from
        property_file (string): file name of single property files to be used for all models
        command_output_file (string): file to write the command
        output_path (string): path for the output
        time (bool): if True time measurement is added
    """
    ## print(colored(f"model prefixx {model_prefix}", "blue"))
    ## print(colored(f"model pathhhhh {model_path}", "blue"))

    root = output_path

    if not agents_quantities:
        agents_quantities = [""]

    if command_output_file:
        print(f"command file here: {command_output_file}")
    else:
        command_output_file = f"{os.path.join(output_path, str(strftime('%d-%b-%Y-%H-%M-%S', localtime())+'.cmd'))}"
        print(f"command file here: {command_output_file}")

    with open(command_output_file, "w") as output_filee:
        output_filee.write(f"cd {os.path.join(model_path, '..')} \n")
        print(f"cd {os.path.join(model_path, '..')}")

        # output_filee.write("sudo docker pull movesrwth/storm:travis \n")
        # print("sudo docker pull movesrwth/storm:travis")
        output_filee.write(f'sudo docker run --mount type=bind,source="$(pwd)",target=/DiPS -w /opt/storm/build/bin --rm -it --name storm movesrwth/storm:travis \n')
        print(f'sudo docker run --mount type=bind,source="$(pwd)",target=/DiPS -w /opt/storm/build/bin --rm -it --name storm movesrwth/storm:travis')

    # print("model_path", model_path)
    # print("model_prefix", model_prefix)
    for N in sorted(agents_quantities):
        if "." in model_prefix:
            files = glob.glob(os.path.join(model_path, model_prefix))
        else:
            files = glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm"))
        # print("files", files)
        if not files:
            with open(command_output_file, "w") as output_filee:
                output_filee.write("No model files for N="+str(N)+" found")
                print(colored("No model files for N="+str(N)+" found", "red"))
            continue
        for file in files:
            file = Path(file)
            # print("{} {}".format(file, property_file))
            # call_storm("{} {}".format(file, property_file), model_path=model_path, properties_path=properties_path, std_output_path=output_path, std_output_file="{}_{}.txt".format(str(file.stem).split(".")[0], property_file.split(".")[0]), time=time)

            if property_file:
                # print("property_file", property_file)
                # print("file", file)
                # print("file stem", file.resolve().stem)
                # print("{}_{}.txt".format(str(file.stem).split(".")[0], property_file.split(".")[0]))
                call_storm("{} {}".format(file, property_file), model_path=model_path, properties_path=properties_path, output_folder=output_path, storm_file_name=command_output_file)
            else:
                call_storm("{} prop_{}.pctl".format(file, N), model_path=model_path, properties_path=properties_path, output_folder=output_path, storm_file_name=command_output_file)
