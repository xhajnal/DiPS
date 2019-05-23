import glob
import os
import platform
import re
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import psutil
from termcolor import colored

import configparser

config = configparser.ConfigParser()
# print(os.getcwd())
workspace = os.path.dirname(__file__)
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)

config.read(os.path.join(workspace, "../config.ini"))
# config.sections()
# prism_path = config.paths['prism_path']

prism_path = config.get("paths", "prism_path")
if not os.path.exists(prism_path):
    raise OSError("Directory does not exist: " + str(prism_path))

model_path = Path(config.get("paths", "models"))
if not os.path.exists(model_path):
    raise OSError("Directory does not exist: " + str(model_path))

properties_path = Path(config.get("paths", "properties"))
if not os.path.exists(properties_path):
    raise OSError("Directory does not exist: " + str(model_path))

prism_results = config.get("paths", "prism_results")
if not os.path.exists(prism_results):
    os.makedirs(prism_results)

storm_results = config.get("paths", "storm_results")
if not os.path.exists(storm_results):
    os.makedirs(storm_results)


if "prism" not in os.environ["PATH"]:
    print("prism was probably not in PATH, adding it there")
    if "wind" in platform.system().lower():
        os.environ["PATH"] = os.environ["PATH"] + ";" + prism_path
    else:
        os.environ["PATH"] = os.environ["PATH"] + ":" + prism_path

os.chdir(cwd)


def write_to_file(std_output_path, output_file_path, output, silent, append=False):
    """  Generic writing to a file

    Args
    ----------
    std_output_path: (string) path to write the output
    output_file_path: (string) path of the output file
    output: (string) text to be written into the file
    silent: (Bool) if silent command line output is set to minimum
    append: (Bool) if True appending instead of writing from the start
    """
    if std_output_path is not None:
        if not silent:
            print("output here: " + str(output_file_path))
        if append:
            with open(output_file_path, 'a') as output_file:
                output_file.write(output)
        else:
            with open(output_file_path, 'w') as output_file:
                output_file.write(output)
                output_file.close()


def set_javaheap_win(size):
    """  Changing the java heap size for the PRISM on Windows

    Args
    ----------
    size: (str) sets maximum memory, see https://www.prismmodelchecker.org/manual/ConfiguringPRISM/OtherOptions

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


def call_prism(args, seq=False, silent=False, model_path=model_path, properties_path=properties_path,
               prism_output_path=prism_results, std_output_path=prism_results, std_output_file=False):
    """  Solves problem of calling prism from another directory.

    Args
    ----------
    args: (string) args for executing prism
    seq: (Bool) if true it will take properties one by one and append the results (helps to deal with memory)
    silent: (Bool) if silent command line output is set to minimum
    model_path: (string) path to load  models from
    properties_path: (string) path to load properties from
    prism_output_path: (string) path to save the files inside the command
    std_output_path: (string) path to save the results of the command
    std_output_file: (string) file name to save the output
    """
    # print("std_output_path", std_output_path)
    # print("prism_results", prism_results)
    # print("std_output_file", std_output_file)

    if std_output_path is not None:
        output_file_path = Path(args.split()[0]).stem
        if not std_output_file:
            output_file_path = os.path.join(std_output_path, Path(str(output_file_path) + ".txt"))
        else:
            output_file_path = os.path.join(prism_results, Path(str(std_output_file)))
            # print("new output_file_path", output_file_path)
    else:
        output_file_path = ""


    # print(output_file)

    # os.chdir(config.get("paths","cwd"))
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
            if re.compile('\.pm').search(arg) is not None:
                model_file_path = os.path.join(model_path, arg)
                # print(model_file)
                if not os.path.isfile(model_file_path):
                    print(f"{colored('file', 'red')} {model_file_path} {colored('not found -- skipped', 'red')}")
                    return 404
                prism_args.append(model_file_path)
            elif re.compile('\.pctl').search(arg) is not None:
                property_file_path = os.path.join(properties_path, arg)
                # print(property_file)
                if not os.path.isfile(property_file_path):
                    print(f"{colored('file', 'red')} {property_file_path} {colored('not found -- skipped', 'red')}")
                    return 404
                prism_args.append(property_file_path)
            elif re.compile('\.txt').search(arg) is not None:
                prism_output_file_path = os.path.join(prism_output_path, arg)
                if not os.path.isfile(prism_output_file_path):
                    if not silent:
                        print(
                            f"{colored('file', 'red')} {prism_output_file_path} {colored('not found, this may cause trouble', 'red')}")
                prism_args.append(os.path.join(prism_output_path, arg))
            else:
                prism_args.append(arg)
        # print(prism_args)
        # prism_args.append(" ".join(args.split(" ")[-2:]))
        # print(prism_args)

        # print(sys.platform)
        if sys.platform.startswith("win"):
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
                            if sys.platform.startswith("win"):
                                previous_memory = set_javaheap_win(f"{memory}g")
                            if not silent:
                                print("calling \"", " ".join(args), "\"")
                            output = subprocess.run(args, stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT).stdout.decode("utf-8")
                            if 'OutOfMemoryError' in output:
                                write_to_file(std_output_path, output_file_path, output, silent, append=True)
                                print(colored(f"A memory error occurred while seq even after increasing the memory, close some programs and try again", "red"))
                                if sys.platform.startswith("win"):
                                    set_javaheap_win(previous_memory)
                                return "memory_fail"
                        else:
                            write_to_file(std_output_path, output_file_path, output, silent, append=True)
                            print(colored(f"A memory error occurred while seq with given amount of memory", "red"))
                            ## Changing the memory setting back
                            if sys.platform.startswith("win"):
                                set_javaheap_win(previous_memory)
                            return "memory"

                    write_to_file(std_output_path, output_file_path, output, silent, append=True)

                    if 'Syntax error' in output:
                        print(colored(f"A syntax error occurred", "red"))
                        return "syntax"
                    if "Cannot allocate memory" in output:
                        print(colored(f"A memory error occurred while seq, close some programs and try again", "red"))
                        return "memory_fail"
                    if 'NullPointerException' in output:
                        print(colored(f"A NullPointerException occurred", "red"))
                        return "NullPointerException"
                    if 'use -noprobchecks' in output:
                        print(colored(f"Outgoing transitions checksum error occurred", "red"))
                        return "noprobchecks"

        else:
            if not silent:
                print("calling \"", " ".join(args))
            output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")
            write_to_file(std_output_path, output_file_path, output, silent, append=False)

            ## Check for errors
            if ('OutOfMemoryError' in output) or ("Cannot allocate memory" in output):
                print(colored(f"A memory error occurred", "red"))
                return "memory"
            if 'Syntax error' in output:
                print(colored(f"A syntax error occurred", "red"))
                return "syntax"
            if 'NullPointerException' in output:
                print(colored(f"A NullPointerException occurred", "red"))
                return "NullPointerException"
            if 'use -noprobchecks' in output:
                print(colored(f"Outgoing transitions checksum error occurred", "red"))
                return "noprobchecks"

        return 0
    finally:
        os.chdir(curr_dir)


def call_prism_files(model_prefix, agents_quantities, param_intervals=False, seq=False, noprobchecks=False, memory="",
                     model_path=model_path, properties_path=properties_path, property_file=False, output_path=prism_results):
    """  Calls prism for each file matching the prefix

    Args
    ----------
    model_prefix: file prefix to be matched
    agents_quantities: (int) pop_sizes to be used
    param_intervals (list of pairs): list of intervals to be used for respective parameter (default all intervals are from 0 to 1)
    seq: (Bool) if true it will take properties one by one and append the results (helps to deal with memory)
    noprobchecks: (Bool) True if no noprobchecks option is to be used for prism
    model_path: (string) path to load  models from
    properties_path: (string) path to load properties from
    param_intervals: (list of pairs) parameter intervals
    property_file: (string) file name of single property files to be used for all models
    output_path: (string) path for the output
    memory: (int) sets maximum memory in GB, see https://www.prismmodelchecker.org/manual/ConfiguringPRISM/OtherOptions

    """
    # os.chdir(config.get("paths","cwd"))
    if noprobchecks:
        noprobchecks = '-noprobchecks '
    else:
        noprobchecks = ""

    if memory == "":
        memory = ""
    elif "javamaxmem" not in str(memory):
        memory = f'-javamaxmem {memory}g '

    for N in sorted(agents_quantities):
        # print(glob.glob(os.path.join(model_path, file_prefix + str(N) + ".pm")))
        if not glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm")):
            print(colored("No files for N="+str(N)+" found", "red"))
        for file in glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm")):
            file = Path(file)
            start_time = time.time()
            # print("{} seq={}{} >> {}".format(file, seq, noprobchecks, str(prism_results)))

            ## Parsing the parameters from the files
            params = ""
            # print(file)
            with open(file, 'r') as input_file:
                i = 0
                for line in input_file:
                    if line.startswith('const'):
                        # print(line)
                        line = line.split(" ")[-1].split(";")[0]
                        if param_intervals:
                            params = f"{params}{line}={param_intervals[i][0]}:{param_intervals[i][1]},"
                        else:
                            params = f"{params}{line}=0:1,"
                        i = i + 1
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
                error = call_prism("{} prop_{}.pctl {}{}-param {}".format(file, N, memory, noprobchecks, params),
                                   seq=seq, model_path=model_path, properties_path=properties_path,
                                   std_output_path=output_path)
            else:
                # print("output_path", output_path)
                # print("file", file.stem)
                error = call_prism("{} {} {}{}-param {}".format(file, property_file, memory, noprobchecks, params),
                                   seq=seq, model_path=model_path, properties_path=properties_path, std_output_path=output_path,
                                   std_output_file="{}_{}.txt".format(str(file.stem).split(".")[0], property_file.split(".")[0]))

            print(f"  Return code is: {error}")
            print(f"  It took {socket.gethostname()}, {time.time() - start_time} seconds to run")

            if error == 404:
                print(colored("A file not found, skipped", "red"))
                continue

            ## Check for syntax error
            if error == "syntax":
                print(colored("A syntax error occurred, sorry we can not correct that", "red"))
                continue

            ## Check if there was problem with sum of probabilities
            if error == "noprobchecks":
                if noprobchecks == "":
                    print(colored("Outgoing transitions checksum error occurred. Running with noprobchecks option", "red"))
                    noprobchecks = '-noprobchecks '
                else:
                    print(colored("This is embarrassing, but Outgoing transitions checksum error occurred while noprobchecks option", "red"))

            ## Check if memory problem has occurred
            if error == "memory":
                if not seq:
                    ## A memory occurred while not seq, trying seq now
                    seq = True
                    ## Remove the file because appending would no overwrite the file
                    os.remove(os.path.join(output_path, "{}.txt".format(file.stem)))
                    print(colored("A memory error occurred. Running prop by prob now", "red"))
                else:
                    ## A memory occurred while seq
                    ## Remove the file because appending would not overwrite the file
                    os.remove(os.path.join(output_path, "{}.txt".format(file.stem)))
                    memory = round(psutil.virtual_memory()[0] / 1024 / 1024 / 1024)  ## total memory converted to GB
                    if sys.platform.startswith("win"):
                        previous_memory = set_javaheap_win(f"{memory}g")
                    print(colored(f"A memory error occurred while seq, max memory increased to {memory}GB", "red"))

            if error == "memory_fail":
                ## A error occured even when seq and max memory, no reason to continue
                break

            if error == "NullPointerException":
                if seq:
                    print(colored("Sorry, I do not know to to fix this, please try it manually", "red"))
                    break
                else:
                    print(colored("Trying to fix the null pointer exception by running prop by prop", "red"))
                    seq = True
                    ## Remove the file because appending would no overwrite the file
                    os.remove(os.path.join(output_path, "{}.txt".format(file.stem)))

            if error is not 0:
                ## If an error occurred call this function for this file again
                print()
                # print("seq",seq)
                # print("noprobchecks", noprobchecks)
                call_prism_files(model_prefix, [N], seq=seq, noprobchecks=noprobchecks, memory=memory, model_path=model_path,
                                 properties_path=properties_path, property_file=property_file, output_path=prism_results)
            print()

    if sys.platform.startswith("win"):
        set_javaheap_win(previous_memory)


def call_storm(args, silent=False, model_path=model_path, properties_path=properties_path,
               prism_output_path=prism_results, std_output_path=storm_results, std_output_file=False, time=False):
    """  Prints calls for prism model checking.

    Args
    ----------
    args: (string) args for executing prism
    silent: (Bool) if silent command line output is set to minimum
    model_path: (string) path to load  models from
    properties_path: (string) path to load properties from
    prism_output_path: (string) path to save the files inside the command
    std_output_path: (string) path to save the results of the command
    std_output_file: (string) file name to save the output
    time: (Bool) if True time measurement is added
    """
    # print("std_output_path", std_output_path)
    # print("prism_results", prism_results)
    # print("std_output_file", std_output_file)

    if std_output_path is not None:
        output_file_path = Path(args.split()[0]).stem
        if not std_output_file:
            output_file_path = os.path.join(std_output_path, Path(str(output_file_path) + ".txt"))
        else:
            output_file_path = os.path.join(prism_results, Path(str(std_output_file)))
            # print("new output_file_path", output_file_path)
    else:
        output_file_path = ""

    # print(std_output_file)

    # os.chdir(config.get("paths","cwd"))
    os.chdir(prism_path)
    # print(os.getcwd())
    prism_args = []

    # print(args.split(" "))
    args = args.split(" ")

    # print(args)
    for arg in args:
        # print(arg)
        # print(re.compile('\.[a-z]').search(arg))
        if re.compile('\.pm').search(arg) is not None:
            model_file_path = os.path.join(model_path, arg)
            # print(model_file)
            if not os.path.isfile(model_file_path):
                print(f"{colored('file', 'red')} {model_file_path} {colored('not found -- skipped', 'red')}")
                return 404
            prism_args.append(model_file_path)
        elif re.compile('\.pctl').search(arg) is not None:
            property_file_path = os.path.join(properties_path, arg)
            # print(property_file)
            if not os.path.isfile(property_file_path):
                print(f"{colored('file', 'red')} {property_file_path} {colored('not found -- skipped', 'red')}")
                return 404
            # prism_args.append(property_file_path)
            prism_args.append("my_super_cool_string")
        elif re.compile('\.txt').search(arg) is not None:
            prism_output_file_path = os.path.join(prism_output_path, arg)
            if not os.path.isfile(prism_output_file_path):
                if not silent:
                    print(
                        f"{colored('file', 'red')} {prism_output_file_path} {colored('not found, this may cause trouble', 'red')}")
            prism_args.append(os.path.join(prism_output_path, arg))
        else:
            prism_args.append(arg)

    args = ["./storm-pars --prism"]
    args.extend(prism_args)
    if time:
        args.append(")")
    if output_file_path is not "":
        args.append(">>")
        args.append(output_file_path)
        args.append("2>&1")

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
            print(spam.replace("my_super_cool_string", f"--prop \"{line[:-1]}\""))

    return True
    # output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")
    # write_to_file(std_output_path, output_file_path, output, silent, append=False)


def call_storm_files(model_prefix, agents_quantities, model_path=model_path, properties_path=properties_path, property_file=False, output_path=storm_results, time=False):
    """  Calls prism for each file matching the prefix

    Args
    ----------
    model_prefix: file prefix to be matched
    agents_quantities: (int) pop_sizes to be used
    model_path: (string) path to load  models from
    param_intervals (list of pairs): list of intervals to be used for respective parameter (default all intervals are from 0 to 1)
    properties_path: (string) path to load properties from
    property_file: (string) file name of single property files to be used for all models
    output_path: (string) path for the output
    time: (Bool) if True time measurement is added

    """
    print("docker pull movesrwth/storm:travis")
    print(f'docker run --mount type=bind,source="$(pwd)",target=/{os.path.basename(os.path.normpath(os.getcwd()))} -w /opt/storm/build/bin --rm -it --name storm movesrwth/storm:travis')

    for N in sorted(agents_quantities):
        # print(glob.glob(os.path.join(model_path, file_prefix + str(N) + ".pm")))
        if not glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm")):
            print(colored("No files for N="+str(N)+" found", "red"))
            continue
        for file in glob.glob(os.path.join(model_path, model_prefix + str(N) + ".pm")):
            file = Path(file)
            # print("{} {}".format(file, property_file))
            # call_storm("{} {}".format(file, property_file), model_path=model_path, properties_path=properties_path, std_output_path=output_path, std_output_file="{}_{}.txt".format(str(file.stem).split(".")[0], property_file.split(".")[0]), time=time)

            if property_file:
                # print("property_file", property_file)
                # print("file", file)
                # print("file stem", file.resolve().stem)
                # print("{}_{}.txt".format(str(file.stem).split(".")[0], property_file.split(".")[0]))
                call_storm("{} {}".format(file, property_file), model_path=model_path, properties_path=properties_path, std_output_path=output_path, std_output_file="{}_{}.txt".format(str(file.stem).split(".")[0], property_file.split(".")[0]))
            else:
                call_storm("{} prop_{}.pctl".format(file, N), model_path=model_path, properties_path=properties_path, std_output_path=output_path)


class TestLoad(unittest.TestCase):
    def test_changing_javahep(self):
        print(colored('Test_changing_javaheap on Windows', 'blue'))
        if sys.platform.startswith("win"):
            a = (set_javaheap_win("9g"))
            print("previous memory:", a)
            set_javaheap_win(a)
        else:
            print("Skipping this test since not on windows")

    def test_storm_single_file(self):
        print(colored('Test storm call with single file', 'blue'))
        agents_quantities = [2, 3]
        for population in agents_quantities:
            call_storm("semisynchronous_{}.pm prop_{}.pctl".format(population, population), std_output_path=os.path.join(cwd, "test"))
        print(colored('Test storm call with single file with timer', 'blue'))
        for population in agents_quantities:
            call_storm("semisynchronous_{}.pm prop_{}.pctl".format(population, population), std_output_path=os.path.join(cwd, "test"), time=True)

    def test_astorm_multiple_files(self):
        print(colored('Test storm call multiple files', 'blue'))
        agents_quantities = [2, 3]
        call_storm_files("syn*_", agents_quantities, output_path=os.path.join(cwd, "test"))
        print(colored('Test storm call multiple files with specified files', 'blue'))
        call_storm_files("syn*_", agents_quantities, property_file="moments.pctl", output_path=os.path.join(cwd, "test"))

    def test_prism_easy(self):
        print(colored('Test prism call with single file', 'blue'))
        agents_quantities = [2, 3]
        try:
            os.mkdir("test")
        except FileExistsError:
            print("folder src/test probably already exists, if not this will fail")
        os.chdir("test")

        call_prism(
            "multiparam_synchronous_10.pm -const p=0.028502714675268215,q1=0.5057623641293089 -simpath 2 dummy_path1550773616.0244777.txt",
            silent=True, prism_output_path="/home/matej/Git/mpm/src/test/new", std_output_path=None)

        ## Model checking
        print(colored('Testing simple model checking', 'blue'))
        for population in agents_quantities:
            call_prism("semisynchronous_{}.pm prop_{}.pctl -param p=0:1,q=0:1,alpha=0:1"
                       .format(population, population), seq=False, std_output_path=cwd)

        ## Simulating the path
        print(colored('Testing simulation', 'blue'))
        call_prism(
            'synchronous_2.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
            'path1.txt', prism_output_path=cwd, std_output_path=None)

        print(colored('Test simulation change the path of the path files output', 'blue'))
        print(colored('This should produce a file in ', 'blue'))
        call_prism(
            'synchronous_2.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
            'path1.txt', prism_output_path="../test/new", std_output_path=None)

        # print(colored('testing simulation with stdout', 'blue'))
        # file = open("path_synchronous__2_3500_0.028502714675268215_0.5057623641293089.txt", "w+")
        # print(colored('testing not existing input file', 'blue'))
        # call_prism(
        #    'fake.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
        #    'path_synchronous__2_3500_0.028502714675268215_0.5057623641293089.txt', prism_output_path=cwd,std_output_path=None)

        ## call_prism_files
        print(colored('Call_prism_files', 'blue'))
        call_prism_files("syn*_", agents_quantities)

        print(colored('Call_prism_files2', 'blue'))
        call_prism_files("multiparam_syn*_", agents_quantities)

    def test_prism_heavy_load(self):
        agents_quantities = [20, 40]
        ## 20 should pass
        ## This will require noprobcheck for 40

        call_prism_files("syn*_", agents_quantities, output_path=os.path.join(cwd, "test"))
        ## This will require seq for 40
        call_prism_files("semi*_", agents_quantities, output_path=os.path.join(cwd, "test"))
        ## This will require seq with adding the memory for 40
        call_prism_files("asyn*_", agents_quantities, output_path=os.path.join(cwd, "test"))


if __name__ == "__main__":
    unittest.main()
