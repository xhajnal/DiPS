import glob
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from termcolor import colored

import configparser
config = configparser.ConfigParser()
# print(os.getcwd())
workspace = os.path.dirname(__file__)
# print("workspace",workspace)
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

if "prism" not in os.environ["PATH"]:
    print("prism was probably not in PATH, adding it there")
    if "wind" in platform.system().lower():
        os.environ["PATH"] = os.environ["PATH"] + ";" + prism_path
    else:
        os.environ["PATH"] = os.environ["PATH"] + ":" + prism_path


def call_prism(args, seq=False, silent=False, model_path=model_path, properties_path=properties_path,
               prism_output_path=prism_results, std_output_path=prism_results):
    """  Solves problem of calling prism from another directory.
    
    Parameters
    ----------
    args: string for executing prism
    seq: if true it will take properties by one and append results (necessary if out of the memory)
    silent: if silent the output si set to minimum
    model_path: path to load  models from
    properties_path: path to load properties from
    std_output_path: path to save the results of the command
    prism_output_path: path to save the files inside the command
    """
    if std_output_path is not None:
        output_file_path = Path(args.split()[0]).stem
        output_file_path = os.path.join(std_output_path, Path(str(output_file_path) + ".txt"))
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
                    print(f"{colored('file', 'red')} {model_file_path} {colored('not not found -- skipped', 'red')}")
                    return False
                prism_args.append(model_file_path)
            elif re.compile('\.pctl').search(arg) is not None:
                property_file_path = os.path.join(properties_path, arg)
                # print(property_file)
                if not os.path.isfile(property_file_path):
                    print(f"{colored('file', 'red')} {property_file_path} {colored('not not found -- skipped', 'red')}")
                    return False
                prism_args.append(property_file_path)
            elif re.compile('\.txt').search(arg) is not None:
                prism_output_file_path = os.path.join(prism_output_path, arg)
                if not os.path.isfile(prism_output_file_path):
                    if not silent:
                        print(f"{colored('file', 'red')} {prism_output_file_path} {colored('not found, this may cause trouble', 'red')}")
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

        if seq:
            with open(property_file_path, 'r') as property_file:
                args.append("-property")
                args.append("")
                property_file = property_file.readlines()
                for i in range(1, len(property_file) + 1):
                    args[-1] = str(i)
                    if not silent:
                        print("calling \"", " ".join(args))
                    output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode(
                        "utf-8")
                    if std_output_path is not None:
                        with open(output_file_path, 'a') as output_file:
                            if not silent:
                                print("output here: " + str(output_file_path))
                            output_file.write(output)
        else:
            if not silent:
                print("calling \"", " ".join(args))
            output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode("utf-8")
            if std_output_path is not None:
                with open(output_file_path, 'w') as output_file:
                    if not silent:
                        print("output here: " + str(output_file_path))
                    output_file.write(output)
                    output_file.close()
    finally:
        os.chdir(curr_dir)


def call_prism_files(file_prefix, multiparam, agents_quantities, seq=False, noprobchecks=False, model_path=model_path,
                     properties_path=properties_path, output_path=prism_results):
    """  Calls prism for each file matching the prefix

    Parameters
    ----------
    file_prefix: file prefix to be matched
    multiparam: true if multiparam models are to be used
    agents_quantities: pop_sizes to be used
    seq: if true it will take properties by one and append results (necessary if out of the memory)
    noprobchecks: True if no noprobchecks option is to be used for prism
    model_path: path to load  models from
    properties_path: path to load properties from
    output_path: path for the output
    """
    # os.chdir(config.get("paths","cwd"))
    if noprobchecks:
        noprobchecks = '-noprobchecks '
    else:
        noprobchecks = ""
    for N in agents_quantities:
        for file in glob.glob(os.path.join(model_path, file_prefix + str(N) + ".pm")):
            file = Path(file)
            start_time = time.time()
            # print("{} seq={}{} >> {}".format(file, seq, noprobchecks, str(prism_results)))
            if multiparam:
                q = ""
                for i in range(1, N):
                    q = "{},q{}=0:1".format(q, i)
                    # q=q+",q"+str(i)"=0:1"
            else:
                q = ",q=0:1"
            # print("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q))
            skipped = call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file, N, noprobchecks, q), seq=seq,
                       model_path=model_path, properties_path=properties_path, std_output_path=output_path)
            if skipped:
                continue
            if not seq:
                # if 'GC overhead' in tailhead.tail(open('output_path/{}.txt'.format(file.split('.')[0])),40).read():
                if 'GC overhead' in open(os.path.join(output_path, "{}.txt".format(file.stem))).read():
                    seq = True
                    print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
                    start_time = time.time()
                    call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file, N, noprobchecks, q), seq=False,
                               model_path=model_path, properties_path=properties_path, std_output_path=output_path)
            if not noprobchecks:
                if '-noprobchecks' in open(os.path.join(output_path, "{}.txt".format(file.stem))).read():
                    print("An error occurred, running with noprobchecks option")
                    noprobchecks = '-noprobchecks '
                    print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
                    start_time = time.time()
                    call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file, N, noprobchecks, q), seq=False,
                               model_path=model_path, properties_path=properties_path, std_output_path=output_path)
            print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run \n")


if __name__ == "__main__":
    agents_quantities = [2, 3]
    try:
        os.mkdir("test")
    except:
        print("folder src/test probably already exists, if not this will fail")
    os.chdir("test")
    cwd = os.getcwd()

    call_prism(
        "multiparam_synchronous_parallel_10.pm -const p=0.028502714675268215,q1=0.5057623641293089 -simpath 2 dummy_path1550773616.0244777.txt",
        silent=True, prism_output_path="/home/matej/Git/mpm/src/test/new", std_output_path=None)

    ## model checking
    print(colored('testing simple model checking', 'blue'))
    for population in agents_quantities:
        call_prism("semisynchronous_parallel_{}.pm prop_{}.pctl -param p=0:1,q=0:1,alpha=0:1"
                   .format(population, population), seq=False, std_output_path=cwd)

    ## simulating the path
    print(colored('testing simulation', 'blue'))
    call_prism(
        'synchronous_parallel_2.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
        'path1.txt', prism_output_path=cwd,std_output_path=None)

    print(colored('test simulation change the path of the path files output', 'blue'))
    call_prism(
        'synchronous_parallel_2.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
        'path1.txt', prism_output_path="/home/matej/Git/mpm/src/test/new", std_output_path=None)

    #print(colored('testing simulation with stdout', 'blue'))
    #file = open("path_synchronous_parallel__2_3500_0.028502714675268215_0.5057623641293089.txt", "w+")
    #print(colored('testing not existing input file', 'blue'))
    #call_prism(
    #    'fake.pm -const p=0.028502714675268215,q=0.5057623641293089 -simpath 2 '
    #    'path_synchronous_parallel__2_3500_0.028502714675268215_0.5057623641293089.txt', prism_output_path=cwd,std_output_path=None)

    ## call_prism_files
    print(colored('call_prism_files', 'blue'))
    call_prism_files("syn*_", False, agents_quantities)
    print(colored('call_prism_files2', 'blue'))
    call_prism_files("multiparam_syn*_", True, agents_quantities)