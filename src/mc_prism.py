import configparser
import glob
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

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


def call_prism(args, seq, silent=True, model_path=model_path, properties_path=properties_path, output_path=prism_results):
    """  Solves problem of calling prism from another directory.
    
    Parameters
    ----------
    args: string for executing prism
    seq: if true it will take properties by one and append results (neccesary if out of the memory)
    model_path: path to models
    properties_path: path to properties
    output_path: path for the output
    """
    output_file = Path(args.split()[0]).stem
    output_file = Path(str(output_file) + ".txt")

    # os.chdir(config.get("paths","cwd"))
    curr_dir = os.getcwd()
    os.chdir(prism_path)
    # print(os.getcwd())
    prism_args = []

    try:
        # print(args.split(" "))

        args = args.split(" ")
        # print(args)
        propfile = args[1]
        # print(propfile)
        for arg in args:
            # print(arg)
            # print(re.compile('\.[a-z]').search(arg))
            if re.compile('\.pm').search(arg) is not None:
                prism_args.append(os.path.join(model_path, arg))
            elif re.compile('\.pctl').search(arg) is not None:
                prism_args.append(os.path.join(properties_path, arg))
                # print(prism_args)
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
            with open(os.path.join(output_path, output_file), 'a') as f:
                with open(os.path.join(properties_path, propfile), 'r') as prop:
                    args.append("-property")
                    args.append("")
                    prop = prop.readlines()
                    for i in range(1, len(prop) + 1):
                        args[-1] = str(i)
                        if not silent:
                            print(str(args) + " >> " + str(os.path.join(output_path, output_file)))
                        output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode(
                            "utf-8")
                        # print(output)
                        f.write(output)
        else:
            with open(os.path.join(output_path, output_file), 'w') as f:
                if not silent:
                    print(str(args) + " >> " + str(os.path.join(output_path, output_file)))
                output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode("utf-8")
                # print(output)
                f.write(output)
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
    model_path: path to models
    properties_path: path to properties
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
            print("{} seq={}{} >> {}".format(file, seq, noprobchecks, str(prism_results)))
            if multiparam:
                q = ""
                for i in range(1, N):
                    q = "{},q{}=0:1".format(q, i)
                    # q=q+",q"+str(i)"=0:1"
            else:
                q = ",q=0:1"
            # print("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q))
            call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file, N, noprobchecks, q), seq)
            if not seq:
                # if 'GC overhead' in tailhead.tail(open('prism_results/{}.txt'.format(file.split('.')[0])),40).read():
                if 'GC overhead' in open(os.path.join(prism_results, "{}.txt".format(file.stem))).read():
                    seq = True
                    print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
                    start_time = time.time()
                    call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file, N, noprobchecks, q), False, model_path,
                               properties_path, output_path)
            if not noprobchecks:
                if '-noprobchecks' in open(os.path.join(prism_results, "{}.txt".format(file.stem))).read():
                    print("An error occurred, running with noprobchecks option")
                    noprobchecks = '-noprobchecks '
                    print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
                    start_time = time.time()
                    call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file, N, noprobchecks, q), False, model_path,
                               properties_path, output_path)
            print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
