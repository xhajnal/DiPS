import configparser
import os
from pathlib import Path
from typing import Any, Union
import common.math as mymath

config = configparser.ConfigParser()
# print(os.getcwd())
workspace = os.path.dirname(__file__)
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)

config.read(os.path.join(workspace, "../config.ini"))
# config.sections()
model_path: Union[Path, Any] = Path(config.get("paths", "models"))
if not os.path.exists(model_path):
    os.makedirs(model_path)

properties_path = Path(config.get("paths", "properties"))
if not os.path.exists(properties_path):
    os.makedirs(properties_path)

os.chdir(cwd)


def create_synchronous_model(file, N):
    """ Creates synchronous model of *N* agents to a *file* with probabilities p and q in [0,1].
    For more information see the HSB19 paper.

    Args
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity

    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    p: (float) - probability to succeed in the first attempt
    q: (float) - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)
    
    ## computing how many zeros to put
    decimals = (N // 10)+1

    first_attempt = []
    coefficient = ""

    for i in range(N + 1):
        for j in range(N - i):
            coefficient = coefficient + "*p"
        for j in range(N - i, N):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(N, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")
    file.write(f"const double q;\n")
    file.write(f"\n")

    # module here
    file.write(f"module two_param_agents_{N}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, N):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(N + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(N):
            file.write(f"(a{j}'=" + str(1 if N - i > j else 2) + ")")
            if j < N - 1:
                file.write(f" & ")
        if i < N:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # non-initial transition

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("N",N,"i",i)
        file.write(f"(a{N - 1}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for ones in range(1, N):
        file.write(f"       []   a0 = 1")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if ones > j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        twos = N - ones
        # file.write(f"twos: {}".format(twos))

        for successes in range(0, twos + 1):
            file.write(str(mymath.nCr(twos, successes)))
            for ok in range(successes):
                file.write(f"*q")
            for nok in range(twos - successes):
                file.write(f"*(1-q)")
            file.write(f": ")

            for k in range(1, N + 1):
                if k <= ones + successes:
                    if k == N:
                        file.write(f"(a{k - 1}'=1)")
                    else:
                        file.write(f"(a{k - 1}'=1) & ")
                elif k == N:
                    file.write(f"(a{k - 1}'=0)")
                else:
                    file.write(f"(a{k - 1}'=0) & ")
            if successes < twos:
                file.write(f" + ")

        file.write(f";\n")
    file.write(f"\n")

    # all twos transitions
    file.write(f"       // all twos transition\n")
    file.write(f"       []   a0 = 2")
    for i in range(1, N):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(N - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{N - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":" + str(i*i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_semisynchronous_model(file, N):
    """ Creates semisynchronous model of *N* agents to a *file* with probabilities p and q in [0,1]. For more information see the HSB19 paper.

    Args
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity

    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    p: (float) - probability to succeed in the first attempt
    q: (float) - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)
    
    ## computing how many zeros to put
    decimals = (N // 10)+1

    first_attempt = []
    coefficient = ""

    for i in range(N + 1):
        for j in range(N - i):
            coefficient = coefficient + "*p"
        for j in range(N - i, N):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(N, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")
    file.write(f"const double q;\n")
    file.write(f"\n")

    # module here
    file.write(f"module two_param_agents_{N}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, N):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(N + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(N):
            file.write(f"(a{j}'=" + str(1 if N - i > j else 2) + ")")
            if j < N - 1:
                file.write(f" & ")
        if i < N:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("N",N,"i",i)
        file.write(f"(a{N - 1}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(N - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q:")
        for j in range(N):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < N - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q:")
        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{N - 1}'= 0)")
        # print("N",N,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    for o in range(1, N - 1):
        # file.write(f"help")
        for t in range(1, N - o):
            z = N - t - o
            file.write(f"       []   a0 = 1")
            for j in range(1, o):
                file.write(f" & a{j} = 1")
            for j in range(o, o + t):
                file.write(f" & a{j} = 2")
            for j in range(o + t, o + t + z):
                file.write(f" & a{j} = 0")

            file.write(f" -> ")
            file.write(f"q: (a0' = 1)")
            for j in range(1, o + 1):
                file.write(f" & (a{j}'= 1)")
            for j in range(o + 1, o + t):
                file.write(f" & (a{j}'= 2)")
            for j in range(o + t, o + t + z):
                file.write(f" & (a{j}'= 0)")

            file.write(f" + ")
            file.write(f"1-q: (a0' = 1)")
            for j in range(1, o):
                file.write(f" & (a{j}'= 1)")
            for j in range(o, o + t - 1):
                file.write(f" & (a{j}'= 2)")
            for j in range(o + t - 1, o + t + z):
                file.write(f" & (a{j}'= 0)")
            file.write(f";\n")

            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write(f"\n")

    # all twos transitions
    file.write(f"       // all twos transition\n")
    file.write(f"       []   a0 = 2")
    for i in range(1, N):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(N - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{N - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":" + str(i*i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_asynchronous_model(file, N):
    """ Creates asynchronous model of *N* agents to a *file* with probabilities p and q in [0,1]. For more information see the HSB19 paper.

    Args
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity


    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    p: (float) - probability to succeed in the first attempt
    q: (float) - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)
    
    ## computing how many zeros to put
    decimals = (N // 10)+1

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")
    file.write(f"const double q;\n")
    file.write(f"\n")

    # module here
    file.write(f"module two_param_agents_{N}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transitions\n")

    # some -1, some 1
    file.write(f"       // some -1, some 1 transitions\n")
    for i in reversed(range(0, N)):
        # for k in range(1,N):
        file.write(f"       []   a0 = -1")
        for j in range(1, N):
            if j > i:
                file.write(f" & a{j} = 1 ")
            else:
                file.write(f" & a{j} = -1 ")
        file.write(f"-> ")

        file.write(f"p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}' = -1)")
        if i > 0:
            file.write(f" & (a{i}' = 1)")
        for j in range(i + 1, N):
            file.write(f" & (a{j}' = 1)")

        file.write(f" + 1-p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}'= -1)")
        for j in range(max(i, 1), N - 1):
            file.write(f" & (a{j}' = 1)")
        file.write(f" & (a{N - 1}' = 2)")

        file.write(f";\n")
        # file.write(f"i="+str(i)+" j="+str(j)+" \n")
    file.write(f"\n")

    # some -1, some 2
    file.write(f"       // some -1, some 2 transitions\n")
    for i in reversed(range(0, N - 1)):
        # for k in range(1,N):
        file.write(f"       []   a0 = -1")
        for j in range(1, N):
            if j > i:
                file.write(f" & a{j} = 2")
            else:
                file.write(f" & a{j} = -1")
        file.write(f"-> ")

        file.write(f"p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}' = -1)")
        if i > 0:
            file.write(f" & (a{i}' = 1)")
        for j in range(i + 1, N):
            file.write(f" & (a{j}' = 2)")

        file.write(f" + 1-p: ")
        if i == 0:
            file.write(f"(a0' = 2)")
        else:
            file.write(f"(a0' = -1)")
        for j in range(1, i):
            file.write(f" & (a{j}'= -1)")
        if i > 0:
            file.write(f" & (a{i}' = 2)")
        for j in range(i + 1, N):
            file.write(f" & (a{j}' = 2)")

        file.write(f";\n")

    file.write(f"\n")

    # some -1, some 1, some 2
    file.write(f"       // some -1, some 1, some 2 transitions\n")
    for o in range(1, N - 1):
        # file.write(f"help")
        for t in range(1, N - o):
            z = N - t - o
            file.write(f"       []   a0 = -1")
            for j in range(1, o):
                file.write(f" & a{j} = -1")
            for j in range(o, o + t):
                file.write(f" & a{j} = 1")
            for j in range(o + t, o + t + z):
                file.write(f" & a{j} = 2")

            file.write(f" -> ")
            if o > 1:
                file.write(f"p: (a0' = -1)")
            else:
                file.write(f"p: (a0' = 1)")
            for j in range(1, o - 1):
                file.write(f" & (a{j}'= -1)")
            for j in range(max(1, o - 1), o + t):
                file.write(f" & (a{j}'= 1)")
            for j in range(o + t, o + t + z):
                file.write(f" & (a{j}'= 2)")

            file.write(f" + ")
            if o > 1:
                file.write(f"1-p: (a0' = -1)")
            else:
                file.write(f"1-p: (a0' = 1)")
            for j in range(1, o - 1):
                file.write(f" & (a{j}'= -1)")
            for j in range(max(1, o - 1), o + t - 1):
                file.write(f" & (a{j}'= 1)")
            for j in range(o + t - 1, o + t + z):
                file.write(f" & (a{j}'= 2)")
            file.write(f";\n")
    file.write(f"\n")

    # not initial transition
    file.write(f"       //  not initial transitions\n")

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("N",N,"i",i)
        file.write(f"(a{N - 1}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(N - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q:")
        for j in range(N):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < N - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q:")
        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{N - 1}'= 0)")
        # print("N",N,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    for o in range(1, N - 1):
        # file.write(f"help")
        for t in range(1, N - o):
            z = N - t - o
            file.write(f"       []   a0 = 1")
            for j in range(1, o):
                file.write(f" & a{j} = 1")
            for j in range(o, o + t):
                file.write(f" & a{j} = 2")
            for j in range(o + t, o + t + z):
                file.write(f" & a{j} = 0")

            file.write(f" -> ")
            file.write(f"q: (a0' = 1)")
            for j in range(1, o + 1):
                file.write(f" & (a{j}'= 1)")
            for j in range(o + 1, o + t):
                file.write(f" & (a{j}'= 2)")
            for j in range(o + t, o + t + z):
                file.write(f" & (a{j}'= 0)")

            file.write(f" + ")
            file.write(f"1-q: (a0' = 1)")
            for j in range(1, o):
                file.write(f" & (a{j}'= 1)")
            for j in range(o, o + t - 1):
                file.write(f" & (a{j}'= 2)")
            for j in range(o + t - 1, o + t + z):
                file.write(f" & (a{j}'= 0)")
            file.write(f";\n")

            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write(f"\n")

    # all twos transitions
    file.write(f"       // all twos transition\n")
    file.write(f"       []   a0 = 2")
    for i in range(1, N):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(N - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{N - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":" + str(i*i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_multiparam_synchronous_model(file, N):
    """ Creates synchronous model of *N* agents to a *file* with probabilities p and q1, q2, ... q(N-1) in [0,1]. 
    For more information see the HSB19 paper.
    
    Args:
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity

    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    p: (float) - probability to succeed in the first attempt
    q: (float) - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)
    
    ## computing how many zeros to put
    decimals = (N // 10)+1

    first_attempt = []
    coefficient = ""

    for i in range(N + 1):
        for j in range(N - i):
            coefficient = coefficient + "*p"
        for j in range(N - i, N):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(N, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")

    for i in range(1, N):
        file.write(f"const double q{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_agents_{N}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, N):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(N + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(N):
            file.write(f"(a{j}'=" + str(1 if N - i > j else 2) + ")")
            if j < N - 1:
                file.write(f" & ")
        if i < N:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # non-initial transition

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("N",N,"i",i)
        file.write(f"(a{N - 1}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for ones in range(1, N):
        file.write(f"       []   a0 = 1")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if ones > j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        twos = N - ones
        # file.write(f"twos: {}".format(twos))
        for successes in range(0, twos + 1):
            file.write(str(mymath.nCr(twos, successes)))
            for ok in range(successes):
                file.write(f"*q{ones:0{decimals}d}")
            for nok in range(twos - successes):
                file.write(f"*(1-q{ones:0{decimals}d})")
            file.write(f": ")

            for k in range(1, N + 1):
                if k <= ones + successes:
                    if k == N:
                        file.write(f"(a{k - 1}'=1)")
                    else:
                        file.write(f"(a{k - 1}'=1) & ")
                elif k == N:
                    file.write(f"(a{k - 1}'=0)")
                else:
                    file.write(f"(a{k - 1}'=0) & ")
            if successes < twos:
                file.write(f" + ")

        file.write(f";\n")
    file.write(f"\n")

    # all twos transitions
    file.write(f"       // all twos transition\n")
    file.write(f"       []   a0 = 2")
    for i in range(1, N):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(N - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{N - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":" + str(i*i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_multiparam_semisynchronous_model(file, N):
    """ Creates semisynchronous model of *N* agents to a *file* with probabilities p and q1, q2, ... q(N-1) in [0,1].
    For more information see the HSB19 paper.

    Args
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity

    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    p: (float) - probability to succeed in the first attempt
    q: (float) - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)
    
    ## computing how many zeros to put
    decimals = (N // 10)+1

    first_attempt = []
    coefficient = ""

    for i in range(N + 1):
        for j in range(N - i):
            coefficient = coefficient + "*p"
        for j in range(N - i, N):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(N, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")

    for i in range(1, N):
        file.write(f"const double q{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_agents_{N}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, N):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(N + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(N):
            file.write(f"(a{j}'=" + str(1 if N - i > j else 2) + ")")
            if j < N - 1:
                file.write(f" & ")
        if i < N:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("N",N,"i",i)
        file.write(f"(a{N - 1}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(N - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q{i+1:0{decimals}d}:")
        for j in range(N):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < N - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q{i+1:0{decimals}d}:")
        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{N - 1}'= 0)")
        # print("N",N,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    i = 0
    for o in range(1, N - 1):
        # file.write(f"help")
        for t in range(1, N - o):
            z = N - t - o
            i = i + 1
            file.write(f"       []   a0 = 1")
            for j in range(1, o):
                file.write(f" & a{j} = 1")
            for j in range(o, o + t):
                file.write(f" & a{j} = 2")
            for j in range(o + t, o + t + z):
                file.write(f" & a{j} = 0")

            file.write(f" -> ")
            file.write(f"q" + str(o) + ": (a0' = 1)")
            for j in range(1, o + 1):
                file.write(f" & (a{j}' = 1)")
            for j in range(o + 1, o + t):
                file.write(f" & (a{j}' = 2)")
            for j in range(o + t, o + t + z):
                file.write(f" & (a{j}' = 0)")

            file.write(f" + ")
            file.write(f"1-q" + str(o) + ": (a0' = 1)")
            for j in range(1, o):
                file.write(f" & (a{j}' = 1)")
            for j in range(o, o + t - 1):
                file.write(f" & (a{j}' = 2)")
            for j in range(o + t - 1, o + t + z):
                file.write(f" & (a{j}' = 0)")
            file.write(f";\n")

            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write(f"\n")

    # all twos transition
    file.write(f"       // all twos transition\n")
    file.write(f"       []   a0 = 2")
    for i in range(1, N):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(N - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{N - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":" + str(i*i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_multiparam_asynchronous_model(file, N):
    """ Creates semisynchronous model of *N* agents to a *file* with probabilities p and q1, q2, ... q(N-1) in [0,1].
    For more information see the HSB19 paper.

    Args
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity

    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    p: (float) - probability to succeed in the first attempt
    q: (float) - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)
    
    ## computing how many zeros to put
    decimals = (N // 10)+1
    
    first_attempt = []
    coefficient = ""

    for i in range(N + 1):
        for j in range(N - i):
            coefficient = coefficient + "*p"
        for j in range(N - i, N):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(N, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")

    for i in range(1, N):
        file.write(f"const double q{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_agents_{N}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transitions\n")

    # some -1, some 1
    file.write(f"       // some -1, some 1 transitions\n")
    for i in reversed(range(0, N)):
        # for k in range(1,N):
        file.write(f"       []   a0 = -1")
        for j in range(1, N):
            if j > i:
                file.write(f" & a{j} = 1 ")
            else:
                file.write(f" & a{j} = -1 ")
        file.write(f"-> ")

        file.write(f"p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}' = -1)")
        if i > 0:
            file.write(f" & (a{i}' = 1)")
        for j in range(i + 1, N):
            file.write(f" & (a{j}' = 1)")

        file.write(f" + 1-p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}'= -1)")
        for j in range(max(i, 1), N - 1):
            file.write(f" & (a{j}' = 1)")
        file.write(f" & (a{N - 1}' = 2)")

        file.write(f";\n")
        # file.write(f"i="+str(i)+" j="+str(j)+" \n")
    file.write(f"\n")

    # some -1, some 2
    file.write(f"       // some -1, some 2 transitions\n")
    for i in reversed(range(0, N - 1)):
        # for k in range(1,N):
        file.write(f"       []   a0 = -1")
        for j in range(1, N):
            if j > i:
                file.write(f" & a{j} = 2")
            else:
                file.write(f" & a{j} = -1")
        file.write(f"-> ")

        file.write(f"p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}' = -1)")
        if i > 0:
            file.write(f" & (a{i}' = 1)")
        for j in range(i + 1, N):
            file.write(f" & (a{j}' = 2)")

        file.write(f" + 1-p: ")
        if i == 0:
            file.write(f"(a0' = 2)")
        else:
            file.write(f"(a0' = -1)")
        for j in range(1, i):
            file.write(f" & (a{j}'= -1)")
        if i > 0:
            file.write(f" & (a{i}' = 2)")
        for j in range(i + 1, N):
            file.write(f" & (a{j}' = 2)")

        file.write(f";\n")

    file.write(f"\n")

    # some -1, some 1, some 2
    file.write(f"       // some -1, some 1, some 2 transitions\n")
    for o in range(1, N - 1):
        # file.write(f"help")
        for t in range(1, N - o):
            z = N - t - o
            file.write(f"       []   a0 = -1")
            for j in range(1, o):
                file.write(f" & a{j} = -1")
            for j in range(o, o + t):
                file.write(f" & a{j} = 1")
            for j in range(o + t, o + t + z):
                file.write(f" & a{j} = 2")

            file.write(f" -> ")
            if o > 1:
                file.write(f"p: (a0' = -1)")
            else:
                file.write(f"p: (a0' = 1)")
            for j in range(1, o - 1):
                file.write(f" & (a{j}'= -1)")
            for j in range(max(1, o - 1), o + t):
                file.write(f" & (a{j}'= 1)")
            for j in range(o + t, o + t + z):
                file.write(f" & (a{j}'= 2)")

            file.write(f" + ")
            if o > 1:
                file.write(f"1-p: (a0' = -1)")
            else:
                file.write(f"1-p: (a0' = 1)")
            for j in range(1, o - 1):
                file.write(f" & (a{j}'= -1)")
            for j in range(max(1, o - 1), o + t - 1):
                file.write(f" & (a{j}'= 1)")
            for j in range(o + t - 1, o + t + z):
                file.write(f" & (a{j}'= 2)")
            file.write(f";\n")
    file.write(f"\n")

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("N",N,"i",i)
        file.write(f"(a{N - 1}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(N - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q{i + 1 :0{decimals}d}:")
        for j in range(N):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < N - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q{i + 1 :0{decimals}d}:")
        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{N - 1}'= 0)")
        # print("N",N,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    i = 0
    for o in range(1, N - 1):
        # file.write(f"help")
        for t in range(1, N - o):
            z = N - t - o
            i = i + 1
            file.write(f"       []   a0 = 1")
            for j in range(1, o):
                file.write(f" & a{j} = 1")
            for j in range(o, o + t):
                file.write(f" & a{j} = 2")
            for j in range(o + t, o + t + z):
                file.write(f" & a{j} = 0")

            file.write(f" -> ")
            file.write(f"q" + str(o) + ": (a0' = 1)")
            for j in range(1, o + 1):
                file.write(f" & (a{j}' = 1)")
            for j in range(o + 1, o + t):
                file.write(f" & (a{j}' = 2)")
            for j in range(o + t, o + t + z):
                file.write(f" & (a{j}' = 0)")

            file.write(f" + ")
            file.write(f"1-q" + str(o) + ": (a0' = 1)")
            for j in range(1, o):
                file.write(f" & (a{j}' = 1)")
            for j in range(o, o + t - 1):
                file.write(f" & (a{j}' = 2)")
            for j in range(o + t - 1, o + t + z):
                file.write(f" & (a{j}' = 0)")
            file.write(f";\n")

            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write(f"\n")

    # all twos transition
    file.write(f"       // all twos transition\n")
    file.write(f"       []   a0 = 2")
    for i in range(1, N):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(N - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{N - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print("N",N,"i",i,"j",j)
            file.write(f" & a{j} = {(0,1)[i>j]}")
        file.write(f":" + str(i*i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_bee_multiparam_synchronous_model(file, N):
    """ Creates synchronous model of *N* agents to a *file* with probabilities r_i in [0,1].

    Args:
    ----------
    file: (string) - filename with extension
    N: (int) - agent quantity

    Model meaning
    ----------
    Params:
    N: (int) - number of agents (agents quantity)
    r_i: (float) - probability of success of an agent when i amount of pheromone is present
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = (N // 10)+1

    first_attempt = []
    coefficient = ""

    for i in range(N + 1):
        for j in range(N - i):
            coefficient = coefficient + f"*r_{0:0{decimals}d}"
        for j in range(N - i, N):
            coefficient = coefficient + f"*(1-r_{0:0{decimals}d})"
        coefficient = str(mymath.nCr(N, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")

    for i in range(0, N):
        file.write(f"const double r_{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_bee_agents_{N}\n")
    file.write(f"       // ai - state of agent i: 3:init 1:success -j: failure when j amount of pheromone present \n")
    for i in range(N):
        file.write(f"       a{i} : [-{N-1}..3] init 3; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = 3")
    for i in range(1, N):
        file.write(f" & a{i} = 3 ")
    file.write(f" & b = 0 ")
    file.write(f"-> ")
    for i in range(N + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(N):
            file.write(f"(a{j}'=" + str(1 if N - i > j else -0) + ")")
            if j < N - 1:
                file.write(f" & ")
        if i < N:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # non-initial transition

    # some ones, some nonpositive final transitions
    file.write(f"       // some ones, some nonpositive final transitions\n")
    for i in range(N + 1):
        file.write(f"       []   a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print(f"N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f" & b = 0 ")
        file.write(f" -> ")

        for j in range(N - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print(f"N",N,"i",i)
        file.write(f"(a{(N-1)}'= " + str(1 if i == N else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some nonpositive transitions
    file.write(f"       // some ones, some nonpositive transitions\n")
    for ones in range(1, N):
        for fails in range(0, ones):
            file.write(f"       []   a0 = 1")
            for j in range(1, N):
                # print(f"N",N,"i",i,"j",j)
                file.write(f" & a{j} = " + str(1 if ones > j else (-fails)))
                # print(f" & a"+str(j)+" = "+str( 1 if i>=j else 0 ))
            file.write(f" & b = 0 ")
            file.write(f" -> ")

            twos = N - ones
            # file.write(f"twos: {}".format(twos))
            for successes in range(0, twos + 1):
                file.write(str(mymath.nCr(twos, successes)))
                for ok in range(successes):
                    file.write(f"* ((r_{ones:0{decimals}d} - r_{fails:0{decimals}d})/(1 - r_{fails:0{decimals}d}))")
                for nok in range(twos - successes):
                    file.write(f"*(1-(r_{ones:0{decimals}d} - r_{fails:0{decimals}d})/(1 - r_{fails:0{decimals}d}))")
                file.write(f": ")

                for k in range(1, N + 1):
                    if k <= ones + successes:
                        if k == N:
                            file.write(f"(a{(k-1)}'=1)")
                        else:
                            file.write(f"(a{(k-1)}'=1) & ")
                    elif k == N:
                        file.write(f"(a{(k-1)}'={-ones})")
                    else:
                        file.write(f"(a{(k-1)}'={-ones}) & ")
                if successes < twos:
                    file.write(f" + ")
            file.write(f";\n")

    # all twos transitions
    # file.write(f"       // all twos transition\n")
    # file.write(f"       []   a0 = 2")
    # for i in range(1, N):
    #     file.write(f" & a{i} = 2 ")
    # file.write(f"-> ")
    # for i in range(N - 1):
    #     file.write(f"(a{i}'= 0) & ")
    # file.write(f"(a{(N-1)}'= 0)")
    # file.write(f";\n")

    file.write(f"endmodule \n")
    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print(f"N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(N + 1):
        file.write(f"       a0 = {(1,0)[i==0]}")
        for j in range(1, N):
            # print(f"N",N,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f":{i*i};\n")
    file.write(f"endrewards \n")

    file.close()


def create_properties(N):
    """ Creates property file of reaching respective BSCC of the model of *N* agents as prop_<N>.pctl file.
    For more information see the HSB19 paper.

    Args
    ----------
    N: (int) - agent quantity
    """

    filename = properties_path / Path(f"prop_{N}.pctl")
    file = open(filename, "w")
    print(filename)

    for i in range(1, N + 2):
        if i > 1:
            file.write(f"P=? [ F (a0=1)")
        else:
            file.write(f"P=? [ F (a0=0)")

        for j in range(1, N):
            file.write(f"&(a{j}={(0,1)[i>j+1]})")
        file.write(f"&(b=1)")
        file.write(f"]\n")
    file.close()


def create_rewards_prop():
    """ Creates rewards properties file of moments of reaching BSCC of the model.
    For more information see the HSB19 paper.
    """

    filename = properties_path / Path("moments_2_.pctl")
    file = open(filename, "w")
    print(filename)

    file.write('R{"mean"}=? [ F b=1] \n')
    file.write('R{"mean_squared"}=? [ F b=1] \n')
    file.close()
