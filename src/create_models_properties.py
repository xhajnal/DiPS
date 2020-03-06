import math
from pathlib import Path
import common.mathematics as mymath
from common.config import load_config
from common.state_gen import gen_semisync_statespace

spam = load_config()
model_path = spam["models"]
properties_path = spam["properties"]
del spam


def create_synchronous_model(file, population_size):
    """ Creates synchronous model of *population_size* agents to a *file* with probabilities p and q in [0,1].
    For more information see the HSB19 paper.

    Args:
        file (string): filename with extension
        population_size (int): agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    p (float): - probability to succeed in the first attempt
    q (float): - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + "*p"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")
    file.write(f"const double q;\n")
    file.write(f"\n")

    # module here
    file.write(f"module two_param_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(population_size):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, population_size):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(population_size + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(population_size):
            file.write(f"(a{j}'=" + str(1 if population_size - i > j else 2) + ")")
            if j < population_size - 1:
                file.write(f" & ")
        if i < population_size:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # non-initial transition

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("population_size",population_size,"i",i)
        file.write(f"(a{population_size - 1}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for ones in range(1, population_size):
        file.write(f"       []   a0 = 1")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if ones > j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        twos = population_size - ones
        # file.write(f"twos: {}".format(twos))

        for successes in range(0, twos + 1):
            file.write(str(mymath.nCr(twos, successes)))
            for ok in range(successes):
                file.write(f"*q")
            for nok in range(twos - successes):
                file.write(f"*(1-q)")
            file.write(f": ")

            for k in range(1, population_size + 1):
                if k <= ones + successes:
                    if k == population_size:
                        file.write(f"(a{k - 1}'=1)")
                    else:
                        file.write(f"(a{k - 1}'=1) & ")
                elif k == population_size:
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
    for i in range(1, population_size):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(population_size - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{population_size - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":" + str(i * i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_semisynchronous_model(file, population_size):
    """ Creates semisynchronous model of *population_size* agents to a *file* with probabilities p and q in [0,1]. For more information see the HSB19 paper.

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    p (float): - probability to succeed in the first attempt
    q (float): - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + "*p"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")
    file.write(f"const double q;\n")
    file.write(f"\n")

    # module here
    file.write(f"module two_param_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(population_size):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, population_size):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(population_size + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(population_size):
            file.write(f"(a{j}'=" + str(1 if population_size - i > j else 2) + ")")
            if j < population_size - 1:
                file.write(f" & ")
        if i < population_size:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("population_size",population_size,"i",i)
        file.write(f"(a{population_size - 1}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(population_size - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q:")
        for j in range(population_size):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < population_size - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q:")
        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{population_size - 1}'= 0)")
        # print("population_size",population_size,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    for o in range(1, population_size - 1):
        # file.write(f"help")
        for t in range(1, population_size - o):
            z = population_size - t - o
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
    for i in range(1, population_size):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(population_size - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{population_size - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":" + str(i * i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_asynchronous_model(file, population_size):
    """ Creates asynchronous model of *population_size* agents to a *file* with probabilities p and q in [0,1]. For more information see the HSB19 paper.

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity


    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    p (float): - probability to succeed in the first attempt
    q (float): - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")
    file.write(f"const double q;\n")
    file.write(f"\n")

    # module here
    file.write(f"module two_param_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(population_size):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transitions\n")

    # some -1, some 1
    file.write(f"       // some -1, some 1 transitions\n")
    for i in reversed(range(0, population_size)):
        # for k in range(1,population_size):
        file.write(f"       []   a0 = -1")
        for j in range(1, population_size):
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
        for j in range(i + 1, population_size):
            file.write(f" & (a{j}' = 1)")

        file.write(f" + 1-p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}'= -1)")
        for j in range(max(i, 1), population_size - 1):
            file.write(f" & (a{j}' = 1)")
        file.write(f" & (a{population_size - 1}' = 2)")

        file.write(f";\n")
        # file.write(f"i="+str(i)+" j="+str(j)+" \n")
    file.write(f"\n")

    # some -1, some 2
    file.write(f"       // some -1, some 2 transitions\n")
    for i in reversed(range(0, population_size - 1)):
        # for k in range(1,population_size):
        file.write(f"       []   a0 = -1")
        for j in range(1, population_size):
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
        for j in range(i + 1, population_size):
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
        for j in range(i + 1, population_size):
            file.write(f" & (a{j}' = 2)")

        file.write(f";\n")

    file.write(f"\n")

    # some -1, some 1, some 2
    file.write(f"       // some -1, some 1, some 2 transitions\n")
    for o in range(1, population_size - 1):
        # file.write(f"help")
        for t in range(1, population_size - o):
            z = population_size - t - o
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
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("population_size",population_size,"i",i)
        file.write(f"(a{population_size - 1}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(population_size - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q:")
        for j in range(population_size):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < population_size - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q:")
        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{population_size - 1}'= 0)")
        # print("population_size",population_size,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    for o in range(1, population_size - 1):
        # file.write(f"help")
        for t in range(1, population_size - o):
            z = population_size - t - o
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
    for i in range(1, population_size):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(population_size - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{population_size - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":" + str(i * i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_multiparam_synchronous_model(file, population_size):
    """ Creates synchronous model of *population_size* agents to a *file* with probabilities p and q1, q2, ... q(population_size-1) in [0,1].
    For more information see the HSB19 paper.

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    p (float): - probability to succeed in the first attempt
    q (float): - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + "*p"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")

    for i in range(1, population_size):
        file.write(f"const double q{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(population_size):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, population_size):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(population_size + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(population_size):
            file.write(f"(a{j}'=" + str(1 if population_size - i > j else 2) + ")")
            if j < population_size - 1:
                file.write(f" & ")
        if i < population_size:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # non-initial transition

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("population_size",population_size,"i",i)
        file.write(f"(a{population_size - 1}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for ones in range(1, population_size):
        file.write(f"       []   a0 = 1")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if ones > j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        twos = population_size - ones
        # file.write(f"twos: {}".format(twos))
        for successes in range(0, twos + 1):
            file.write(str(mymath.nCr(twos, successes)))
            for ok in range(successes):
                file.write(f"*q{ones:0{decimals}d}")
            for nok in range(twos - successes):
                file.write(f"*(1-q{ones:0{decimals}d})")
            file.write(f": ")

            for k in range(1, population_size + 1):
                if k <= ones + successes:
                    if k == population_size:
                        file.write(f"(a{k - 1}'=1)")
                    else:
                        file.write(f"(a{k - 1}'=1) & ")
                elif k == population_size:
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
    for i in range(1, population_size):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(population_size - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{population_size - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":" + str(i * i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_multiparam_semisynchronous_model(file, population_size):
    """ Creates semisynchronous model of *population_size* agents to a *file* with probabilities p and q1, q2, ... q(population_size-1) in [0,1].
    For more information see the HSB19 paper.

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    p (float): - probability to succeed in the first attempt
    q (float): - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + "*p"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")

    for i in range(1, population_size):
        file.write(f"const double q{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(population_size):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = -1")
    for i in range(1, population_size):
        file.write(f" & a{i} = -1 ")
    file.write(f"-> ")
    for i in range(population_size + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(population_size):
            file.write(f"(a{j}'=" + str(1 if population_size - i > j else 2) + ")")
            if j < population_size - 1:
                file.write(f" & ")
        if i < population_size:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # some ones, some zeros transitions
    file.write(f"       // some ones, some zeros transitions\n")
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("population_size",population_size,"i",i)
        file.write(f"(a{population_size - 1}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(population_size - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q{i + 1:0{decimals}d}:")
        for j in range(population_size):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < population_size - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q{i + 1:0{decimals}d}:")
        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{population_size - 1}'= 0)")
        # print("population_size",population_size,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    i = 0
    for o in range(1, population_size - 1):
        # file.write(f"help")
        for t in range(1, population_size - o):
            z = population_size - t - o
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
    for i in range(1, population_size):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(population_size - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{population_size - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":" + str(i * i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_multiparam_asynchronous_model(file, population_size):
    """ Creates semisynchronous model of *population_size* agents to a *file* with probabilities p and q1, q2, ... q(population_size-1) in [0,1].
    For more information see the HSB19 paper.

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    p (float): - probability to succeed in the first attempt
    q (float): - probability to succeed in the second attempt
    ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + "*p"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + "*(1-p)"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # start here
    file.write(f"dtmc \n \n")
    file.write(f"const double p;\n")

    for i in range(1, population_size):
        file.write(f"const double q{i:0{decimals}d};\n")
    file.write(f"\n")

    # module here
    file.write(f"module multi_param_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i:  -1:init 0:total_failure 1:success 2:failure_after_first_attempt\n")
    for i in range(population_size):
        file.write(f"       a{i} : [-1..2] init -1; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # transitions here
    # initial transition
    file.write(f"       //  initial transitions\n")

    # some -1, some 1
    file.write(f"       // some -1, some 1 transitions\n")
    for i in reversed(range(0, population_size)):
        # for k in range(1,population_size):
        file.write(f"       []   a0 = -1")
        for j in range(1, population_size):
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
        for j in range(i + 1, population_size):
            file.write(f" & (a{j}' = 1)")

        file.write(f" + 1-p: ")
        if i == 0:
            file.write(f"(a0' = 1)")
        else:
            file.write(f"(a0' = -1)")

        for j in range(1, i):
            file.write(f" & (a{j}'= -1)")
        for j in range(max(i, 1), population_size - 1):
            file.write(f" & (a{j}' = 1)")
        file.write(f" & (a{population_size - 1}' = 2)")

        file.write(f";\n")
        # file.write(f"i="+str(i)+" j="+str(j)+" \n")
    file.write(f"\n")

    # some -1, some 2
    file.write(f"       // some -1, some 2 transitions\n")
    for i in reversed(range(0, population_size - 1)):
        # for k in range(1,population_size):
        file.write(f"       []   a0 = -1")
        for j in range(1, population_size):
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
        for j in range(i + 1, population_size):
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
        for j in range(i + 1, population_size):
            file.write(f" & (a{j}' = 2)")

        file.write(f";\n")

    file.write(f"\n")

    # some -1, some 1, some 2
    file.write(f"       // some -1, some 1, some 2 transitions\n")
    for o in range(1, population_size - 1):
        # file.write(f"help")
        for t in range(1, population_size - o):
            z = population_size - t - o
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
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print("population_size",population_size,"i",i)
        file.write(f"(a{population_size - 1}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # some ones, some twos transitions
    file.write(f"       // some ones, some twos transitions\n")
    for i in range(population_size - 1):
        file.write(f"       []   a0 = 1")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i >= j else 2))
            # print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(f" -> ")

        file.write(f"q{i + 1 :0{decimals}d}:")
        for j in range(population_size):
            file.write(f"(a{j}'= " + str(1 if i + 1 >= j else 2) + ")" + str(" & " if j < population_size - 1 else ""))
        file.write(f" + ")
        file.write(f"1-q{i + 1 :0{decimals}d}:")
        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i >= j else 2) + ") & ")
        file.write(f"(a{population_size - 1}'= 0)")
        # print("population_size",population_size,"i",i)
        file.write(f";\n")
    file.write(f"\n")

    # some ones, some twos transitions, some zeros transitions
    file.write(f"       // some ones, some twos, some zeros transitions\n")
    i = 0
    for o in range(1, population_size - 1):
        # file.write(f"help")
        for t in range(1, population_size - o):
            z = population_size - t - o
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
    for i in range(1, population_size):
        file.write(f" & a{i} = 2 ")
    file.write(f"-> ")
    for i in range(population_size - 1):
        file.write(f"(a{i}'= 0) & ")
    file.write(f"(a{population_size - 1}'= 0)")
    file.write(f";\n")
    file.write(f"endmodule \n")

    file.write(f"\n")

    # rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print("population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {(0, 1)[i > j]}")
        file.write(f":" + str(i * i) + ";\n")
    file.write(f"endrewards \n")

    file.close()


def create_bee_multiparam_synchronous_model(file, population_size):
    """ Creates synchronous model of *population_size* agents to a *file* with probabilities r_i in [0,1].

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    r_i (float): - probability of success of an agent when i amount of pheromone is present
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## Computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + f"*r_{0:0{decimals}d}"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + f"*(1-r_{0:0{decimals}d})"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""
    # print(first_attempt)

    # Model starts here
    file.write(f"dtmc \n \n")

    for i in range(0, population_size):
        file.write(f"const double r_{i:0{decimals}d};\n")
    file.write(f"\n")

    # Module here
    file.write(f"module multi_param_bee_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i: 3:init 1:success -j: failure when j amount of pheromone present \n")
    for i in range(population_size):
        file.write(f"       a{i} : [-{population_size - 1}..3] init 3; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # Transitions here
    # Initial transition
    file.write(f"       //  initial transition\n")
    file.write(f"       []   a0 = 3")
    for i in range(1, population_size):
        file.write(f" & a{i} = 3 ")
    file.write(f" & b = 0 ")
    file.write(f"-> ")
    for i in range(population_size + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(population_size):
            file.write(f"(a{j}'=" + str(1 if population_size - i > j else -0) + ")")
            if j < population_size - 1:
                file.write(f" & ")
        if i < population_size:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # Non-initial transition

    # Some ones, some nonpositive final transitions
    file.write(f"       // some ones, some nonpositive final transitions\n")
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f" & b = 0 ")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print(f"population_size",population_size,"i",i)
        file.write(f"(a{(population_size - 1)}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # Some ones, some nonpositive transitions
    file.write(f"       // some ones, some nonpositive transitions\n")
    for ones in range(1, population_size):
        for fails in range(0, ones):
            file.write(f"       []   a0 = 1")
            for j in range(1, population_size):
                # print(f"population_size",population_size,"i",i,"j",j)
                file.write(f" & a{j} = " + str(1 if ones > j else (-fails)))
                # print(f" & a"+str(j)+" = "+str( 1 if i>=j else 0 ))
            file.write(f" & b = 0 ")
            file.write(f" -> ")

            twos = population_size - ones
            # file.write(f"twos: {}".format(twos))
            for successes in range(0, twos + 1):
                file.write(str(mymath.nCr(twos, successes)))
                for ok in range(successes):
                    file.write(f"* ((r_{ones:0{decimals}d} - r_{fails:0{decimals}d})/(1 - r_{fails:0{decimals}d}))")
                for nok in range(twos - successes):
                    file.write(f"*(1-(r_{ones:0{decimals}d} - r_{fails:0{decimals}d})/(1 - r_{fails:0{decimals}d}))")
                file.write(f": ")

                for k in range(1, population_size + 1):
                    if k <= ones + successes:
                        if k == population_size:
                            file.write(f"(a{(k - 1)}'=1)")
                        else:
                            file.write(f"(a{(k - 1)}'=1) & ")
                    elif k == population_size:
                        file.write(f"(a{(k - 1)}'={-ones})")
                    else:
                        file.write(f"(a{(k - 1)}'={-ones}) & ")
                if successes < twos:
                    file.write(f" + ")
            file.write(f";\n")

    file.write(f"endmodule \n")
    file.write(f"\n")

    # Rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f":{i * i};\n")
    file.write(f"endrewards \n")

    file.close()


def create_bee_multiparam_semisynchronous_model(file, population_size):
    """ Creates semisynchronous model of *population_size* agents to a *file* with probabilities r_i in [0,1].

    Args:
        file (string): filename with extension
        population_size (int):  agent quantity

    Model meaning
    ----------
    Params:
    population_size (int):  number of agents (agents quantity)
    r_i (float): - probability of success of an agent when i amount of pheromone is present
    """
    filename = model_path / Path(file.split(".")[0] + ".pm")
    file = open(filename, "w")
    print(filename)

    ## Computing how many zeros to put
    decimals = math.ceil((math.log(population_size, 10)))

    first_attempt = []
    coefficient = ""

    for i in range(population_size + 1):
        for j in range(population_size - i):
            coefficient = coefficient + f"*r_{0:0{decimals}d}"
        for j in range(population_size - i, population_size):
            coefficient = coefficient + f"*(1-r_{0:0{decimals}d})"
        coefficient = str(mymath.nCr(population_size, i)) + coefficient
        first_attempt.append(coefficient)
        coefficient = ""

    # Model starts here
    file.write(f"dtmc \n \n")

    for i in range(0, population_size):
        file.write(f"const double r_{i:0{decimals}d};\n")
    file.write(f"\n")

    # Module here
    file.write(f"module multi_param_bee_agents_{population_size}\n")
    file.write(f"       // ai - state of agent i: 3:init 1:success -j: failure when j amount of pheromone present \n")
    for i in range(population_size):
        file.write(f"       a{i} : [-{population_size - 1}..3] init 3; \n")
    file.write(f"       b : [0..1] init 0; \n")
    file.write(f"\n")

    # Transitions here
    # Initial transition
    file.write(f"       //  initial transitions\n")
    file.write(f"       []   a0 = 3")
    for i in range(1, population_size):
        file.write(f" & a{i} = 3 ")
    file.write(f" & b = 0 ")
    file.write(f"-> ")
    for i in range(population_size + 1):
        file.write(first_attempt[i] + ": ")
        for j in range(population_size):
            file.write(f"(a{j}'=" + str(1 if population_size - i > j else -0) + ")")
            if j < population_size - 1:
                file.write(f" & ")
        if i < population_size:
            file.write(f" + ")
    file.write(f";\n")
    file.write(f"\n")

    # Non-initial transition

    # Some ones, some nonpositive final transitions
    file.write(f"       // some ones, some nonpositive final transitions\n")
    for i in range(population_size + 1):
        file.write(f"       []   a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f" & b = 0 ")
        file.write(f" -> ")

        for j in range(population_size - 1):
            file.write(f"(a{j}'= " + str(1 if i > j else 0) + ") & ")
        # print(f"population_size",population_size,"i",i)
        file.write(f"(a{(population_size - 1)}'= " + str(1 if i == population_size else 0) + ")")
        file.write(f" & (b'=1);\n")
    file.write(f"\n")

    # Some ones, some nonpositive transitions
    file.write(f"       // some ones, some nonpositive transitions\n")

    ## Get the full semisyn state space
    states = gen_semisync_statespace(population_size)

    for state in states:
        # print(state)
        ## skipping initial state
        if 3 in state:
            continue
        ## skipping all ones state
        if all(map(lambda x: x == 1, state)):
            continue
        ## skipping all zeros state
        if all(map(lambda x: x == 0, state)):
            continue
        ## skipping the rest of final states
        ## if for all non ones the item is equal to ones, it means they cannot update any more
        if all(map(lambda x: x == -state.count(1), list(filter(lambda x: x is not 1, state)))):
            continue

        # print("solving state", state)

        ## Number of stinging bees
        ones = state.count(1)

        ## Distinct bee states which failed
        distinct_fails = list(set(filter(lambda x: x is not 1, state)))
        ## Filter those who can be updated
        # print(distinct_fails)
        distinct_fails = list(filter(lambda x: abs(x) < ones, distinct_fails))
        # print(distinct_fails)

        ## Double checking if at least one guy can be updated
        if not distinct_fails:
            continue

        file.write(f"       []   a0 = {state[0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = {state[j]}")
            # print(f" & a"+str(j)+" = "+str( 1 if i>=j else 0 ))
        file.write(f" & b = 0 ")
        file.write(f" -> ")

        ## Probabilities and successor nodes
        for index, fail_value in enumerate(distinct_fails):
            ## Current state value of the be to be updated
            fail_value = abs(fail_value)
            ## Number of the bees with the same state
            fail_value_count = state.count(fail_value)

            for successes in [True, False]:
                ## When more bees have DIFFERENT state it can be any of them
                ## Putting equal probability of update for all possible updates
                if len(distinct_fails) > 1:
                    file.write(f"1/{len(distinct_fails)} * ")
                ## When more bees have THE SAME state it can be any of them
                ## since we pick just one it is C(fail_value_count, 1) = fail_value_count
                if fail_value_count > 1:
                    file.write(f"{fail_value_count} * ")

                ## The bee either stings (success) or not, hence 2 outgoing states
                if successes:
                    file.write(f"((r_{ones:0{decimals}d} - r_{fail_value:0{decimals}d})/(1 - r_{fail_value:0{decimals}d}))")
                    new_state = list(state)
                    new_state[new_state.index(-fail_value)] = 1
                    new_state.sort(reverse=True)
                    # print("new state", new_state)
                else:
                    file.write(f"(1-(r_{ones:0{decimals}d} - r_{fail_value:0{decimals}d})/(1 - r_{fail_value:0{decimals}d}))")
                    new_state = list(state)
                    new_state[new_state.index(-fail_value)] = - ones
                    new_state.sort(reverse=True)
                    # print("new state", new_state)
                file.write(f": ")

                for indexx, bee in enumerate(new_state):
                    if indexx == len(new_state) - 1:
                        file.write(f"(a{indexx}'={bee})")
                    else:
                        file.write(f"(a{indexx}'={bee})  & ")

                if successes:
                    file.write(f" + ")
            ## If the last possible state update
            if index == len(distinct_fails) - 1:
                file.write(f";\n")
            else:
                file.write(f"+")

    file.write(f"endmodule \n")
    file.write(f"\n")

    # Rewards here
    file.write('rewards "mean" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f":{i};\n")
    file.write(f"endrewards \n")

    file.write('rewards "mean_squared" \n')
    for i in range(population_size + 1):
        file.write(f"       a0 = {(1, 0)[i == 0]}")
        for j in range(1, population_size):
            # print(f"population_size",population_size,"i",i,"j",j)
            file.write(f" & a{j} = " + str(1 if i > j else -i))
        file.write(f":{i * i};\n")
    file.write(f"endrewards \n")

    file.close()

population_size = 3
create_bee_multiparam_semisynchronous_model(f"bee_multiparam_semisynchronous_{population_size}", population_size)


def create_properties(population_size):
    """ Creates property file of reaching respective BSCC of the model of *population_size* agents as prop_<population_size>.pctl file.
    For more information see the HSB19 paper.

    Args:
        population_size (int):  agent quantity
    """

    filename = properties_path / Path(f"prop_{population_size}.pctl")
    file = open(filename, "w")
    print(filename)

    for i in range(1, population_size + 2):
        if i > 1:
            file.write(f"P=? [ F (a0=1)")
        else:
            file.write(f"P=? [ F (a0=0)")

        for j in range(1, population_size):
            file.write(f"&(a{j}={(0, 1)[i > j + 1]})")
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
