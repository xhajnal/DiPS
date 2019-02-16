# create distribution data, by Tanja, 14.1.2019
# edited, by xhajnal, 18.1.2019, still name 'a' is not defined
# edited, by xhajnal, 26.1.2019, a set as [] but does not work
# totally rewritten to not paste empty line, doc added, by xhajnal, 03.02.2019

def generate_all_data_twoparam(p_v=None, q_v=None):
    """ Generates data for all agents_quantities in the current as .csv files    
    """
    import random

    dic_fun = f
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
