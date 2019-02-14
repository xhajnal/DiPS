from numpy import prod
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import struct, os,sys, time, socket, re

from load import find_param
from load import noise

from synthetise import *

#import redis

import configparser
config = configparser.ConfigParser()
print("here ",os.getcwd())
config.read("../config.ini")
#config.sections()
z3_path = config.get("paths", "z3_path")
data_path = config.get("paths", "data")

if not os.path.exists(z3_path):
    raise OSError("Directory does not exist: "+str(z3_path))


#cwd=os.getcwd()
#os.chdir(z3_path)
#from z3 import *
#os.chdir(cwd)


from load import get_f,get_rewards,load_all_prism
f = get_f("./sem*[0-9].txt",True)


import pickle
import copy

p_values = [0.028502714675268215, 0.45223461506339047, 0.8732745414252937, 0.6855555397734584, 0.13075717833714784]
q_values = [0.5057623641293089  , 0.29577906622244676, 0.8440550299528644, 0.8108008054929994, 0.03259111103419188]


from load import load_pickled_data
print(data_path)

print(os.getcwd())
#D = load_pickled_data("Experiments_freq_two_param")
D = pickle.load(open("Experiments_freq_two_param.p","rb"))

#D[(model_type,N,n_sample,v_p,v_q)]

alphas = [0.95]
n_sampless = [3500,1500,100]
depths = [12] 
epsilons = [10e-6]
coverage_threshs = [0.95]
populations=[2,3,5,10]
methods = [1,2,3]
sample_size = 5


import datetime
now = datetime.datetime.now()
outputfile="synthetise_performance_{}_{}.csv".format(now.isoformat().split(".")[0].replace(":","-"),socket.gethostname())
file = open(outputfile, "w")
file.write("semisynchronous/synchronous \n")
print("semisynchronous/synchronous")
file.write(" v_p, v_q, alpha, n_samples, recursion_depth, min_rect_size, population, algorithm, run times{} average time, standard deviation\n".format(","*sample_size))
#file.write("---------------------------------------------------------------------------------------------------------\n")
print(" alpha, n_samples, min_rect_size, recursion_depth, population, algorithm")
for v_p in p_values:
    for v_q in q_values:
        for alpha in alphas:
            for n_samples in n_sampless:
                for epsilon in epsilons:
                    for depth in depths:
                        for coverage_thresh in coverage_threshs:
                            for N in populations:
                                for method in methods:
                                    if N==10 and n_samples==100:
                                        continue
                                    if N==10 and n_samples==1500 and method==1 and data==D:
                                        continue
                                    if N==5 and n_samples==100 and data==D2:
                                        continue
                                    file.write(" {},   {},   {},       {},          {},             {},           {},          {},    ".format(v_p,v_q,alpha,n_samples,depth,epsilon,N,method))
                                    print(" {}    {}   {}     {}         {}               {}           {}          {}    ".format(v_p,v_q,alpha,n_samples,depth,epsilon,N,method))
                                    runs=[]
                                    for run in range(0,sample_size):
                                        start_time = time.time()
                                        check_deeper([(0,1),(0,1)],f[N],D[("synchronous_parallel_",N,n_samples,v_p,v_q)],alpha,n_samples,depth,epsilon,coverage_thresh,True,method)
                                        time_foo= round(time.time() - start_time, 2)
                                        file.write(str(time_foo))
                                        runs.append(time_foo)
                                        file.write(" (cov={}), ".format( round(non_white_area/whole_area,4)))
                                    average=sum(runs)/len(runs)
                                    deviations=[]
                                    for run in range(0,sample_size):
                                        deviations.append((runs[run]-average)*(runs[run]-average))
                                    file.write(" {}, {}".format(average, math.sqrt(sum(deviations)/(sample_size)))) 
                                    file.write("\n")
                                    file.flush()
                                file.write("\n")
            #file.write("---------------------------------------------------------------------------------------------------------\n")
    #file.write("=========================================================================================================\n")
file.close()



data=[D]
alphas = [0.95]
n_sampless = [100]
depths = [12] 
epsilons = [10e-6]
coverage_threshs = [0.95]
populations=[5]
methods = [2,3]
sample_size = 5