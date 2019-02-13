import numpy as np
import pylab as pl
from matplotlib import collections  as mc

import os,sys
workspace = os.path.dirname(__file__)
sys.path.append(os.path.join(workspace, '../src/'))
#sys.path.append(os.path.dirname(__file__))
from load import find_param

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def eval_and_show(fun_list,parameter_value):
    """ Creates bar plot of probabilities of i successes for given parametrisation

    Parameters
    ----------
    fun_list : list of polynomes
    N: dictionary key
    parameter_value: array of param values 
    """
    for polynome in fun_list:
        parameters=set()
        parameters.update(find_param(polynome))
    parameters= sorted(list(parameters))
    #print(parameters)

    title=""
    a=[]
    for param in range(len(parameters)):
        #print(parameters[param])
        #print(parameter_value[param])
        globals()[parameters[param]]= parameter_value[param]
        title = "{}{}={} ".format(title,parameters[param],parameter_value[param])
    #print("eval ", polynome, eval(polynome))
    for polynome in fun_list:
        a.append(eval(polynome))
    #print(a)
    fig, ax = plt.subplots()
    width = 0.2
    ax.set_ylabel('Probability')
    ax.set_xlabel('i')
    ax.set_title('{}'.format(title))
    rects1 = ax.bar(range(len(fun_list)), a, width, color='b')
    plt.show()
    return a

def sample(dic_fun,agents_quantities,size_q):
    """ Returns probabilities of i successes for sampled parametrisations

    Parameters
    ----------
    dic_fun : dictionary N -> list of polynomes
    size_q : sample size in each parameter

    Returns array of [N,i,parameters,value]
    """
    arr = []
    for N in agents_quantities:
        for polynome in dic_fun[N]:
            parameters=set()
            if len(parameters)<N:
                parameters.update(find_param(polynome))
            #print(parameters)
            parameters= sorted(list(parameters))
            #print(parameters)
            parameter_values = []
            for param in range(len(parameters)):
                parameter_values.append(np.linspace(0, 1, size_q, endpoint=True)) 
            parameter_values=cartesian_product(*parameter_values)
            if (len(parameters)-1)==0:
                parameter_values=np.linspace(0, 1, size_q, endpoint=True)[np.newaxis, :].T
            #print(parameter_values)
            
            for parameter_value in parameter_values:
                #print(parameter_value)
                a=[N,dic_fun[N].index(polynome)]
                for param in range(len(parameters)):
                    a.append(parameter_value[param])
                    #print(parameters[param])
                    #print(parameter_value[param])
                    globals()[parameters[param]]= parameter_value[param]
                #print("eval ", polynome, eval(polynome))
                a.append(eval(polynome))
                arr.append(a)
    return arr

def visualise(dic_fun,agents_quantities,size_q):
    """ Creates bar plot of probabilities of i successes for sampled parametrisation

    Parameters
    ----------
    dic_fun : dictionary N -> list of polynomes
    size_q : sample size in each parameter
    """
    for N in agents_quantities:
        parameters = set()
        for polynome in dic_fun[N]:
            if len(parameters)<N:
                parameters.update(find_param(polynome))
        #print(parameters)
        parameters = sorted(list(parameters))
        #print(parameters)
        parameter_values = []
        for param in range(len(parameters)):
            parameter_values.append(np.linspace(0, 1, size_q, endpoint=True)) 
        parameter_values=cartesian_product(*parameter_values)
        if (len(parameters)-1)==0:
            parameter_values=np.linspace(0, 1, size_q, endpoint=True)[np.newaxis, :].T
        #print(parameter_values)
            
        for parameter_value in parameter_values:
            #print(parameter_value)
            a=[N,dic_fun[N].index(polynome)]
            title=""
            for param in range(len(parameters)):
                a.append(parameter_value[param])
                #print(parameters[param])
                #print(parameter_value[param])
                globals()[parameters[param]]= parameter_value[param]
                title = "{}{}={} ".format(title,parameters[param],parameter_value[param])
            #print("eval ", polynome, eval(polynome))
            for polynome in dic_fun[N]:
                a.append(eval(polynome))
            
            print(a)
            fig, ax = plt.subplots()
            width = 0.2
            ax.set_ylabel('Probability')
            ax.set_xlabel('i')
            ax.set_title('N={} {}'.format(N,title))
            rects1 = ax.bar(range(N+1), a[len(parameters)+2:], width, color='b')
            plt.show()

def visualise_byparam(hyper_rectangles):
    """
    Visualise intervals of each dimension in plot.
    
    Parameters
    ----------
    hyper_rectangles : list of hyperrectangles
    """
    
    #https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
    if hyper_rectangles:
        lines=[]
        for i in range(len(hyper_rectangles[0])):
            for j in range(len(hyper_rectangles)):
                #print(hyper_rectangles_sat[j][i])
                lines.append([(i+1, hyper_rectangles[j][i][0]), (i+1, hyper_rectangles[j][i][1])])
                #print([(i+1, hyper_rectangles_sat[j][i][0]), (i+1, hyper_rectangles_sat[j][i][1])])
        c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

        lc = mc.LineCollection(lines, color='g',linewidths=2)
        

        fig, ax = pl.subplots()

        ax.set_xlabel('params')
        ax.set_ylabel('parameter value')
        ax.set_title("intervals in which are parameter in green regions")
        
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
    else:
        print("No green areas to be visualised")

def heatmap(fun,region,sampling_sizes):
    """ Creates 2D heatmap plot of sampled points of given function

    Parameters
    ----------
    dic_fun : dictionary N -> list of polynomes
    region: boundaries of parameter space to be sampled
    sampling_sizes : tuple of sample size of respective parameter

    Example
    ----------
    heatmap("p+q",[[0,1],[3,4]],[5,5])
    """

    parameters = sorted(list(find_param(fun)))
    #print(parameters)
    if len(parameters) != 2:
        raise InvalidFunction("Number of paramters of given function is not equal to 2 but",len(parameters))

    arr = np.zeros((sampling_sizes[0]*sampling_sizes[1],3))

    #f = lambda locals()[parameters[0]),locals()[parameters[1]): fun

    ii=-1
    jj=-1
    for i in np.linspace(region[0][0], region[0][1], sampling_sizes[0], endpoint=True):
        ii += 1
        #print("ii: ",ii)
        locals()[parameters[0]]=i
        for j in np.linspace(region[1][0], region[1][1], sampling_sizes[1], endpoint=True):
            jj += 1
            #print("jj: ",jj)
            locals()[parameters[1]]=j
            arr[jj,0] = round(i,2)
            arr[jj,1] = round(j,2)
            arr[jj,2] = eval(fun)
    #print(arr)
    #d = pd.DataFrame(arr, columns=["p","q","E"])
    d = pd.DataFrame(arr, columns=[parameters[0],parameters[1],"E"])
    #d = d.pivot("p", "q", "E")  
    d = d.pivot(parameters[0], parameters[1], "E")
    ax = sns.heatmap(d)
    plt.show()

if __name__ == "__main__":
    #heatmap("p+0*q",[[1,5],[1,5]],[6,6])
    heatmap("p+q",[[0,1],[0,1]],[5,5])