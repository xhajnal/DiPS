from src.load import find_param
from src.load import noise
from numpy import prod
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import struct, os, time, socket, re

#import redis

import configparser
config = configparser.ConfigParser()
#print(os.getcwd())
config.read("../config.ini")
#config.sections()
z3_path = config.get("paths", "z3_path")

if not os.path.exists(z3_path):
    raise OSError("Directory does not exist: "+str(z3_path))


os.chdir(z3_path)
from z3 import *

#non_white_area=0
#whole_area=0



def check(thresh,prop,data,alpha,n_samples,silent=False,called=False):
    """ Check if the given region is safe, unsafe, or neither one

    check if for p s.t. p_low < p < p_high and q s.t. q_low < q <q_high 
    the prob reaching i successes f_N(i) is in the interval [D_N[i]*(1-a)-b , D_N[i]*(1+a)+b]

    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise 
    silent: if silent the output is set to minimum
    """
    #initialisation
    if not silent:
        print("checking unsafe",thresh)
        
    if called:
        globals()["parameters"] = set()
        for polynome in prop:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        for param in parameters:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p) 
        if not len(globals()["parameters"])==len(thresh) and not silent:
            print("number of parameters in property ({}) and thresholds ({}) is not equal".format(len(globals()["parameters"]),len(thresh)))

    s = Solver()

    #adding formulas
    for j in range(len(parameters)):
        s.add(globals()[parameters[j]] > thresh[j][0])
        #print(str(globals()[parameters[j]] > thresh[j][0]))
        s.add(globals()[parameters[j]] < thresh[j][1])
        #print(str(globals()[parameters[j]] < thresh[j][1]))
        
    for i in range(0,len(prop)):
        #print("noise is: ", noise(alpha,n_samples,data[i]), "where i=",i)
        
        #if data[i]<100/n_samples:
        #    continue
        
        ##ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        #if  data[i]<0.01:
        #    continue
        
        s.add(eval(prop[i]) > data[i]-noise(alpha,n_samples,data[i]), eval(prop[i]) < data[i]+noise(alpha,n_samples,data[i]))
        #print(str(eval(prop[i]))+">"+str(data[i]-noise(alpha,n_samples,data[i]))+","+str(eval(prop[i]))+"<"+str(data[i]+noise(alpha,n_samples,data[i])))
        #print(prop[i],data[i])

    if s.check() == sat:
        return s.model()
    else:
        add_space=[]    
        for interval in thresh:
            add_space.append(interval[1]-interval[0])
        #print("add_space", add_space)
        globals()["non_white_area"]=globals()["non_white_area"]+prod(add_space)
        #print("area added: ",prod(add_space))
        hyper_rectangles_unsat.append(thresh)
        if len(thresh) == 2: #if two-dim param space
            rectangles_unsat.append(Rectangle((thresh[0][0],thresh[1][0]), thresh[0][1]-thresh[0][0], thresh[1][1]-thresh[1][0], fc='r'))
        if len(thresh) == 1:
            rectangles_unsat.append(Rectangle((thresh[0][0],0.33), thresh[0][1]-thresh[0][0], 0.33, fc='r'))
        #print("red", Rectangle((thresh[0][0],thresh[1][0]), thresh[0][1]-thresh[0][0], thresh[1][1]-thresh[1][0], fc='r'))
        return("unsafe")


def check_safe(thresh,prop,data,alpha,n_samples,silent=False,called=False):
    """ Check if the given region is safe, unsafe, or neither one

    check if for p s.t. p_low < p < p_high and q s.t. q_low < q <q_high 
    the prob reaching i successes f_N(i) is in the interval [D_N[i]*(1-a)-b , D_N[i]*(1+a)+b]

    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    silent: if silent the output is set to minimum
    """
    #initialisation
    if not silent:
        print("checking safe",thresh)
        
    if called:
        globals()["parameters"] = set()
        for polynome in prop:
            globals()["parameters"].update(find_param(polynome))
        globals()["parameters"] = sorted(list(globals()["parameters"]))
        ## EXAMPLE:  parameters >> ['p','q']
        for param in parameters:
            globals()[param] = Real(param)
        ## EXAMPLE: p = Real(p) 
        if not len(globals()["parameters"])==len(thresh) and not silent:
            print("number of parameters in property ({}) and thresholds ({}) is not equal".format(len(globals()["parameters"]),len(thresh)))
    s = Solver()

    #adding formulas
    for j in range(len(parameters)):
        s.add(globals()[parameters[j]] > thresh[j][0])
        s.add(globals()[parameters[j]] < thresh[j][1])
    
    formula = Or(Not(eval(prop[0]) > data[0]-noise(alpha,n_samples,data[0])), Not(eval(prop[0]) < data[0]+noise(alpha,n_samples,data[0]))) 
    
    ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
    #if data[0]<0.01:
    #    formula = Or(Not(eval(prop[0]) > data[0]-noise(alpha,n_samples,data[0])), Not(eval(prop[0]) < data[0]+noise(alpha,n_samples,data[0]))) 
    #else:
    #    formula = False
    
    for i in range(1,len(prop)):
        #if data[i]<100/n_samples:
        #    continue       
        
        ## ALTERNATIVE HEURISTIC APPROACH COMMENTED HERE
        #if  data[i]<0.01:
        #    continue
        formula = Or(formula, Or(Not(eval(prop[i]) > data[i]-noise(alpha,n_samples,data[i])), Not(eval(prop[i]) < data[i]+noise(alpha,n_samples,data[i]))))
    s.add(formula)
    #print(s.check())
    #return s.check()
    if s.check() == unsat:
        add_space=[]    
        for interval in thresh:
            add_space.append(interval[1]-interval[0])
        #print("add_space", add_space)
        globals()["non_white_area"]=globals()["non_white_area"]+prod(add_space)
        #print("area added: ",prod(add_space))
        hyper_rectangles_sat.append(thresh)
        if len(thresh) == 2: #if two-dim param space
            rectangles_sat.append(Rectangle((thresh[0][0],thresh[1][0]), thresh[0][1]-thresh[0][0], thresh[1][1]-thresh[1][0], fc='g'))
        if len(thresh) == 1:
            rectangles_sat.append(Rectangle((thresh[0][0],0.33), thresh[0][1]-thresh[0][0], 0.33, fc='g'))

        #print("green", Rectangle((thresh[0][0],thresh[1][0]), thresh[0][1]-thresh[0][0], thresh[1][1]-thresh[1][0], fc='g'))
        return "safe"
    else:
        return s.model()

class Queue:
  #Constructor creates a list
  def __init__(self):
      self.queue = list()

  #Adding elements to queue
  def enqueue(self,data):
      #Checking to avoid duplicate entry (not mandatory)
      if data not in self.queue:
          self.queue.insert(0,data)
          return True
      return False

  #Removing the last element from the queue
  def dequeue(self):
      if len(self.queue)>0:
          return self.queue.pop()
      return ("Queue Empty!")

  #Getting the size of the queue
  def size(self):
      return len(self.queue)

  #printing the elements of the queue
  def printQueue(self):
      return self.queue

def check_deeper(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent,version):
    """Splitting the parameter space into safe and unsafe regions with respective alg/method
    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computattion
    silent: if silent the output is set to minimum
    """
    
    #initialisation 
    ## region
    globals()["default_region"]=copy.copy(thresh)

    globals()["rectangles_sat"]=[]  
    globals()["rectangles_unsat"]=[]
    globals()["rectangles_unsat_added"]=[]
    
    globals()["hyper_rectangles_sat"]=[] 
    globals()["hyper_rectangles_unsat"]=[]
    globals()["hyper_rectangles_white"]=[thresh]
        
    globals()["non_white_area"]=0
    globals()["whole_area"]=[]
    for interval in thresh:
        globals()["whole_area"].append(interval[1]-interval[0])
    globals()["whole_area"] = prod(globals()["whole_area"])
    
    if not silent:
        print("the area is: ",thresh)
        print("the volume of the whole area is:",globals()["whole_area"])
        
    #params
    globals()["parameters"] = set()
    for polynome in prop:
        globals()["parameters"].update(find_param(polynome))
    globals()["parameters"] = sorted(list(globals()["parameters"]))
    ## EXAMPLE:  parameters >> ['p','q']
    for param in parameters:
        globals()[param] = Real(param)
    ## EXAMPLE: p = Real(p) 
    if not len(globals()["parameters"])==len(thresh) and not silent:
        print("number of parameters in property ({}) and thresholds ({}) is not equal".format(len(globals()["parameters"]),len(thresh)))
    
    #choosing from versions
    start_time = time.time()
    if version==1:
        private_check_deeper(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent)
    if version==2:
        globals()["que"] = Queue()
        private_check_deeper_queue(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent)
    if version==3:
        globals()["que"] = Queue()
        private_check_deeper_queue_checking(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent,None)
    if version==4:
        globals()["que"] = Queue()
        private_check_deeper_queue_checking_both(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent,None)
    if version==5:
        print(check_deeper_iter(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent))
    
    #visualisation
    if len(thresh)==1 or len(thresh)==2:
        colored(globals()["default_region"],thresh)
        #from matplotlib import rcParams
        #rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
        fig = plt.figure()
        pic = fig.add_subplot(111, aspect='equal')
        pic.set_xlabel('p')
        if len(thresh)==2:
            pic.set_ylabel('q') 
        pic.set_title("red = unsafe region, green = safe region, white = in between \n alpha:{}, n_samples:{}, max_recursion_depth:{}, \n min_rec_size:{}, achieved_coverage:{}, alg{} \n It took {} {} second(s)".format(alpha,n_samples,n,epsilon,globals()["non_white_area"]/globals()["whole_area"],version,socket.gethostname(), round(time.time() - start_time, 1)))
        pc = PatchCollection(rectangles_unsat,facecolor='r', alpha=0.5)
        pic.add_collection(pc)
        pc = PatchCollection(rectangles_sat,facecolor='g', alpha=0.5)
        pic.add_collection(pc)
        pc = PatchCollection(rectangles_unsat_added,facecolor='xkcd:grey', alpha=0.5)
        pic.add_collection(pc)
        plt.show()
    print("result coverage is: ", globals()["non_white_area"]/globals()["whole_area"])


def private_check_deeper(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent):
    """
    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computattion
    silent: if silent the output is set to minimum
    """
    
    #checking this:
    #print("check equal", globals()["non_white_area"],non_white_area)
    #print("check equal", globals()["whole_area"],whole_area)
    
    #stop if the given hyperrectangle is to small    
    add_space=[]    
    for interval in thresh:
        add_space.append(interval[1]-interval[0])
    add_space = prod(add_space)
    if (add_space < epsilon):
        if len(thresh)>2:
            #if not silent:
            #    print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            #if not silent:
            #    print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            #if not silent:
            #    print("interval too small, skipped")
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"]/globals()["whole_area"] > coverage:
            return "coverage ",globals()["non_white_area"]/globals()["whole_area"]," is above the threshold"
    
    #HERE MAY ADDING THE MODEL
    if check(thresh,prop,data,alpha,n_samples,silent)=="unsafe":
        result="unsafe"
    elif check_safe(thresh,prop,data,alpha,n_samples,silent)=="safe":
        result="safe"
    else:
        result="unknown"
        
    #print("result",result)
    if (result == "safe" or result == "unsafe"):
        globals()["hyper_rectangles_white"].remove(thresh)
    if n == 0:
        #print("[",p_low,",",p_high ,"],[",q_low,",",q_high ,"]",result)
        if not silent:
            print("maximal recursion reached here with coverage:", globals()["non_white_area"]/globals()["whole_area"])
        return result
    else:
        if not (result == "safe" or result == "unsafe"):#here is necessary to check only 3 of 4, since this line check 1 segment
            # find max interval
            index,maximum = 0,0
            for i in range(len(thresh)):
                value = thresh[i][1]-thresh[i][0]
                if value > maximum:
                    index = i
                    maximum = value
            low = thresh[index][0]
            high = thresh[index][1]
            foo = copy.copy(thresh)
            foo[index] = (low,low+(high-low)/2)
            foo2 = copy.copy(thresh)
            foo2[index] = (low+(high-low)/2,high)
            globals()["hyper_rectangles_white"].remove(thresh)
            globals()["hyper_rectangles_white"].append(foo) #add this region as white
            globals()["hyper_rectangles_white"].append(foo2) #add this region as white
            #print("white area",globals()["hyper_rectangles_white"])
            if silent:
                private_check_deeper(foo,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent)
                if globals()["whole_area"] > 0:
                    if globals()["non_white_area"]/globals()["whole_area"] > coverage:
                        return "coverage ",globals()["non_white_area"]/globals()["whole_area"]," is above the threshold"
                private_check_deeper(foo2,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent)
            else:    
                print(n,foo,globals()["non_white_area"]/globals()["whole_area"],private_check_deeper(foo,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent))
                if globals()["whole_area"] > 0:
                    if globals()["non_white_area"]/globals()["whole_area"] > coverage:
                        return "coverage ",globals()["non_white_area"]/globals()["whole_area"]," is above the threshold"
                print(n,foo2,globals()["non_white_area"]/globals()["whole_area"],private_check_deeper(foo2,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent))                
        
    return result    

def colored(greater,smaller):
    """ Colors outside of the smaller region in the greater region as previously unsat

    Parameters
    ----------
    greater: region in which the smaller region is located
    smaller: smaller region which is not to be colored
    """ 
    #rectangles_sat.append(Rectangle((low_x,low_y), sirka, vyska, fc='g'))
    #print("greater ",greater)
    #print("smaller ",smaller)
    if greater == None or smaller == None:
        return
    globals()["rectangles_unsat_added"].append(Rectangle((greater[0][0],0),smaller[0][0]-greater[0][0] , 1, fc='r'))
    globals()["rectangles_unsat_added"].append(Rectangle((smaller[0][1],0),greater[0][1]-smaller[0][1] , 1, fc='r'))
    globals()["rectangles_unsat_added"].append(Rectangle((smaller[0][0],0),smaller[0][1]-smaller[0][0] , smaller[1][0], fc='r'))
    globals()["rectangles_unsat_added"].append(Rectangle((smaller[0][0],smaller[1][1]),smaller[0][1]-smaller[0][0] ,1- smaller[1][0], fc='r'))  
    
    #re=[]
    #re.append(Rectangle((greater[0][0],0),smaller[0][0]-greater[0][0] , 1, fc='r'))
    #re.append(Rectangle((smaller[0][1],0),greater[0][1]-smaller[0][1] , 1, fc='r'))    
    #re.append(Rectangle((smaller[0][0],0),smaller[0][1]-smaller[0][0] , smaller[1][0], fc='r'))
    #re.append(Rectangle((smaller[0][0],smaller[1][1]),smaller[0][1]-smaller[0][0] ,1- smaller[1][0], fc='r'))   
     
    #fig = plt.figure()
    #pic = fig.add_subplot(111, aspect='equal')
    #pic.set_xlabel('p')
    #pic.set_ylabel('q')
    #pic.set_title("red = unsafe region, green = safe region, white = in between")
    #pc = PatchCollection(re,facecolor='r', alpha=0.5)
    #pic.add_collection(pc)
    
def check_deeper_iter(thresh,props,data,alpha,n_samples,n,epsilon,coverage,silent):
    """New iterative method using alg1

    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computattion
    silent: if silent the output is set to minimum
    """ 
    new_tresh=copy.copy(thresh)
    
    #implement ordering of the props with data
    for i in range(len(props)-1):
        if not silent:
            #print("white: ",globals()["hyper_rectangles_white"])
            print("check_deeper(",new_tresh,[props[i]],[data[i]],")")
        check_deeper(new_tresh,[props[i]],[data[i]],alpha,n_samples,n,epsilon,coverage,True,1)
        
        new_tresh = []
        for interval_index in range(len(thresh)):
            minimum = 9001
            maximum = 0
            #iterate though green regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_sat"])):
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0]<minimum:
                    minimum=globals()["hyper_rectangles_sat"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1]>maximum:
                    maximum=globals()["hyper_rectangles_sat"][rectangle_index][interval_index][1]
            #iterate though white regions to find min and max
            for rectangle_index in range(len(globals()["hyper_rectangles_white"])):
                if globals()["hyper_rectangles_white"][rectangle_index][interval_index][0]<minimum:
                    minimum=globals()["hyper_rectangles_white"][rectangle_index][interval_index][0]
                if globals()["hyper_rectangles_white"][rectangle_index][interval_index][1]>maximum:
                    maximum=globals()["hyper_rectangles_white"][rectangle_index][interval_index][1]
            new_tresh.append((minimum,maximum))
            
        
        if not silent:
            print("Computed hull of nonred region is:",new_tresh)
        #globals()["hyper_rectangles_white"]=[new_tresh]
    default_region=None
    check_deeper(new_tresh,props,data,alpha,n_samples,n,epsilon,coverage,True,1)

def private_check_deeper_queue(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent):
    """
    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computattion
    silent: if silent the output is set to minimum
    """
    #print(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent)
    
    #checking this:
    #print("check equal", globals()["non_white_area"],non_white_area)
    #print("check equal", globals()["whole_area"],whole_area)
    
    #stop if the given hyperrectangle is to small    
    add_space=[]    
    for interval in thresh:
        add_space.append(interval[1]-interval[0])
    add_space = prod(add_space)
    #print("add_space",add_space)
    if (add_space < epsilon):
        if len(thresh)>2:
            if not silent:
                print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            if not silent:
                print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            if not silent:
                print("interval too small, skipped")
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"]/globals()["whole_area"] > coverage:
            globals()["que"] = Queue()
            return "coverage ",globals()["non_white_area"]/globals()["whole_area"]," is above the threshold"
    
    #HERE I CAN APPEND THE VALUE OF EXAMPLE AND COUNTEREXAMPLE
    #print("hello check =",check(thresh,prop,data,alpha,n_samples,silent))
    #print("hello check safe =",check_safe(thresh,prop,data,alpha,n_samples,silent))
    if check(thresh,prop,data,alpha,n_samples,silent)=="unsafe":
        result="unsafe"
    elif check_safe(thresh,prop,data,alpha,n_samples,silent)=="safe":
        result="safe"
    else:
        result="unknown"
        
    if not silent:
        print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],result)
    
    #print("hello")
    if n == 0:
        return
    if (result == "safe" or result == "unsafe"):
        return
    
    #find maximum interval
    index,maximum = 0,0
    for i in range(len(thresh)):
        value = thresh[i][1]-thresh[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = thresh[index][0]
    high = thresh[index][1]
    foo = copy.copy(thresh)
    foo[index] = (low,low+(high-low)/2)
    foo2 = copy.copy(thresh)
    foo2[index] = (low+(high-low)/2,high)
    
    #ADD CALLS TO QUEUE
    #print("adding",[copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent]))
    #print("adding",[copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent]))
    globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
    globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
    
    #CALL QUEUE
    #print(globals()["que"].printQueue())
    while globals()["que"].size()>0:
        private_check_deeper_queue(*que.dequeue())

def private_check_deeper_queue_checking(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent,model=None):    
    """ THIS IS OBSOLETE METHOD, HERE JUST TO BE COPARED WITH THE NEW ONE
    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computattion
    silent: if silent the output is set to minimum
    """
    #print(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent)
    
    #checking this:
    #print("check equal", globals()["non_white_area"],non_white_area)
    #print("check equal", globals()["whole_area"],whole_area)
    
    #stop if the given hyperrectangle is to small    
    add_space=[]    
    for interval in thresh:
        add_space.append(interval[1]-interval[0])
    add_space = prod(add_space)
    #print("add_space",add_space)
    if (add_space < epsilon):
        if len(thresh)>2:
            if not silent:
                print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            if not silent:
                print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            if not silent:
                print("interval too small, skipped")
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"]/globals()["whole_area"] > coverage:
            globals()["que"] = Queue()
            return "coverage ",globals()["non_white_area"]/globals()["whole_area"]," is above the threshold"
          
    if model is None:
        example = check(thresh,prop,data,alpha,n_samples,silent)
        #counterexample = check_safe(thresh,prop,data,alpha,n_samples,silent)
    elif model[0] is None:
        example = check(thresh,prop,data,alpha,n_samples,silent)
    else:
        if not silent:
            print("skipping check_unsafe at", thresh, "since example", model[0])
        example = model[0]


    #resolving the result
    if example == "unsafe":
        if not silent:
            print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],"unsafe") 
        return
    elif check_safe(thresh,prop,data,alpha,n_samples,silent)=="safe":
        if not silent:
            print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],"safe") 
        return
    else: #uknown
        if not silent:
            print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],(example))
    
    if n == 0:
        return 
    
    example_points= re.findall(r'[0-9/]+', str(example))
    #counterexample_points= re.findall(r'[0-9/]+', str(counterexample))
    #print(example_points)
    #print(counterexample_points)
    
    #find maximum dimension an make a cut
    index,maximum = 0,0
    for i in range(len(thresh)):
        value = thresh[i][1]-thresh[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = thresh[index][0]
    high = thresh[index][1]
    foo = copy.copy(thresh)
    foo[index] = (low,low+(high-low)/2)
    foo2 = copy.copy(thresh)
    foo2[index] = (low+(high-low)/2,high)

    model_low  = [9,9]
    model_high = [9,9]
    if float(eval(example_points[index]))>low+(high-low)/2:
        model_low[0]= None
        model_high[0]= example
    else:
        model_low[0]= example
        model_high[0]= None
    #overwrite if equal
    if float(eval(example_points[index]))==low+(high-low)/2:
        model_low[0] = None
        model_high[0] = None
        
    globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent,model_low])
    globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent,model_high])    

    #print(globals()["que"].printQueue())
    while globals()["que"].size()>0:
        private_check_deeper_queue_checking(*que.dequeue())

def private_check_deeper_queue_checking_both(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent,model=None):    
    """
    Parameters
    ----------
    thresh : array of pairs low and high limit of respective param of formula
    prop: array of polynomes
    data: array of int    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    n : max number of recursions to do
    epsilon: minimal size of rectangle to be checked
    cov: coverage threshold to stop computattion
    silent: if silent the output is set to minimum
    """
    #print(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent)
    
    #checking this:
    #print("check equal", globals()["non_white_area"],non_white_area)
    #print("check equal", globals()["whole_area"],whole_area)
    
    #stop if the given hyperrectangle is to small    
    add_space=[]    
    for interval in thresh:
        add_space.append(interval[1]-interval[0])
    add_space = prod(add_space)
    #print("add_space",add_space)
    if (add_space < epsilon):
        if len(thresh)>2:
            if not silent:
                print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            if not silent:
                print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            if not silent:
                print("interval too small, skipped")
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if globals()["whole_area"] > 0:
        if globals()["non_white_area"]/globals()["whole_area"] > coverage:
            globals()["que"] = Queue()
            return "coverage ",globals()["non_white_area"]/globals()["whole_area"]," is above the threshold"
    
    #resolving if the region safe/unsafe/uknown
    if model is None:
        example = check(thresh,prop,data,alpha,n_samples,silent)
        counterexample = check_safe(thresh,prop,data,alpha,n_samples,silent)
    elif model[0] is None:
        example = check(thresh,prop,data,alpha,n_samples,silent)
    else:
        if not silent:
            print("skipping check_unsafe at", thresh, "since example", model[0])
        example=model[0]
    if model is not None:
        if model[1] is None:
            counterexample = check_safe(thresh,prop,data,alpha,n_samples,silent)
        else:
            if not silent:
                print("skipping check_safe at", thresh, "since counterexample", model[1])
            counterexample=model[1]        

    #resolving the result
    if example == "unsafe":
        if not silent:
            print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],"unsafe") 
        return
    elif counterexample == "safe":
        if not silent:
            print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],"safe") 
        return
    else: #uknown
        if not silent:
            print(n,thresh,globals()["non_white_area"]/globals()["whole_area"],(example,counterexample))
    
    if n == 0:
        return 
    
    example_points= re.findall(r'[0-9/]+', str(example))
    counterexample_points= re.findall(r'[0-9/]+', str(counterexample))
    #print(example_points)
    #print(counterexample_points)
    
    #find maximum dimension an make a cut
    index,maximum = 0,0
    for i in range(len(thresh)):
        value = thresh[i][1]-thresh[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = thresh[index][0]
    high = thresh[index][1]
    foo = copy.copy(thresh)
    foo[index] = (low,low+(high-low)/2)
    foo2 = copy.copy(thresh)
    foo2[index] = (low+(high-low)/2,high)

    model_low  = [9,9]
    model_high = [9,9]
    if float(eval(example_points[index]))>low+(high-low)/2:
        model_low[0]= None
        model_high[0]= example
    else:
        model_low[0]= example
        model_high[0]= None
    if float(eval(counterexample_points[index]))>low+(high-low)/2:
        model_low[1]= None
        model_high[1]= counterexample
    else:
        model_low[1]= counterexample
        model_high[1]= None
    #overwrite if equal
    if float(eval(example_points[index]))==low+(high-low)/2:
        model_low[0] = None
        model_high[0] = None
    if float(eval(counterexample_points[index]))==low+(high-low)/2:
        model_low[1] = None
        model_high[1] = None
        
    globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent,model_low])
    globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent,model_high])    

    #print(globals()["que"].printQueue())
    while globals()["que"].size()>0:
        private_check_deeper_queue_checking_both(*que.dequeue())        