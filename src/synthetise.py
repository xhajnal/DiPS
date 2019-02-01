from src.load import find_param
from src.load import noise
from numpy import prod
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import struct, os, time, socket

#import redis


z3_path = "C:/z3py/z3-4.6.0-x64-win/bin/python"  #THUNDER
#z3_path = "C:/z3-4.6.0-x86-win/bin/python"  #SPICY
# z3_path = "C:/z3/z3-4.6.0-x64-win/bin/python" # NELEUS
#z3_path = "/home/matej/z3/build/python" #Freya

os.chdir(z3_path)
from z3 import *

#non_white_area=0
#whole_area=0



def check(thresh,prop,data,alpha,n_samples,silent=False):
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
    #if not silent:
    #    print("checking unsafe",thresh)
    parameters = set()
    for polynome in prop:
        parameters.update(find_param(polynome))
    parameters = sorted(list(parameters))
    if not len(parameters)==len(thresh) and not silent:
        print("number of parameters in property(",len(parameters),") and thresholds(",len(thresh),") is not equal")
    
    for param in parameters:
        globals()[param] = Real(param)
    s = Solver()

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
        model = s.model()
        if check_safe(thresh,prop,data,alpha,n_samples,silent)=="safe":
            return("safe")
        return(model)
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


def check_safe(thresh,prop,data,alpha,n_samples,silent=False):
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
    
    #if not silent:
    #    print("checking safe",thresh)
    parameters = set()
    for polynome in prop:
        parameters.update(find_param(polynome))
    parameters = sorted(list(parameters))
    if not len(parameters)==len(thresh) and not silent:
        print("number of parameters in property(",len(parameters),") and thresholds(",len(thresh),") is not equal")    
    for param in parameters:
        globals()[param] = Real(param)
    s = Solver()

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
        return s.check()

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
    
    #initialisation
    globals()["rectangles_sat"]=[]  
    globals()["rectangles_unsat"]=[]
    
    globals()["hyper_rectangles_sat"]=[] 
    globals()["hyper_rectangles_unsat"]=[]
    globals()["hyper_rectangles_white"]=[thresh]
        
    globals()["non_white_area"]=0
    globals()["whole_area"]=[]
    for interval in thresh:
        whole_area.append(interval[1]-interval[0])
    globals()["whole_area"] = prod(whole_area)
    
    if not silent:
        print("the area is: ",thresh)
        print("the volume of the whole area is:",whole_area)
        

    
    #choosing from versions
    start_time = time.time()
    if version==1:
        print(private_check_deeper(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent))
    if version==2:
        globals()["que"] = Queue()
        private_check_deeper_queue(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent)
    if version==3:
        globals()["que"] = Queue()
        private_check_deeper_queue_checking(thresh,prop,data,alpha,n_samples,n,epsilon,cov,silent,None)
    
    if len(thresh)==1 or len(thresh)==2:
        #from matplotlib import rcParams
        #rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
        fig = plt.figure()
        pic = fig.add_subplot(111, aspect='equal')
        pic.set_xlabel('p')
        if len(thresh)==2:
            pic.set_ylabel('q') 
        pic.set_title("red = unsafe region, green = safe region, white = in between \n alpha:{}, n_samples:{}, max_recursion_depth:{}, \n min_rec_size:{}, achieved_coverage:{}, alg{} \n It took {} {} second(s)".format(alpha,n_samples,n,epsilon,non_white_area/whole_area,version,socket.gethostname(), round(time.time() - start_time, 1)))
        pc = PatchCollection(rectangles_unsat,facecolor='r', alpha=0.5)
        pic.add_collection(pc)
        pc = PatchCollection(rectangles_sat,facecolor='g', alpha=0.5)
        pic.add_collection(pc)

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
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            return "rectangle too small, skipped"
        else:
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if whole_area > 0:
        if non_white_area/whole_area > coverage:
            return "coverage ",non_white_area/whole_area," is above the threshold"
    
    result = check(thresh,prop,data,alpha,n_samples,silent)
    if n == 0:
        #print("[",p_low,",",p_high ,"],[",q_low,",",q_high ,"]",result)
        if not silent:
            print("maximal recursion reached here with coverage:", non_white_area/whole_area)
        return result
    else:
        if not (result == "safe" or result == "unsafe"):#here is necessary to check only 3 of 4, since this line check 1 segment
            globals()["hyper_rectangles_white"].append(thresh) #add this region as white
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
            if silent:
                private_check_deeper(foo,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent)
                if whole_area > 0:
                    if non_white_area/whole_area > coverage:
                        return "coverage ",non_white_area/whole_area," is above the threshold"
                private_check_deeper(foo2,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent)
            else:    
                print(n,foo,non_white_area/whole_area,private_check_deeper(foo,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent))
                if whole_area > 0:
                    if non_white_area/whole_area > coverage:
                        return "coverage ",non_white_area/whole_area," is above the threshold"
                print(n,foo2,non_white_area/whole_area,private_check_deeper(foo2,prop,data,alpha,n_samples,n-1,epsilon,coverage,silent))                
        
    return result    

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
            print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            print("interval too small, skipped")
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if whole_area > 0:
        if non_white_area/whole_area > coverage:
            globals()["que"] = Queue()
            return "coverage ",non_white_area/whole_area," is above the threshold"
    
    #print("hello1")
    result = check(thresh,prop,data,alpha,n_samples,silent)
    #print("hello")
    if n == 0:
        print(n,thresh,non_white_area/whole_area,result)
    else:
        if not (result == "safe" or result == "unsafe"):#here is necessary to check only 3 of 4, since this line check 1 segment
            # find max interval
            print(n,thresh,non_white_area/whole_area,result)
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
            #print("adding",[copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent]))
            #print("adding",[copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent], "with len", len([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent]))
            globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
            globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
            #print(globals()["que"].printQueue())
            while globals()["que"].size()>0:
                private_check_deeper_queue(*que.dequeue())
        else:
            print(n,thresh,non_white_area/whole_area,result)   
        
def private_check_deeper_queue_checking(thresh,prop,data,alpha,n_samples,n,epsilon,coverage,silent,model=None):    
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
            #print("hyperrectangle too small, skipped")
            return "hyperrectangle too small, skipped"
        elif len(thresh)==2:
            #print("rectangle too small, skipped")
            return "rectangle too small, skipped"
        else:
            #print("interval too small, skipped")
            return "interval too small, skipped"

    #stop if the the current coverage is above the given thresholds
    if whole_area > 0:
        if non_white_area/whole_area > coverage:
            globals()["que"] = Queue()
            return "coverage ",non_white_area/whole_area," is above the threshold"
    
    #print("hello1")
    if model is None:
        result = check(thresh,prop,data,alpha,n_samples,silent)
    else:
        print("skipping check at", thresh, "since model", model)
        result = check_safe(thresh,prop,data,alpha,n_samples,silent)
    #print("hello")

    if n == 0:
        print(n,thresh,non_white_area/whole_area,result)
    else:
        if (result == "safe" or result == "unsafe"): 
            model=None
            print(n,thresh,non_white_area/whole_area,result) 
        else:
            # find max interval
            print(n,thresh,non_white_area/whole_area,result)
            #print("result",result)
            if str(result)=="sat":
                model=None
            else:
                model= re.findall(r'[0-9/]+', str(result))
                #print("model",model)
                
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
            

            if model is None:
                globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
                globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
            elif float(eval(model[index]))>low+(high-low)/2:
                #print(model[index])
                #print(low+(high-low))
                globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
                globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent,model])
            else:
                #print(model[index])
                #print(low+(high-low))
                globals()["que"].enqueue([copy.copy(foo),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent,model])
                globals()["que"].enqueue([copy.copy(foo2),prop,data,alpha,n_samples,n-1,epsilon,coverage,silent])
            #print(globals()["que"].printQueue())
            while globals()["que"].size()>0:
                private_check_deeper_queue_checking(*que.dequeue())

