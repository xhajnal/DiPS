import glob, re,sys,math
from sympy import factor
import scipy.stats as st
sys.path.append("..")


def load_all_prism(path,factorize=True, rewards_only=False, f_only=False):
    """ Loads all results of parameter synthesis in *path* folder into two maps - f list of rational functions for each property, and rewards list of rational functions for each reward
    
    Parameters
    ----------
    path: string - file name regex
    factorize: if true it will factorise polynomial results 
    rewards_only: if true it compute only rewards
    f_only: if true it will compute only standard properties
    
    Returns:
        (f,reward), where
        f: dictionary N -> list of rational functions for each property
        rewards: dictionary N -> list of rational functions for each reward
    """

    here = False
    N = 0
    f = {}
    rewards = {}
    for file in glob.glob(path):
        #print(file)
        N = int(re.findall('\d+', file )[0])  
        file=open(file,"r")
        i=-1
        here=""
        f[N] = []
        rewards[N] = []
        for line in file:
            if line.startswith( 'Parametric model checking:' ):
                i=i+1 
            if line.startswith( 'Parametric model checking: R=?' ):
                here="r"
            if i>=0 and line.startswith( 'Result' ):
                line = line.split(":")[2]
                line = line.replace("{", "")
                line = line.replace("}", "")
                line = line.replace("p", "* p")
                line = line.replace("q", "* q")
                line = line.replace("**", "*")
                line = line.replace("* *", "*")
                line = line.replace("*  *", "*")
                line = line.replace("+ *", "+")
                line = line.replace("^", "**")
                line = line.replace(" ", "")
                if line.startswith( '*' ):
                    line=line[1:]
                if here=="r" and not f_only:
                    if factorize:
                        try:
                            rewards[N].append(str(factor(line[:-1])))
                        except:
                            print("Error while factorising rewards, used not factorised instead")
                            rewards[N].append(line[:-1])
                    else:
                        rewards[N]=line[:-1]
                elif not here=="r" and not rewards_only:
                    #print(f[N])
                    #print(line[:-1])
                    if factorize:
                        try:
                            f[N].append(str(factor(line[:-1])))
                        except:
                            print("Error while factorising polynome f[{}][{}], used not factorised instead".format(N,i))
                            f[N]=line[:-1]
                    else:
                        f[N].append(line[:-1])
    return(f,rewards)

def get_f(path,factorize):
    return load_all_prism(path,factorize,False,True)[0]

def get_rewards(path,factorize):
    return load_all_prism(path,factorize,True,False)[1]

def load_all_data(path):
    """ loads all experimental data for respective property, returns as dictionary D
    
    Parameters
    ----------
    path: string - file name regex
    
    Returns:
        D: dictionary N -> list of propbabilities for respective property
    """
    D = {}
    for file in glob.glob(path):
        file=open(file,"r")
        N = 0
        for line in file:
            #print("line: ",line)
            if re.search("n", line) is not None:
                N = int(line.split(",")[0].split("=")[1])
                #print("N, ",N)
                D[N] = []
                continue
            D[N] = line[:-1].split(",")
            #print(D[N])
            for value in range(len(D[N])):
                #print(D[N][value])
                try:
                    D[N][value]=float(D[N][value])
                except:
                    print("error while parsing N=",N," i=",value," of value=",D[N][value])
                    D[N][value]=0
                #print(type(D[N][value]))
            #D[N].append(1-sum(D[N]))
            break
            #print(D[N])
    if D:
        return D
    else:
        print("Error, No data loaded, please check path")

def catch_data_error(data,minimum,maximum):
    """ Corrects all data value to be in range min,max
    
    Parameters
    ----------
    data: map structure of data
    minimum: minimal value in data to be set to
    maximum: maximal value in data to be set to
    
    """
    for n in data.keys():
        for i in range(len(data[n])):
            if data[n][i]<minimum:
                data[n][i]=minimum
            if data[n][i]>maximum:
                data[n][i]=maximum

def noise(alpha, n_samples, data):
    """ Estimates expected interval with respect to parameters
    TBA shortly describe this type of margin

    Parameters
    ----------
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise 
    data: data point
    """
    return st.norm.ppf(1-(1-alpha)/2)*math.sqrt(data*(1-data)/n_samples)+0.5/n_samples

def noise_experimental(alpha, n_samples, data):
    """ Estimates expected interval with respect to parameters
    This noise was used to produce the visual outputs for hsb19 

    Parameters
    ----------
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise 
    data: data point
    """
    return st.norm.ppf(1-(1-alpha)/2)*math.sqrt(data*(1-data)/n_samples)+0.5/n_samples+0.005

def find_param(polynome):
    """ Finds parameters of a polynomes

    Parameters
    ----------
    polynome : polynome as string
    
    Returns set of strings - parameters
    """
    parameters = polynome.replace('(', '').replace(')', '').replace('**', '*').replace(' ', '')
    parameters = re.split('\+|\*|\-|/',parameters)
    parameters = [i for i in parameters if not i.isnumeric()]
    parameters = set(parameters)
    parameters.add("")
    parameters.remove("")
    #print("hello",set(parameters))
    return set(parameters)
