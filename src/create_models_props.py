import os,math

from pathlib import Path
import configparser
config = configparser.ConfigParser()
#print(os.getcwd())
workspace = os.path.dirname(__file__)
#print("workspace",workspace)
config.read(os.path.join(workspace,"../config.ini"))
#config.sections()
model_path = Path(config.get("paths", "models"))
if not os.path.exists(model_path):
    os.makedirs(model_path)

properties_path = Path(config.get("paths", "properties"))
if not os.path.exists(properties_path):
    os.makedirs(properties_path)

def nCr(n,k):
    """ Return conbinatorial number n take k
    
    Parameters
    ----------
    n : int
    k : int
    """
    
    f = math.factorial
    return f(n) / f(k) / f(n-k)

def create_synchronous_model(file,N):
    """ Creates synchronous model of *N* agents to a *file* with probabilities p and q in [0,1]. For more information see paper.
    
    Parameters
    ----------
    file : string - filename with extesion
    N : int - agent quantity
    """
    filename = model_path / Path(file.split(".")[0]+".pm") 
    file = open(filename,"w") 
    print(filename)
    
    first_attempt = []
    coeficient = ""

    for i in range(N+1):
        for j in range(N-i):
            coeficient = coeficient + "*p"
        for j in range(N-i,N):
            coeficient = coeficient + "*(1-p)"
        coeficient = str(nCr(N,i))+ coeficient
        first_attempt.append(coeficient)
        coeficient = ""
    #print(first_attempt)

    # start here 
    file.write("dtmc \n \n") 
    file.write("const double p;\n")
    file.write("const double q;\n")
    file.write("\n" )

    # module here 
    file.write("module bees_"+str(N)+"\n" )
    file.write("       // ai - state of agent i:  -1:init 0:total_failure 1:succes 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write("       a"+str(i)+" : [-1..2] init -1; \n")
    file.write("       b : [0..1] init 0; \n")
    file.write("\n" )

    # transitions here 
    # initial transition
    file.write("       //  initial transition\n")
    file.write("       []   a0 = -1")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = -1 ")
    file.write("-> ")
    for i in range(N+1):
        file.write( first_attempt[i]+": " )
        for j in range(N):
            file.write("(a"+str(j)+"'="+ str( 1 if N-i>j else 2 )+")")
            if j<N-1:
                file.write(" & ")    
        if i<N:
            file.write(" + " )
    file.write(";\n")
    file.write("\n")
    
    # non-initial transition
    
    # some ones, some zeros transitions
    file.write("       // some ones, some zeros transitions\n")
    for i in range(N+1):
        file.write("       []   a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(" -> ")    

        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>j else 0 ) +") & ")
        # print("N",N,"i",i)
        file.write("(a"+str(N-1)+"'= "+str( 1 if i == N else 0 )+")")
        file.write(" & (b'=1);\n")
    file.write("\n")


    # some ones, some twos transitions
    file.write("       // some ones, some twos transitions\n")
    for ones in range(1,N):
        file.write("       []   a0 = 1")
        for j in range(1,N):
            #print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if ones>j else 2 ))
            #print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(" -> ")    
        
        twos=N-ones
        #file.write("twos: {}".format(twos))
        
        for sucessses in range(0,twos+1):
            file.write(str(nCr(twos,sucessses)))
            for ok in range(sucessses):
                file.write("*q")
            for nok in range(twos-sucessses):
                file.write("*(1-q)")
            file.write(": ")
            
            for k in range(1,N+1):
                if k<=ones+sucessses:
                    if k==N:
                        file.write("(a"+str(k-1)+"'=1)")
                    else:
                        file.write("(a"+str(k-1)+"'=1) & ")
                elif k==N:
                    file.write("(a"+str(k-1)+"'=0)")
                else:
                    file.write("(a"+str(k-1)+"'=0) & ")
            if sucessses<twos:
                file.write(" + " ) 

        file.write(";\n")
    file.write("\n")

    # all twos transitions
    file.write("       // all twos transition\n")
    file.write("       []   a0 = 2")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = 2 ")
    file.write("-> ")
    for i in range(N-1):
        file.write("(a"+str(i)+"'= 0) & ")
    file.write("(a"+str(N-1)+"'= 0)")
    file.write(";\n")
    file.write("endmodule \n") 

    file.write("\n")

    # rewards here 
    file.write('rewards "coin_flips" \n')
    for i in range(N+1):
        file.write("       a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(":"+str(i)+";\n")
    file.write("endrewards \n")
    file.close()


def create_semisynchronous_model(file,N):
    """ Creates semisynchronous model of *N* agents to a *file* with probabilities p and q in [0,1]. For more information see paper.
    
    Parameters
    ----------
    file : string - filename with extesion
    N : int - agent quantity
    """
    filename = model_path / Path(file.split(".")[0]+".pm")  
    file = open(filename,"w")  
    print(filename)
    
    first_attempt = []
    coeficient = ""

    for i in range(N+1):
        for j in range(N-i):
            coeficient = coeficient + "*p"
        for j in range(N-i,N):
            coeficient = coeficient + "*(1-p)"
        coeficient = str(nCr(N,i))+ coeficient
        first_attempt.append(coeficient)
        coeficient = ""
    #print(first_attempt)

    # start here 
    file.write("dtmc \n \n") 
    file.write("const double p;\n")
    file.write("const double q;\n")
    file.write("\n" )

    # module here 
    file.write("module bees_"+str(N)+"\n" )
    file.write("       // ai - state of agent i:  -1:init 0:total_failure 1:succes 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write("       a"+str(i)+" : [-1..2] init -1; \n")
    file.write("       b : [0..1] init 0; \n")
    file.write("\n" )

    # transitions here 
    # initial transition
    file.write("       //  initial transition\n")
    file.write("       []   a0 = -1")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = -1 ")
    file.write("-> ")
    for i in range(N+1):
        file.write( first_attempt[i]+": " )
        for j in range(N):
            file.write("(a"+str(j)+"'="+ str( 1 if N-i>j else 2 )+")")
            if j<N-1:
                file.write(" & ")    
        if i<N:
            file.write(" + " )
    file.write(";\n")
    file.write("\n")

    # some ones, some zeros transitions
    file.write("       // some ones, some zeros transitions\n")
    for i in range(N+1):
        file.write("       []   a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(" -> ")    

        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>j else 0 ) +") & ")
        # print("N",N,"i",i)
        file.write("(a"+str(N-1)+"'= "+str( 1 if i == N else 0 )+")")
        file.write(" & (b'=1);\n")
    file.write("\n")


    # some ones, some twos transitions
    file.write("       // some ones, some twos transitions\n")
    for i in range(N-1):
        file.write("       []   a0 = 1")
        for j in range(1,N):
            #print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
            #print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(" -> ")    

        file.write("q:")
        for j in range(N):
            file.write("(a"+str(j)+"'= "+str( 1 if i+1>=j else 2 ) +")"+str( " & " if j<N-1 else ""))
        file.write(" + ")
        file.write("1-q:")    
        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>=j else 2 ) +") & ")
        file.write("(a"+str(N-1)+"'= 0)")
        #print("N",N,"i",i)
        file.write(";\n")
    file.write("\n")


    # some ones, some twos transitions, some zeros transitions
    file.write("       // some ones, some twos, some zeros transitions\n")
    for o in range(1,N-1):
        #file.write("help")
        for t in range(1,N-o):
            z=N-t-o
            file.write("       []   a0 = 1")
            for j in range(1,o):
                file.write(" & a"+str(j)+" = 1")
            for j in range(o,o+t):
                file.write(" & a"+str(j)+" = 2")
            for j in range(o+t,o+t+z):
                file.write(" & a"+str(j)+" = 0")

            file.write(" -> ")    
            file.write("q: (a0' = 1)")
            for j in range(1,o+1):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o+1,o+t):
                file.write(" & (a"+str(j)+"'= 2)")
            for j in range(o+t,o+t+z):
                file.write(" & (a"+str(j)+"'= 0)")

            file.write(" + ")
            file.write("1-q: (a0' = 1)")
            for j in range(1,o):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o,o+t-1):
                file.write(" & (a"+str(j)+"'= 2)")
            for j in range(o+t-1,o+t+z):
                file.write(" & (a"+str(j)+"'= 0)")
            file.write(";\n")
            
            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write("\n")

    # all twos transitions
    file.write("       // all twos transition\n")
    file.write("       []   a0 = 2")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = 2 ")
    file.write("-> ")
    for i in range(N-1):
        file.write("(a"+str(i)+"'= 0) & ")
    file.write("(a"+str(N-1)+"'= 0)")
    file.write(";\n")
    file.write("endmodule \n") 

    file.write("\n")

    # rewards here 
    file.write('rewards "coin_flips" \n')
    for i in range(N+1):
        file.write("       a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(":"+str(i)+";\n")
    file.write("endrewards \n")
    file.close()

def create_asynchronous_model(file,N):
    """ Creates aynchronous model of *N* agents to a *file* with probabilities p and q in [0,1]. For more information see paper.
    
    Parameters
    ----------
    file : string - filename with extesion
    N : int - agent quantity
    
    
    Model meaning
    ----------
    params:
    N - number of agents (agents quantity)
    p - proprability to succeed in the first attempt
    q - proprability to succeed when getting help
    ai - state of agent i:  -1:init, 0:total_failure, 1:succes, 2:failure_after_first_attempt
    """
    filename = model_path / Path(file.split(".")[0]+".pm")  
    file = open(filename,"w")  
    print(filename)
    
    # start here 
    file.write("dtmc \n \n") 
    file.write("const double p;\n")
    file.write("const double q;\n")
    file.write("\n" )

    # module here 
    file.write("module bees_"+str(N)+"\n" )
    file.write("       // ai - state of agent i:  -1:init 0:total_failure 1:succes 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write("       a"+str(i)+" : [-1..2] init -1; \n")
    file.write("       b : [0..1] init 0; \n")
    file.write("\n" )

    # transitions here 
    # initial transition
    file.write("       //  initial transitions\n")
    
    
    # some -1, some 1
    file.write("       // some -1, some 1 transitions\n")
    for i in reversed(range(0,N)):
        #for k in range(1,N):
        file.write("       []   a0 = -1")
        for j in range(1,N):
            if j>i:
                file.write(" & a"+str(j)+" = 1 ")
            else:
                file.write(" & a"+str(j)+" = -1 ")
        file.write("-> ")

        file.write( "p: " )
        if i==0:
            file.write("(a0' = 1)")
        else:
            file.write("(a0' = -1)")

        for j in range(1,i):
            file.write(" & (a"+str(j)+"' = -1)")
        if i>0:
            file.write(" & (a"+str(i)+"' = 1)")       
        for j in range(i+1,N):
            file.write(" & (a"+str(j)+"' = 1)")

        file.write( " + 1-p: " )
        if i==0:
            file.write("(a0' = 1)")
        else:
            file.write("(a0' = -1)")
            
        for j in range(1,i):
            file.write(" & (a"+str(j)+"'= -1)")
        for j in range( max(i,1),N-1):
            file.write(" & (a"+str(j)+"' = 1)")
        file.write(" & (a"+str(N-1)+"' = 2)")
        
        file.write(";\n")
        # file.write("i="+str(i)+" j="+str(j)+" \n")
    file.write("\n")
    
    # some -1, some 2
    file.write("       // some -1, some 2 transitions\n")
    for i in reversed(range(0,N-1)):
        #for k in range(1,N):
        file.write("       []   a0 = -1")
        for j in range(1,N):
            if j>i:
                file.write(" & a"+str(j)+" = 2")
            else:
                file.write(" & a"+str(j)+" = -1")
        file.write("-> ")

        file.write( "p: " )
        if i==0:
            file.write("(a0' = 1)")
        else:
            file.write("(a0' = -1)")

        for j in range(1,i):
            file.write(" & (a"+str(j)+"' = -1)")
        if i>0:
            file.write(" & (a"+str(i)+"' = 1)")       
        for j in range(i+1,N):
            file.write(" & (a"+str(j)+"' = 2)")

        file.write( " + 1-p: " )
        if i==0:
            file.write("(a0' = 2)")
        else:
            file.write("(a0' = -1)")        
        for j in range(1,i):
            file.write(" & (a"+str(j)+"'= -1)")
        if i>0:
            file.write(" & (a"+str(i)+"' = 2)")       
        for j in range(i+1,N):
            file.write(" & (a"+str(j)+"' = 2)")

        file.write(";\n")
        
    file.write("\n")
    
    # some -1, some 1, some 2
    file.write("       // some -1, some 1, some 2 transitions\n")
    for o in range(1,N-1):
        #file.write("help")
        for t in range(1,N-o):
            z=N-t-o
            file.write("       []   a0 = -1")
            for j in range(1,o):
                file.write(" & a"+str(j)+" = -1")
            for j in range(o,o+t):
                file.write(" & a"+str(j)+" = 1")
            for j in range(o+t,o+t+z):
                file.write(" & a"+str(j)+" = 2")

            file.write(" -> ")    
            if o>1:
                file.write("p: (a0' = -1)")
            else:
                file.write("p: (a0' = 1)")
            for j in range(1,o-1):
                file.write(" & (a"+str(j)+"'= -1)")
            for j in range(max(1,o-1),o+t):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o+t,o+t+z):
                file.write(" & (a"+str(j)+"'= 2)")

            file.write(" + ")
            if o>1:
                file.write("1-p: (a0' = -1)")
            else:
                file.write("1-p: (a0' = 1)")
            for j in range(1,o-1):
                file.write(" & (a"+str(j)+"'= -1)")
            for j in range(max(1,o-1),o+t-1):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o+t-1,o+t+z):
                file.write(" & (a"+str(j)+"'= 2)")
            file.write(";\n")
    file.write("\n")
    
    # not initial transition
    file.write("       //  not initial transitions\n")

    # some ones, some zeros transitions
    file.write("       // some ones, some zeros transitions\n")
    for i in range(N+1):
        file.write("       []   a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(" -> ")    

        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>j else 0 ) +") & ")
        # print("N",N,"i",i)
        file.write("(a"+str(N-1)+"'= "+str( 1 if i == N else 0 )+")")
        file.write(" & (b'=1);\n")
    file.write("\n")


    # some ones, some twos transitions
    file.write("       // some ones, some twos transitions\n")
    for i in range(N-1):
        file.write("       []   a0 = 1")
        for j in range(1,N):
            #print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
            #print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(" -> ")    

        file.write("q:")
        for j in range(N):
            file.write("(a"+str(j)+"'= "+str( 1 if i+1>=j else 2 ) +")"+str( " & " if j<N-1 else ""))
        file.write(" + ")
        file.write("1-q:")    
        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>=j else 2 ) +") & ")
        file.write("(a"+str(N-1)+"'= 0)")
        #print("N",N,"i",i)
        file.write(";\n")
    file.write("\n")


    # some ones, some twos transitions, some zeros transitions
    file.write("       // some ones, some twos, some zeros transitions\n")
    for o in range(1,N-1):
        #file.write("help")
        for t in range(1,N-o):
            z=N-t-o
            file.write("       []   a0 = 1")
            for j in range(1,o):
                file.write(" & a"+str(j)+" = 1")
            for j in range(o,o+t):
                file.write(" & a"+str(j)+" = 2")
            for j in range(o+t,o+t+z):
                file.write(" & a"+str(j)+" = 0")

            file.write(" -> ")    
            file.write("q: (a0' = 1)")
            for j in range(1,o+1):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o+1,o+t):
                file.write(" & (a"+str(j)+"'= 2)")
            for j in range(o+t,o+t+z):
                file.write(" & (a"+str(j)+"'= 0)")

            file.write(" + ")
            file.write("1-q: (a0' = 1)")
            for j in range(1,o):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o,o+t-1):
                file.write(" & (a"+str(j)+"'= 2)")
            for j in range(o+t-1,o+t+z):
                file.write(" & (a"+str(j)+"'= 0)")
            file.write(";\n")
            
            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write("\n")

    # all twos transitions
    file.write("       // all twos transition\n")
    file.write("       []   a0 = 2")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = 2 ")
    file.write("-> ")
    for i in range(N-1):
        file.write("(a"+str(i)+"'= 0) & ")
    file.write("(a"+str(N-1)+"'= 0)")
    file.write(";\n")
    file.write("endmodule \n") 

    file.write("\n")

    # rewards here 
    file.write('rewards "coin_flips" \n')
    for i in range(N+1):
        file.write("       a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(":"+str(i)+";\n")
    file.write("endrewards \n")
    file.close()

def create_multiparam_synchronous_model(file,N):
    """ Creates synchronous model of *N* agents to a *file* with probabilities p and q in [0,1]. For more information see paper.
    
    Parameters
    ----------
    file : string - filename with extesion
    N : int - agent quantity
    """
    filename = model_path / Path(file.split(".")[0]+".pm")  
    file = open(filename,"w")  
    print(filename)
    
    first_attempt = []
    coeficient = ""

    for i in range(N+1):
        for j in range(N-i):
            coeficient = coeficient + "*p"
        for j in range(N-i,N):
            coeficient = coeficient + "*(1-p)"
        coeficient = str(nCr(N,i))+ coeficient
        first_attempt.append(coeficient)
        coeficient = ""
    #print(first_attempt)

    # start here 
    file.write("dtmc \n \n") 
    file.write("const double p;\n")

    for i in range(1,N):
        file.write("const double q"+str(i)+";\n")
    file.write("\n" )
    
    # module here 
    file.write("module bees_"+str(N)+"\n" )
    file.write("       // ai - state of agent i:  -1:init 0:total_failure 1:succes 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write("       a"+str(i)+" : [-1..2] init -1; \n")
    file.write("       b : [0..1] init 0; \n")
    file.write("\n" )

    # transitions here 
    # initial transition
    file.write("       //  initial transition\n")
    file.write("       []   a0 = -1")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = -1 ")
    file.write("-> ")
    for i in range(N+1):
        file.write( first_attempt[i]+": " )
        for j in range(N):
            file.write("(a"+str(j)+"'="+ str( 1 if N-i>j else 2 )+")")
            if j<N-1:
                file.write(" & ")    
        if i<N:
            file.write(" + " )
    file.write(";\n")
    file.write("\n")
    
    # non-initial transition
    
     # some ones, some zeros transitions
    file.write("       // some ones, some zeros transitions\n")
    for i in range(N+1):
        file.write("       []   a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(" -> ")    

        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>j else 0 ) +") & ")
        # print("N",N,"i",i)
        file.write("(a"+str(N-1)+"'= "+str( 1 if i == N else 0 )+")")
        file.write(" & (b'=1);\n")
    file.write("\n")


    # some ones, some twos transitions
    file.write("       // some ones, some twos transitions\n")
    for ones in range(1,N):
        file.write("       []   a0 = 1")
        for j in range(1,N):
            #print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if ones>j else 2 ))
            #print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(" -> ")    
        
        twos=N-ones
        #file.write("twos: {}".format(twos))
        for sucessses in range(0,twos+1):
            file.write(str(nCr(twos,sucessses)))
            for ok in range(sucessses):
                file.write("*q{}".format(ones))
            for nok in range(twos-sucessses):
                file.write("*(1-q{})".format(ones))
            file.write(": ")
            
            for k in range(1,N+1):
                if k<=ones+sucessses:
                    if k==N:
                        file.write("(a"+str(k-1)+"'=1)")
                    else:
                        file.write("(a"+str(k-1)+"'=1) & ")
                elif k==N:
                    file.write("(a"+str(k-1)+"'=0)")
                else:
                    file.write("(a"+str(k-1)+"'=0) & ")
            if sucessses<twos:
                file.write(" + " ) 

        file.write(";\n")
    file.write("\n")

    # all twos transitions
    file.write("       // all twos transition\n")
    file.write("       []   a0 = 2")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = 2 ")
    file.write("-> ")
    for i in range(N-1):
        file.write("(a"+str(i)+"'= 0) & ")
    file.write("(a"+str(N-1)+"'= 0)")
    file.write(";\n")
    file.write("endmodule \n") 

    file.write("\n")

    # rewards here 
    file.write('rewards "coin_flips" \n')
    for i in range(N+1):
        file.write("       a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(":"+str(i)+";\n")
    file.write("endrewards \n")
    file.close()

def create_multiparam_semisynchronous_model(file,N):
    """ Creates semisynchronous model of *N* agents to a *file* with probabilities p and multiple q-s in [0,1].
    For more information see paper.
    
    Parameters
    ----------
    file : string - filename with extesion
    N : int - agent quantity
    """
    filename = model_path / Path(file.split(".")[0]+".pm")  
    file = open(filename,"w")  
    print(filename)

    first_attempt = []
    coeficient = ""

    for i in range(N+1):
        for j in range(N-i):
            coeficient = coeficient + "*p"
        for j in range(N-i,N):
            coeficient = coeficient + "*(1-p)"
        coeficient = str(nCr(N,i))+ coeficient
        first_attempt.append(coeficient)
        coeficient = ""
    #print(first_attempt)

    # start here 
    file.write("dtmc \n \n") 
    file.write("const double p;\n")


    for i in range(1,N):
        file.write("const double q"+str(i)+";\n")
    file.write("\n" )

    # module here 
    file.write("module multiparam_bees_"+str(N)+"\n" )
    file.write("       // ai - state of agent i:  -1:init 0:total_failure 1:succes 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write("       a"+str(i)+" : [-1..2] init -1; \n")
    file.write("       b : [0..1] init 0; \n")
    file.write("\n" )

    # transitions here 
    # initial transition
    file.write("       //  initial transition\n")
    file.write("       []   a0 = -1")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = -1 ")
    file.write("-> ")
    for i in range(N+1):
        file.write( first_attempt[i]+": " )
        for j in range(N):
            file.write("(a"+str(j)+"'="+ str( 1 if N-i>j else 2 )+")")
            if j<N-1:
                file.write(" & ")    
        if i<N:
            file.write(" + " )
    file.write(";\n")
    file.write("\n")

    # some ones, some zeros transitions
    file.write("       // some ones, some zeros transitions\n")
    for i in range(N+1):
        file.write("       []   a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(" -> ")    

        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>j else 0 ) +") & ")
        # print("N",N,"i",i)
        file.write("(a"+str(N-1)+"'= "+str( 1 if i == N else 0 )+")")
        file.write(" & (b'=1);\n")
    file.write("\n")


    # some ones, some twos transitions
    file.write("       // some ones, some twos transitions\n")
    for i in range(N-1):
        file.write("       []   a0 = 1")
        for j in range(1,N):
            #print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
            #print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(" -> ")    

        file.write("q"+str(i+1)+":")
        for j in range(N):
            file.write("(a"+str(j)+"'= "+str( 1 if i+1>=j else 2 ) +")"+str( " & " if j<N-1 else ""))
        file.write(" + ")
        file.write("1-q"+str(i+1)+":")    
        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>=j else 2 ) +") & ")
        file.write("(a"+str(N-1)+"'= 0)")
        #print("N",N,"i",i)
        file.write(";\n")
    file.write("\n")


    # some ones, some twos transitions, some zeros transitions
    file.write("       // some ones, some twos, some zeros transitions\n")
    i=0
    for o in range(1,N-1):
        #file.write("help")
        for t in range(1,N-o):
            z=N-t-o
            i=i+1
            file.write("       []   a0 = 1")
            for j in range(1,o):
                file.write(" & a"+str(j)+" = 1")
            for j in range(o,o+t):
                file.write(" & a"+str(j)+" = 2")
            for j in range(o+t,o+t+z):
                file.write(" & a"+str(j)+" = 0")

            file.write(" -> ")    
            file.write("q"+str(o)+": (a0' = 1)")
            for j in range(1,o+1):
                file.write(" & (a"+str(j)+"' = 1)")
            for j in range(o+1,o+t):
                file.write(" & (a"+str(j)+"' = 2)")
            for j in range(o+t,o+t+z):
                file.write(" & (a"+str(j)+"' = 0)")

            file.write(" + ")
            file.write("1-q"+str(o)+": (a0' = 1)")
            for j in range(1,o):
                file.write(" & (a"+str(j)+"' = 1)")
            for j in range(o,o+t-1):
                file.write(" & (a"+str(j)+"' = 2)")
            for j in range(o+t-1,o+t+z):
                file.write(" & (a"+str(j)+"' = 0)")
            file.write(";\n")

            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write("\n")


    #all twos transition
    file.write("       // all twos transition\n")
    file.write("       []   a0 = 2")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = 2 ")
    file.write("-> ")
    for i in range(N-1):
        file.write("(a"+str(i)+"'= 0) & ")
    file.write("(a"+str(N-1)+"'= 0)")
    file.write(";\n")
    file.write("endmodule \n") 

    file.write("\n")


    # rewards here 
    file.write('rewards "coin_flips" \n')
    for i in range(N+1):
        file.write("       a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(":"+str(i)+";\n")
    file.write("endrewards \n")
    file.close()

def create_multiparam_asynchronous_model(file,N):
    """ Creates semisynchronous model of *N* agents to a *file* with probabilities p and multiple q-s in [0,1].
    For more information see paper.
    
    Parameters
    ----------
    file : string - filename with extesion
    N : int - agent quantity
    """
    filename = model_path / Path(file.split(".")[0]+".pm")  
    file = open(filename,"w")  
    print(filename)
    
    first_attempt = []
    coeficient = ""

    for i in range(N+1):
        for j in range(N-i):
            coeficient = coeficient + "*p"
        for j in range(N-i,N):
            coeficient = coeficient + "*(1-p)"
        coeficient = str(nCr(N,i))+ coeficient
        first_attempt.append(coeficient)
        coeficient = ""
    #print(first_attempt)

    # start here 
    file.write("dtmc \n \n") 
    file.write("const double p;\n")


    for i in range(1,N):
        file.write("const double q"+str(i)+";\n")
    file.write("\n" )

    # module here 
    file.write("module multiparam_bees_"+str(N)+"\n" )
    file.write("       // ai - state of agent i:  -1:init 0:total_failure 1:succes 2:failure_after_first_attempt\n")
    for i in range(N):
        file.write("       a"+str(i)+" : [-1..2] init -1; \n")
    file.write("       b : [0..1] init 0; \n")
    file.write("\n" )

    # transitions here 
    # initial transition
    file.write("       //  initial transitions\n")
    
    
    # some -1, some 1
    file.write("       // some -1, some 1 transitions\n")
    for i in reversed(range(0,N)):
        #for k in range(1,N):
        file.write("       []   a0 = -1")
        for j in range(1,N):
            if j>i:
                file.write(" & a"+str(j)+" = 1 ")
            else:
                file.write(" & a"+str(j)+" = -1 ")
        file.write("-> ")

        file.write( "p: " )
        if i==0:
            file.write("(a0' = 1)")
        else:
            file.write("(a0' = -1)")

        for j in range(1,i):
            file.write(" & (a"+str(j)+"' = -1)")
        if i>0:
            file.write(" & (a"+str(i)+"' = 1)")       
        for j in range(i+1,N):
            file.write(" & (a"+str(j)+"' = 1)")

        file.write( " + 1-p: " )
        if i==0:
            file.write("(a0' = 1)")
        else:
            file.write("(a0' = -1)")
            
        for j in range(1,i):
            file.write(" & (a"+str(j)+"'= -1)")
        for j in range( max(i,1),N-1):
            file.write(" & (a"+str(j)+"' = 1)")
        file.write(" & (a"+str(N-1)+"' = 2)")
        
        file.write(";\n")
        # file.write("i="+str(i)+" j="+str(j)+" \n")
    file.write("\n")
    
    # some -1, some 2
    file.write("       // some -1, some 2 transitions\n")
    for i in reversed(range(0,N-1)):
        #for k in range(1,N):
        file.write("       []   a0 = -1")
        for j in range(1,N):
            if j>i:
                file.write(" & a"+str(j)+" = 2")
            else:
                file.write(" & a"+str(j)+" = -1")
        file.write("-> ")

        file.write( "p: " )
        if i==0:
            file.write("(a0' = 1)")
        else:
            file.write("(a0' = -1)")

        for j in range(1,i):
            file.write(" & (a"+str(j)+"' = -1)")
        if i>0:
            file.write(" & (a"+str(i)+"' = 1)")       
        for j in range(i+1,N):
            file.write(" & (a"+str(j)+"' = 2)")

        file.write( " + 1-p: " )
        if i==0:
            file.write("(a0' = 2)")
        else:
            file.write("(a0' = -1)")        
        for j in range(1,i):
            file.write(" & (a"+str(j)+"'= -1)")
        if i>0:
            file.write(" & (a"+str(i)+"' = 2)")       
        for j in range(i+1,N):
            file.write(" & (a"+str(j)+"' = 2)")

        file.write(";\n")
        
    file.write("\n")
    
    # some -1, some 1, some 2
    file.write("       // some -1, some 1, some 2 transitions\n")
    for o in range(1,N-1):
        #file.write("help")
        for t in range(1,N-o):
            z=N-t-o
            file.write("       []   a0 = -1")
            for j in range(1,o):
                file.write(" & a"+str(j)+" = -1")
            for j in range(o,o+t):
                file.write(" & a"+str(j)+" = 1")
            for j in range(o+t,o+t+z):
                file.write(" & a"+str(j)+" = 2")

            file.write(" -> ")    
            if o>1:
                file.write("p: (a0' = -1)")
            else:
                file.write("p: (a0' = 1)")
            for j in range(1,o-1):
                file.write(" & (a"+str(j)+"'= -1)")
            for j in range(max(1,o-1),o+t):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o+t,o+t+z):
                file.write(" & (a"+str(j)+"'= 2)")

            file.write(" + ")
            if o>1:
                file.write("1-p: (a0' = -1)")
            else:
                file.write("1-p: (a0' = 1)")
            for j in range(1,o-1):
                file.write(" & (a"+str(j)+"'= -1)")
            for j in range(max(1,o-1),o+t-1):
                file.write(" & (a"+str(j)+"'= 1)")
            for j in range(o+t-1,o+t+z):
                file.write(" & (a"+str(j)+"'= 2)")
            file.write(";\n")
    file.write("\n")

    # some ones, some zeros transitions
    file.write("       // some ones, some zeros transitions\n")
    for i in range(N+1):
        file.write("       []   a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(" -> ")    

        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>j else 0 ) +") & ")
        # print("N",N,"i",i)
        file.write("(a"+str(N-1)+"'= "+str( 1 if i == N else 0 )+")")
        file.write(" & (b'=1);\n")
    file.write("\n")


    # some ones, some twos transitions
    file.write("       // some ones, some twos transitions\n")
    for i in range(N-1):
        file.write("       []   a0 = 1")
        for j in range(1,N):
            #print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
            #print(" & a"+str(j)+" = "+str( 1 if i>=j else 2 ))
        file.write(" -> ")    

        file.write("q"+str(i+1)+":")
        for j in range(N):
            file.write("(a"+str(j)+"'= "+str( 1 if i+1>=j else 2 ) +")"+str( " & " if j<N-1 else ""))
        file.write(" + ")
        file.write("1-q"+str(i+1)+":")    
        for j in range(N-1):
            file.write("(a"+str(j)+"'= "+str( 1 if i>=j else 2 ) +") & ")
        file.write("(a"+str(N-1)+"'= 0)")
        #print("N",N,"i",i)
        file.write(";\n")
    file.write("\n")


    # some ones, some twos transitions, some zeros transitions
    file.write("       // some ones, some twos, some zeros transitions\n")
    i=0
    for o in range(1,N-1):
        #file.write("help")
        for t in range(1,N-o):
            z=N-t-o
            i=i+1
            file.write("       []   a0 = 1")
            for j in range(1,o):
                file.write(" & a"+str(j)+" = 1")
            for j in range(o,o+t):
                file.write(" & a"+str(j)+" = 2")
            for j in range(o+t,o+t+z):
                file.write(" & a"+str(j)+" = 0")

            file.write(" -> ")    
            file.write("q"+str(o)+": (a0' = 1)")
            for j in range(1,o+1):
                file.write(" & (a"+str(j)+"' = 1)")
            for j in range(o+1,o+t):
                file.write(" & (a"+str(j)+"' = 2)")
            for j in range(o+t,o+t+z):
                file.write(" & (a"+str(j)+"' = 0)")

            file.write(" + ")
            file.write("1-q"+str(o)+": (a0' = 1)")
            for j in range(1,o):
                file.write(" & (a"+str(j)+"' = 1)")
            for j in range(o,o+t-1):
                file.write(" & (a"+str(j)+"' = 2)")
            for j in range(o+t-1,o+t+z):
                file.write(" & (a"+str(j)+"' = 0)")
            file.write(";\n")

            # print("ones: "+str(o)," twos: "+str(t)," zeros: "+str(z))
    file.write("\n")


    #all twos transition
    file.write("       // all twos transition\n")
    file.write("       []   a0 = 2")
    for i in range(1,N):
        file.write(" & a"+str(i)+" = 2 ")
    file.write("-> ")
    for i in range(N-1):
        file.write("(a"+str(i)+"'= 0) & ")
    file.write("(a"+str(N-1)+"'= 0)")
    file.write(";\n")
    file.write("endmodule \n") 

    file.write("\n")


    # rewards here 
    file.write('rewards "coin_flips" \n')
    for i in range(N+1):
        file.write("       a0 = "+str( 0 if i==0 else 1 ))
        for j in range(1,N):
            # print("N",N,"i",i,"j",j)
            file.write(" & a"+str(j)+" = "+str( 1 if i>j else 0 ))
        file.write(":"+str(i)+";\n")
    file.write("endrewards \n")
    file.close()

def create_properties(N):
    """ Creates property file of reaching each BSCC of the model of *N* agents as prop_<N>.pctl file.
    For more information see paper.
    
    Parameters
    ----------
    N : int - agent quantity
    """
    
    filename = properties_path / Path("prop_"+str(N)+".pctl")
    file = open(filename,"w") 
    print(filename)
    
    for i in range(1,N+2):
        if i>1:
            file.write("P=? [ F (a0=1)")
        else:
            file.write("P=? [ F (a0=0)")

        for j in range(1,N):
            file.write("&(a"+str(j)+"="+str( 1 if i>j+1 else 0 )+")")
        file.write("]\n")
    file.write("R=? [ F b=1] \n")
    file.close()