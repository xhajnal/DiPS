import time
import glob,os
def call_prism(args, seq):
    """  Solves problem of calling prism from another directory.
    
    Parameters
    ----------
    args: string for executing prism
    seq: if true it will take properties by one and append results (neccesary if out of the memory)
    """
    filename = args.split()[0].split(".")[0]+str(".txt")
    filename = os.path.join("prism_results",filename)
    curr_dir = os.getcwd()
    os.chdir(prism_path)
    #print(os.getcwd())
    prism_args = []
    
    try:
        #print(args.split(" "))  

        args=args.split(" ")
        #print(args)
        propfile=args[1]
        #print(propfile)
        for arg in args:
            #print(arg)
            #print(re.compile('\.[a-z]').search(arg))
            if re.compile('\.[a-z]').search(arg) is not None:
                prism_args.append(os.path.join(curr_dir,arg))
                #print(prism_args)
            else:
                prism_args.append(arg)
        #print(prism_args)
        #prism_args.append(" ".join(args.split(" ")[-2:]))
        #print(prism_args)
        
        #print(sys.platform)
        if sys.platform.startswith("win"):
            args=["prism.bat"]
        else:    
            args=["prism"]
        args.extend(prism_args)

        if seq:
            with open(os.path.join(curr_dir,filename), 'a') as f:  
                with open(os.path.join(curr_dir,propfile), 'r') as prop:
                    args.append("-property")                
                    args.append("")
                    prop=prop.readlines()
                    for i in range(1,len(prop)+1):
                        args[-1]= str(i)
                        #print(args)
                        output = subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.decode("utf-8") 
                        #print(output)
                        f.write(output)
        else:    
            with open(os.path.join(curr_dir,filename), 'w') as f:
                #print(args)
                output = subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.decode("utf-8") 
                #print(output)
                f.write(output)
    finally:
        os.chdir(curr_dir)

def call_prism_files(file_prefix,multiparam,agents_quantities,seq=False,noprobchecks=False):
    if noprobchecks:
        noprobchecks='-noprobchecks '
    else:
        noprobchecks=""
    for N in agents_quantities:
        for file in glob.glob(file_prefix+str(N)+".pm"):
            start_time = time.time()
            print("{} seq={}{}".format(file,seq,noprobchecks))
            if multiparam:
                q=""
                for i in range(1,N):
                    q="{},q{}=0:1".format(q,i)
                    #q=q+",q"+str(i)"=0:1"
            else:
                q=",q=0:1"
            #print("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q))
            call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q), seq)
            if not seq: 
                #if 'GC overhead' in tailhead.tail(open('prism_results/{}.txt'.format(file.split('.')[0])),40).read():
                if 'GC overhead' in open('prism_results/{}.txt'.format(file.split('.')[0])).read():
                    seq = True
                    print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
                    start_time = time.time()
                    call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q), False)
            if not noprobchecks:
                if '-noprobchecks' in open('prism_results/{}.txt'.format(file.split('.')[0])).read():
                    print("An error occured, running with noprobchecks option")
                    noprobchecks='-noprobchecks '
                    print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")
                    start_time = time.time()
                    call_prism("{} prop_{}.pctl {}-param p=0:1{}".format(file,N,noprobchecks,q), False)
            print("  It took", socket.gethostname(), time.time() - start_time, "seconds to run")

