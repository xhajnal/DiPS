def call_storm(N,data,alpha,n_samples,multiparam):
    """
    Returns command to call storm with given model and data informed properties
    
    Parameters
    ----------
    N: int number of agents  
    data: map of data    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    multiparam: if True multiparam model is used
    """
    curr_folder=os.getcwd().replace("\\\\","/").replace("\\","/").split("/")[-1]
    
    if multiparam:
        model="multiparam_asynchronous"
        parameters=set()
        for polynome in f_multiparam[N]:
            if len(parameters)<N:
                parameters.update(find_param(polynome))
        
    else:
        model="asynchronous"
        parameters=set()
        for polynome in f[N]:
            if len(parameters)<N:
                parameters.update(find_param(polynome))
    
    parameters = sorted(list(parameters))
    
    print("start=$SECONDS")
    
    suffix=str(N)
    for i in range(len(data[N])):
        # print(data[N][i], "noise: (", data[N][i]-noise(alpha, n_samples, data[N][i]),",",data[N][i]+noise(alpha, n_samples, data[N][i]),")")
        
        if data[N][i]-noise(alpha,n_samples,data[N][i])>0:
            suffix="{}-low".format(i)
            sys.stdout.write('./storm-pars --prism /{}/{}_{}.pm --prop "P>{}'.format(curr_folder,model,N,data[N][i]-noise(alpha,n_samples,data[N][i]))) 
            if i>0:
                sys.stdout.write("[ F (a0=1)")
            else:
                sys.stdout.write("[ F (a0=0)")

            for j in range(1,N):
                sys.stdout.write("&(a"+str(j)+"="+str( 1 if j<i else 0 )+")")
            sys.stdout.write(']"')
            sys.stdout.write(' --region "')
            sys.stdout.write('0.01<={}<=0.99'.format(parameters[0]))
            for param in parameters[1:]:
                sys.stdout.write(',0.01<={}<=0.99'.format(param))      
            sys.stdout.write('" --refine --printfullresult >> /{}/storm_results/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(curr_folder,model,N,alpha,n_samples,suffix))
            
            print()       
            print()
        
        if data[N][i]+noise(alpha,n_samples,data[N][i])<1:
            suffix="{}-high".format(i)
            sys.stdout.write('./storm-pars --prism /{}/{}_{}.pm --prop "P<{}'.format(curr_folder,model,N,data[N][i]+noise(alpha,n_samples,data[N][i])))
            if i>0:
                sys.stdout.write("[ F (a0=1)")
            else:
                sys.stdout.write("[ F (a0=0)")

            for j in range(1,N):
                sys.stdout.write("&(a"+str(j)+"="+str( 1 if j<i else 0 )+")")
            sys.stdout.write(']"')
            sys.stdout.write(' --region "')
            sys.stdout.write('0.01<={}<=0.99'.format(parameters[0]))
            for param in parameters[1:]:
                sys.stdout.write(',0.01<={}<=0.99'.format(param))      
            sys.stdout.write('" --refine --printfullresult >> /{}/storm_results/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(curr_folder,model,N,alpha,n_samples,suffix))
            
            
            print()      
            print()
        print("---")
        print()
    print("end=$SECONDS")    
    print('echo "It took: $((end-start)) seconds." >> /{}/storm_results/{}_{}_{}_{}_seq_{}.txt 2>&1'.format(curr_folder,model,N,alpha,n_samples,suffix))
    print()