def call_data_informed_prism(N,data,alpha,n_samples,multiparam,seq):
    """
    Creates data informed properties.
    
    Parameters
    ----------
    N: int number of agents  
    data: map of data    
    alpha : confidence interval to compute noise
    n_samples : number of samples to compute noise
    multiparam: if True multiparam model is used
    """
    if multiparam:
        model="multiparam_semisynchronous_parallel"
        prop="_multiparam"
        parameters=set()
        for polynome in f_multiparam[N]:
            if len(parameters)<N:
                parameters.update(find_param(polynome))
        
    else:
        model="semisynchronous_parallel"
        prop=""
        parameters=set()
        for polynome in f[N]:
            if len(parameters)<N:
                parameters.update(find_param(polynome))
    
    parameters = sorted(list(parameters))
    if seq:
        print("start=$SECONDS")
        j=1
        
        for i in range(len(data[N])):
            #print(data[N][i], "noise: (", data[N][i]-noise(alpha, n_samples, data[N][i]),",",data[N][i]+noise(alpha, n_samples, data[N][i]),")")
            if data[N][i]-noise(alpha,n_samples,data[N][i])>0:
                sys.stdout.write('prism ../{}_{}.pm '.format(model,N)) 
                sys.stdout.write('../prop{}_{}_{}_{}_seq.pctl '.format(prop,N,alpha,n_samples)) 
                sys.stdout.write('-property {}'.format(j))
                j=j+1
                sys.stdout.write(' -param "')
                sys.stdout.write('{}=0:1'.format(parameters[0]))
                for param in parameters[1:]:
                    sys.stdout.write(',{}=0:1'.format(param))      
                sys.stdout.write('" >> {}_{}_{}_{}_seq.txt 2>&1'.format(model,N,alpha,n_samples))

                print()       
                print()

            if data[N][i]+noise(alpha,n_samples,data[N][i])<1:
                sys.stdout.write('prism ../{}_{}.pm '.format(model,N)) 
                sys.stdout.write('../prop{}_{}_{}_{}_seq.pctl '.format(prop,N,alpha,n_samples)) 
                sys.stdout.write('-property {}'.format(j)) 
                j=j+1
                sys.stdout.write(' -param "')
                sys.stdout.write('{}=0:1'.format(parameters[0]))
                for param in parameters[1:]:
                    sys.stdout.write(',{}=0:1'.format(param))      
                sys.stdout.write('" >> {}_{}_{}_{}_seq.txt 2>&1'.format(model,N,alpha,n_samples))

                print()      
                print()
            print("---")
            print()
        print("end=$SECONDS")    
        print('echo "It took: $((end-start)) seconds." >> {}_{}_{}_{}_seq.txt 2>&1'.format(model,N,alpha,n_samples))
            
    else:
        sys.stdout.write('(time prism ../{}_{}.pm '.format(model,N)) 
        sys.stdout.write('../prop{}_{}_{}_{}.pctl '.format(prop,N,alpha,n_samples)) 
        sys.stdout.write('-param {}=0:1'.format(parameters[0]))
        for param in parameters[1:]:
                    sys.stdout.write(',{}=0:1'.format(param))   
        sys.stdout.write(') > {}_{}_{}_{}.txt 2>&1'.format(model,N,alpha,n_samples)) 

