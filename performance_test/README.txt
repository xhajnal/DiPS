Instructions to reproduce performance test:

1. (optional) pull desired commit to be compared - this may dysfunction parts of the performance test (due to compatibility)
2. copy './config.ini' from this folder to main folder to replace the original (you may want to back it up before replacing)
   This step assigns paths for I/O files

3. (optional, advanced) generate new performance inputs
	3.1 copy models and properties to 'DiPS/model/examples'
	3.2 obtain data
	    3.2a copy created data to './data'
        3.2b generate data
            # edit 'generate_data.py' to create a new set of data for bee models
            # this file produces data only for the bee models
            python generate_data.py
            # save data
	3.3 edit 'performance_test.py' to run with new models, properties, and data

4. (optional) adjust performance test settings
    edit 'performance_test.py' - look for
    4.1 ## SET OF RUNS
        - to set which inputs to run
    4.2 ## SET METHODS
        - to sets which method to run (optimisation, sampling, refinement, Metropolis-Hastings, parameter lifting)
    4.3 ## GLOBAL SETTINGS
        - to sets global settings such as verbose, factorisation, and timeout

5. (optional) adjust method settings
    5.1 edit './config.ini' accordingly

6. run performance test
    $ python performance_test.py
