

from common.config import load_config
from metropolis_hastings import initialise_sampling

config = load_config()
# results_dir = spam["results"]
results_dir = config["results"]
refinement_results = config["refinement_results"]
refine_timeout = config["refine_timeout"]
z3_path = config["z3_path"]
tmp_dir = config["tmp"]
del config


def initialise_mhmh(params, parameter_intervals, functions, data, sample_size: int,  mh_sampling_iterations: int, eps=0,
                        sd=0.15, theta_init=False, where=False, progress=False, burn_in=False, bins=20, timeout=False,
                        debug=False, metadata=True, draw_plot=False):
    """ Initialisation method for MHMH - space refinement with prior splitting based on MH

    Args:
        params (list of strings): parameter names
        parameter_intervals (list of tuples): domains of parameters
        theta_init (list of floats): initial parameter point
        functions (list of strings): expressions to be evaluated and compared with data
        data (list of floats): measurement values
        sample_size (int): total number of observations in data
        mh_sampling_iterations (int): number of iterations/steps in searching in space
        eps (number): very small value used as probability of non-feasible values in prior - not used now
        sd (float): variation of walker in parameter space
        where (tuple/list or False): output matplotlib sources to output created figure, if False a new will be created
        progress (Tkinter element or False): function processing progress
        burn_in (number): fraction or count of how many samples will be trimmed from beginning
        bins (int): number of segments per dimension in the output plot
        timeout (int): timeout in seconds (0 for no timeout)
        debug (bool): if True extensive print will be used
        metadata (bool): if True metadata will be plotted
        draw_plot (Callable): function showing intermediate plots
    """
    ## TODO delete following lines
    metadata = False
    where = True

    ## Run MH
    a = initialise_sampling(params, parameter_intervals, functions, data, sample_size,  mh_sampling_iterations, eps=eps,
                            sd=sd, theta_init=theta_init, where=where, progress=progress, burn_in=burn_in, bins=bins,
                            timeout=timeout, debug=debug, metadata=metadata, draw_plot=draw_plot)
    ## Create bins


    ## Parse the bins
    ## TODO

    ## Split the space based on the MH results
    ## TODO

    ## Run refinement
    ## TODO

    return a


if __name__ == '__main__':
    params = ["x", "y"]
    parameter_intervals = [(0, 1), (0, 1)]
    f = ["p**2-2*p+1", "2*q*p**2-2*p**2-2*q*p+2*p", "(-2)*q*p**2+p**2+2*q*p"]
    spam = initialise_mhmh(params, parameter_intervals, data=[], functions=f, sample_size=100,
                           mh_sampling_iterations=100, eps=0, debug=True)
    print()
    print(spam)
