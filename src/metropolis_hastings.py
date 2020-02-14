import os
from time import time
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from common.document_wrapper import DocumentWrapper

## Importing my code
from space import RefinedSpace
from common.config import load_config

spam = load_config()
# results_dir = spam["results"]
tmp_dir = spam["tmp"]
del spam

wrapper = DocumentWrapper(width=75)


class HastingsResults:
    """ Class to represent Metropolis Hastings results"""

    def __init__(self, params, theta_init, accepted, observations_count: int, observations_samples_count: int,
                 MH_sampling_iterations: int, eps, show=0, pretitle="", title="", bins=20, last_iter=0, timeout=0, time_it_took=0):
        """
        Args:
            params (list of strings): parameter names
            accepted (np.array): accepted points
            observations_count (int): total number of observations
            observations_samples_count (int): sample size from the observations
            MH_sampling_iterations (int): number of iterations/steps in searching in space
            eps (number): very small value used as probability of non-feasible values in prior
            show (number): show from
            pretitle (string): title to be put in front of title
            title (string): title of the plot
            bins (int): number of segments in the plot
        """
        ## Inside variables
        self.params = params
        self.theta_init = theta_init

        ## Results
        self.accepted = accepted

        ## Results setting
        self.observations_count = observations_count
        self.observations_samples_count = observations_samples_count
        self.MH_sampling_iterations = MH_sampling_iterations
        self.eps = eps

        ## Visualisation setting
        # ## Conversion into percents
        # if show is False:
        #     self.show = int(-0.75 * accepted.shape[0])
        # elif 0 < show < 1:
        #     self.show = int(-show * accepted.shape[0])
        # else:
        #     self.show = int(-show/100 * accepted.shape[0])
        self.show = show
        self.title = title
        self.pretitle = pretitle

        self.bins = bins

        self.last_iter = last_iter
        self.timeout = timeout
        self.time_it_took = time_it_took

    def show_mh_heatmap(self, where=False, bins=False, show=False):
        """ Visualises the result of Metropolis Hastings as a heatmap

        Args:
            where (tuple/list): output matplotlib sources to output created figure
            bins (int): number of segments in the plot
            show (number): show last x percents of the accepted values
        """
        if self.title is "":
            if self.last_iter > 0:
                self.title = f'Estimate of MH algorithm, {self.last_iter} iterations, sample size = {self.observations_samples_count}/{self.observations_count}, \n showing last {-self.show} of {self.accepted.shape[0]} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'
            else:
                self.title = f'Estimate of MH algorithm, {self.MH_sampling_iterations} iterations, sample size = {self.observations_samples_count}/{self.observations_count}, \n showing last {-self.show} of {self.accepted.shape[0]} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'

        if bins is not False:
            self.bins = bins

        if show is not False and show > 0:
            self.show = show

        print("self.show", self.show)
        print("self.accepted", self.accepted)
        print("self.accepted[self.show:, 0]", self.accepted[self.show:, 0])

        ## Multidimensional case
        if len(self.accepted[0]) > 2:
            if where:
                fig = where[0]
                ax = where[1]
                plt.autoscale()
                ax.autoscale()
            else:
                fig, ax = plt.subplots()
            ## Creates values of the horizontal axis
            x_axis = list(range(1, len(self.accepted[0]) + 1))
            ## Get values of the vertical axis for respective line
            for sample in self.accepted[self.show:]:
                ax.scatter(x_axis, sample)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.plot(x_axis, sample)
            ax.set_xlabel("param indices")
            ax.set_ylabel("parameter value")
            ax.set_title("\n".join(wrapper.wrap(self.title)))
            ax.autoscale()
            ax.margins(0.1)

            if where:
                return fig, ax
            else:
                plt.show()
        else:
            if where:
                plt.hist2d(self.accepted[self.show:, 0], self.accepted[self.show:, 1], bins=self.bins)
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title("\n".join(wrapper.wrap(self.title)))
                where[1] = plt.colorbar()
                return where[0], where[1]
            else:
                plt.figure(figsize=(12, 6))
                plt.hist2d(self.accepted[self.show:, 0], self.accepted[self.show:, 1], bins=self.bins)
                plt.colorbar()
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title(self.title)
                plt.show()


def sample(functions, data_means):
    """ Will sample according to the pdf as given by the polynomials

    Returns:
         ## TODO @Tanja

    @author: tpetrov
    """
    x = np.random.uniform(0, 1)
    i = 0
    # print(x)
    while x > sum(data_means[0:i]) and (i < len(functions)):
        i = i + 1
    return i - 1


def transition_model_a(theta, parameter_intervals):
    """" Defines how to walk around the parameter space

    Args:
        theta (list): parameter values
        parameter_intervals (list of tuples) parameter intervals

    Returns:
         new parameter space point

    @author: tpetrov
    @edit: xhajnal, denis
    """
    sd = 0.3  ## Standard deviation of the normal distribution
    theta_new = np.zeros(len(theta))

    for index, param in enumerate(theta):
        temp = parameter_intervals[index][0] - 1  ## Lower bound of first parameter - 1
        while (temp <= parameter_intervals[index][0]) or (temp >= parameter_intervals[index][1]):
            temp = np.random.normal(theta[index], sd)
        theta_new[index] = temp

    return theta_new


def prior(x, eps):
    """ TODO @Tanja
    Args:
        x: (tuple) Distribution parameters: x[0] = mu, x[1] = sigma (new or current)
        eps (number): very small value used as probability of non-feasible values in prior


    Returns:
        1 for all valid values of sigma. Log(1) = 0, so it does not affect the summation.
        0 for all invalid values of sigma (<= 0). Log(0) = -infinity, and Log(negative number) is undefined.
        It makes the new sigma infinitely unlikely.

    @author: tpetrov
    """

    for i in range(2):
        if (x[i] < 0) or (x[i] > 1):
            return eps
    return 1 - eps


def acceptance(x, x_new):
    """ Decides whether to accept new sample or not

    Args:
        x: old parameter point
        x_new: new parameter points

    Returns:
         True if the new points is accepted

    @author: tpetrov
    """
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        # print("x", x)
        # print("x_new", x_new)
        # print("x_new - x", x_new - x)
        # print("np.exp(x_new - x)", np.exp(x_new - x))
        return accept < (np.exp(x_new - x))


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, space, observations,
                        acceptance_rule, parameter_intervals, functions, eps, progress=False, timeout=-1, debug=False):
    """ TODO @Tanja
    the main method

        likelihood_computer (function(x, data)): returns the likelihood that these parameters generated the data
        prior (function(x, eps)): prior function
        transition_model (function(x)): a function that draws a sample from a symmetric distribution and returns it
        param_init  (pair of numbers): a starting sample
        iterations (int): number of accepted to generated
        observations (list of numbers): observations that we wish to model
        acceptance_rule (function(x, x_new)): decides whether to accept or reject the new sample
        parameter_intervals (list of pairs): boundaries of parameters
        progress (False or Tkinter_element): progress bar
        timeout (int): timeout in seconds
        debug (bool): if True extensive print will be used

    Returns:
         tuple of accepted and rejected parameter points

    @author: tpetrov
    @edit: xhajnal
    """
    # print("acceptance_rule", acceptance_rule)

    x = param_init
    accepted = []
    rejected = []
    for iteration in range(iterations):
        x_new = transition_model(x, parameter_intervals)
        ## (space, theta, functions, data, eps)
        x_lik = likelihood_computer(space, x, functions, observations, eps)
        # print("x_lik", x_lik)
        x_new_lik = likelihood_computer(space, x_new, functions, observations, eps)
        # print("x_new_lik", x_new_lik)
        if debug:
            print("iteration:", iteration)
        # print("x_lik + np.log(prior(x, eps))", x_lik + np.log(prior(x, eps)))
        # print("x_new_lik + np.log(prior(x_new, eps))", x_new_lik + np.log(prior(x_new, eps)))

        if acceptance_rule(x_lik + np.log(prior(x, eps)), x_new_lik + np.log(prior(x_new, eps))):
            x = x_new
            accepted.append(x_new)
            if debug:
                print(f"new point: {x_new} accepted")
        else:
            rejected.append(x_new)
            if debug:
                print(f"new point: {x_new} rejected")
        if progress:
            progress(iteration/iterations)

        ## Finish iterations after timeout
        if (time() - globals()["start_time"]) > timeout >= 0:
            ## TODO store current iteration index
            globals()["results"].last_iter = iteration
            globals()["results"].time_it_took = time() - globals()["start_time"]
            break

    globals()["results"].time_it_took = time() - globals()["start_time"]
    return np.array(accepted), np.array(rejected)


def manual_log_like_normal(space, theta, functions, observations, eps):
    """ TODO @Tanja

    Args:
        space (Refined space): supporting structure
        theta (list): parameter values
        functions (list of strings): functions to be evaluated in theta
        observations (list of ints): list of function indices which are being observed
        eps (number): very small value used as probability of non-feasible values in prior

    Returns:
         ## TODO @Tanja

    @author: tpetrov
    @edit: xhajnal
    """
    res = 0
    # print("data", data)
    # print("functions", functions)

    for index, param in enumerate(theta):
        locals()[space.get_params()[index]] = theta[index]
    # locals()[space.get_params()[0]] = theta[0]
    # locals()[space.get_params()[1]] = theta[1]

    evaled_functions = {}

    for data_point in observations:  # observations:
        # print("data_point", data_point)
        # print("functions[data_point]", functions[data_point])
        # print("x=", globals()["x"])
        # print("y=", globals()["y"])
        # print(eval("x+y"))

        if data_point in evaled_functions.keys():
            temp = evaled_functions[data_point]
        else:
            evaled_functions[data_point] = eval(functions[data_point])
            temp = evaled_functions[data_point]

        # print(temp)
        # print(np.log(temp))

        if temp < eps:
            temp = eps
        if temp > 1. - eps:
            temp = 1. - eps

        # print(res)
        res = res + np.log(temp)  # +np.log(prior(x))
    # print(res)
    return res


def initialise_sampling(space: RefinedSpace, observations, functions, observations_count: int,
                        observations_samples_size: int, MH_sampling_iterations: int, eps,
                        theta_init=False, where=False, progress=False, show=False, bins=20, timeout=False, debug=False):
    """ Initialisation method for Metropolis Hastings
    Args:
        space (RefinedSpace):
        observations (list of ints): either experiment or data(experiment result frequency)
        functions (list of strings):
        observations_count (int): total number of observations
        observations_samples_size (int): sample size from the observations
        MH_sampling_iterations (int): number of iterations/steps in searching in space
        eps (number): very small value used as probability of non-feasible values in prior
        theta_init (list of numbers): initial point in parameter space
        where (tuple/list): output matplotlib sources to output created figure
        progress (Tkinter element): progress bar
        show (number): show last x percents of the accepted values
        bins (int): number of segments in the plot
        timeout (int): timeout in seconds
        debug (bool): if True extensive print will be used


    @author: tpetrov
    @edit: xhajnal
    """
    ## Internal settings
    start_time = time()
    globals()["start_time"] = start_time

    observations_samples_size = min(observations_count, observations_samples_size)
    ##                     HastingsResults ( params, theta_init, accepted, observations_count, observations_samples_count, MH_sampling_iterations, eps, show,      pretitle, title, bins, last_iter,  timeout, time_it_took, rescale):
    globals()["results"] = HastingsResults(space.params, theta_init, False, observations_count, observations_samples_size, MH_sampling_iterations, eps, show=show, title="", bins=bins, last_iter=0, timeout=timeout)

    # ## Convert z3 functions
    # for index, function in enumerate(functions):
    #     if is_this_z3_function(function):
    #         functions[index] = translate_z3_function(function)

    # for index, param in enumerate(space.get_params()):
    #     globals()[param] = space.true_point[index]
    #     print(f"{param} = {space.true_point[index]}")
    # globals()[space.get_params()[0]] = space.true_point[0]
    # globals()[space.get_params()[1]] = space.true_point[1]
    # print(f"{space.get_params()[0]} = {globals()[space.get_params()[0]]}")
    # print(f"{space.get_params()[1]} = {globals()[space.get_params()[1]]}")

    parameter_intervals = space.get_region()

    # theta_true = np.zeros(len(space.get_params()))
    # for index, param in enumerate(space.true_point):
    #     theta_true[index] = param
    # theta_true = np.zeros(2)
    # theta_true[0] = space.true_point[0]
    # theta_true[1] = space.true_point[1]

    # print("Parameter point", theta_true)

    ## If no starting point given
    if not theta_init:
        theta_init = []
        ## Select point which is center of each interval
        for index, param in enumerate(parameter_intervals):
            theta_init.append((parameter_intervals[index][0] + parameter_intervals[index][1])/2)
        # theta_init = [(parameter_intervals[0][0] + parameter_intervals[0][1])/2, (parameter_intervals[1][0] + parameter_intervals[1][1])/2]  ## Middle of the intervals # np.ones(10)*0.1

    for index, param in enumerate(space.get_params()):
        globals()[param] = theta_init[index]
        print(f"{param} = {theta_init[index]}")

    ## Maintaining observations
    if observations:
        ## Checking the type of observations (experiment/data)
        if len(observations) > len(functions):
            ## Already given observations
            observations_count = len(observations)
        else:
            ## Changing the data into observations
            spam = []
            index = 0
            for observation in observations:
                for times in range(int(observation * float(observations_count))):
                    spam.append(index)
                index = index + 1
            observations = spam
    else:
        data_means = [eval(fi) for fi in functions]
        print("data means", data_means)
        samples = []
        for i in range(observations_count):
            samples.append(sample(functions, data_means))
        print("samples", samples)

        observations = np.array(samples)[np.random.randint(0, observations_count, observations_samples_size)]
    print("observations", observations)

    if not where:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(observations, bins=range(len(functions)))
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Figure 1: Distribution of {observations_samples_size} observations (from full sample= {observations_count})")
        plt.show()

    # theta_new = transition_model_a(theta_true, parameter_intervals)     ## apparently just a print call
    # r = prior(theta_true, eps)
    # np.log(r)  ## another print call
    # res = manual_log_like_normal(space, theta_true, functions, np.array(observations), eps)

    print("Initial parameter point: ", theta_init)

    ##                                      (likelihood_computer,    prior, transition_model,   param_init, iterations,             space,    data, acceptance_rule, parameter_intervals, functions, eps, progress,          timeout,         debug):
    accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model_a, theta_init, MH_sampling_iterations, space, observations, acceptance, parameter_intervals, functions, eps, progress=progress, timeout=timeout, debug=debug)

    globals()["results"].accepted = accepted

    print("accepted.shape", accepted.shape)
    if len(accepted) == 0:
        return False, False

    ## TODO dump the class instead
    print(f"Set of accepted and rejected points is stored here: {tmp_dir}")
    pickle.dump(accepted, open(os.path.join(tmp_dir, f"accepted_{observations_samples_size}.p"), 'wb'))
    pickle.dump(rejected, open(os.path.join(tmp_dir, f"rejected_{observations_samples_size}.p"), 'wb'))

    # accepted = pickle.load(open(os.path.join(tmp_dir, f"accepted_{N_obs}.p"), "rb"))
    # rejected = pickle.load(open(os.path.join(tmp_dir, f"rejected_{N_obs}.p"), "rb"))

    # print("accepted", accepted)
    # to_show = accepted.shape[0]
    # print("accepted[100:to_show, 1]", accepted[100:to_show, 1])
    # print("rejected", rejected)

    ## Create TODO add name plot @Tanja
    if not where:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        to_show = accepted.shape[0]
        ax.plot(rejected[(0, 100)[len(rejected) > 200]:to_show, 1], 'rx', label='Rejected', alpha=0.5)
        ax.plot(accepted[(0, 100)[len(accepted) > 200]:to_show, 1], 'b.', label='Accepted', alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(f"Figure 2: MCMC sampling for N={observations_samples_size} with Metropolis-Hastings. Last {to_show}% of samples are shown.")
        ax.grid()
        ax.legend()
        plt.show()

    if show is False:
        show = int(-0.75 * accepted.shape[0])
    elif 0 < show < 1:
        show = int(-show * accepted.shape[0])
    else:
        show = int(-show / 100 * accepted.shape[0])

    globals()["results"].show = show

    ## Create TODO add name plot @Tanja
    if not where:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(accepted[show:, 0])
        ax.set_title(f"Figure 4: Trace for {space.get_params()[0]}")
        ax.set_ylabel(f"${space.get_params()[0]}$")
        ax = fig.add_subplot(1, 2, 2)
        ax.hist(accepted[show:, 0], bins=20, density=True)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(f"${space.get_params()[0]}$")
        ax.set_title(f"Fig.5: Histogram of ${space.get_params()[0]}$")
        fig.tight_layout()

    ## TODO make a option to set to see the whole space, not zoomed - freaking hard
    ## "Currently hist2d calculates it's own axis limits, and any limits previously set are ignored." (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.hist2d.html)
    ## Option 2 - https://stackoverflow.com/questions/29175093/creating-a-log-linear-plot-in-matplotlib-using-hist2d
    ## No scale
    if where:
        return globals()["results"]
    else:
        globals()["results"].show_mh_heatmap(where=where)
