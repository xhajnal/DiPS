import os
from time import time
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import pickle
from matplotlib.figure import Figure

# from termcolor import colored

## Importing my code
from space import RefinedSpace
from common.config import load_config
from common.document_wrapper import DocumentWrapper
from common.convert import niceprint

spam = load_config()
# results_dir = spam["results"]
tmp_dir = spam["tmp"]
del spam

wrapper = DocumentWrapper(width=75)


class HastingsResults:
    """ Class to represent Metropolis Hastings results"""

    def __init__(self, params, theta_init, accepted, observations_count: int, observations_samples_count: int,
                 mh_sampling_iterations: int, eps, not_burn_in=75, pretitle="", title="", bins=20, last_iter=0, timeout=0, time_it_took=0):
        """
        Args:
            params (list of strings): parameter names
            accepted (np.array): accepted points
            observations_count (int): total number of observations
            observations_samples_count (int): sample size from the observations
            mh_sampling_iterations (int): number of iterations/steps in searching in space
            eps (number): very small value used as probability of non-feasible values in prior
            not_burn_in (number): shows last given percent of accepted points (trim burn-in period)
            pretitle (string): title to be put in front of title
            title (string): title of the plot
            bins (int): number of segments in the heatmap plot (used only for 2D param space)
        """
        ## Inside variables
        self.params = params
        self.theta_init = theta_init

        ## Results
        self.accepted = accepted
        # self.rejected = rejected

        ## Results setting
        self.observations_count = observations_count
        self.observations_samples_count = observations_samples_count
        self.mh_sampling_iterations = mh_sampling_iterations
        self.eps = eps

        self.show = not_burn_in
        self.title = title
        self.pretitle = pretitle

        self.bins = bins

        self.last_iter = last_iter
        self.timeout = timeout
        self.time_it_took = time_it_took

    def show_mh_heatmap(self, where=False, bins=False, not_burn_in=None, as_scatter=False, debug=False):
        """ Visualises the result of Metropolis Hastings as a heatmap

        Args:
            where (tuple/list): output matplotlib sources to output created figure
            bins (int): number of segments in the plot (used only for heatmap - 2D param space)
            not_burn_in (int or False or None): show last x percents of the accepted values, None - use class value (to trim burn-in period)
            as_scatter (bool): Sets the plot to scatter plot even for 2D output
            debug (bool): if True extensive print will be used

        @author: xhajnal
        @edit: denis
        """
        # import matplotlib as mpl
        # mpl.rcParams.update(mpl.rcParamsDefault)
        # plt.style.use('default')

        if debug:
            print("show", not_burn_in)
            print("self.accepted", self.accepted)

        if not_burn_in is None:
            not_burn_in = self.show

        ## Convert percents / fraction to show into exact number
        if 0 < not_burn_in < 1:
            self.show = not_burn_in * 100
            not_burn_in = int(-not_burn_in * self.accepted.shape[0])
        elif not_burn_in < 0:
            self.show = not_burn_in
            not_burn_in = 75
        else:
            self.show = not_burn_in
            not_burn_in = int(-not_burn_in / 100 * self.accepted.shape[0])

        if self.title is "":
            if self.last_iter > 0:
                self.title = f'Estimate of MH algorithm, {niceprint(self.last_iter)} iterations, sample size = {self.observations_samples_count}/{self.observations_count}, \n showing last {self.show}% of {niceprint(self.accepted.shape[0])} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'
            else:
                self.title = f'Estimate of MH algorithm, {niceprint(self.mh_sampling_iterations)} iterations, sample size = {self.observations_samples_count}/{self.observations_count}, \n showing last {self.show}% of {niceprint(self.accepted.shape[0])} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'

        if debug:
            print("self.accepted[show:, 0]", self.accepted[not_burn_in:, 0])

        if bins is not False:
            self.bins = bins

        ## Multidimensional case
        if len(self.accepted[0]) > 2 or as_scatter:
            if where:
                fig = where[0]
                ax = where[1]
                plt.autoscale()
                ax.autoscale()
            else:
                fig, ax = plt.subplots()

            ## Creates values of the horizontal axis
            # x_axis = list(range(1, len(self.accepted[0]) + 1))

            ## Time check
            # from time import time
            # import socket
            # start_time = time()

            ## Get values of the vertical axis for respective line
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ## Thanks to Den for optimisation
            egg = self.accepted[not_burn_in:].T
            ax.plot(egg, '.-', markersize=15)

            # for sample in self.accepted[show:]:
            #    ax.scatter(x_axis, sample)
            #    ax.plot(x_axis, sample)

            ax.set_xlabel("param indices")
            ax.set_ylabel("parameter value")
            ax.set_title("\n".join(wrapper.wrap(self.title)))
            ax.autoscale()
            ax.margins(0.1)
            # print(colored(f"  It took {socket.gethostname()}, {time() - start_time} seconds to run", "blue"))

            if where:
                return fig, ax
            else:
                plt.show()
        else:
            if where:
                plt.hist2d(self.accepted[not_burn_in:, 0], self.accepted[not_burn_in:, 1], bins=self.bins)
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title("\n".join(wrapper.wrap(self.title)))
                where[1] = plt.colorbar()
                return where[0], where[1]
            else:
                plt.figure(figsize=(12, 6))
                plt.hist2d(self.accepted[not_burn_in:, 0], self.accepted[not_burn_in:, 1], bins=self.bins)
                plt.colorbar()
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title(self.title)
                plt.show()

    def get_acc_as_a_list(self):
        return self.accepted.tolist()


def sample_functions(functions, data_means):
    """ Will sample functions according to the pdf as given by the polynomials

    Args:
        functions TODO @Tanja
        data_means TODO @Tanja

    Returns:
         ## TODO @Tanja

    @author: tpetrov
    """
    x = np.random.uniform(0, 1)
    i = 0
    while x > sum(data_means[0:i]) and (i < len(functions)):
        i = i + 1
    return i - 1


def transition_model_a(theta, parameter_intervals):
    """" Defines how to walk around the parameter space
        using normal distribution around old point

    Args:
        theta (list): old parameter point
        parameter_intervals (list of tuples) domains of parameters

    Returns:
        theta_new (list): new parameter point within the domains

    @author: tpetrov
    @edit: xhajnal, denis
    """
    sd = 0.3  ## Standard deviation of the normal distribution
    theta_new = np.zeros(len(theta))  ## New point initialisation

    ## For each parameter
    ## TODO why we change all params and not just one in random?
    for index, param in enumerate(theta):
        temp = parameter_intervals[index][0] - 1  ## Lower bound of first parameter - 1
        while (temp <= parameter_intervals[index][0]) or (temp >= parameter_intervals[index][1]):
            ## Generate new parameter value from normal distribution
            temp = np.random.normal(theta[index], sd)
        ##  Store only if the param value inside the domains
        theta_new[index] = temp

    return theta_new


def prior(theta, parameter_intervals):
    """ Very simple prior estimator only for MH as it checks if the parametrisation is inside of respective domain or not
        This simulates uniform distribution
    Args:
        theta: (tuple): parameter point
        parameter_intervals (list of pairs): parameter domains, (min, max)

    Returns:
        1 for all parametrisations inside the parameter space
        0 otherwise

    @author: xhajnal
    """

    for index, value in enumerate(theta):
        ## If inside of param domains
        if (theta[index] < parameter_intervals[index][0]) or (theta[index] > parameter_intervals[index][1]):
            return 0
    return 1


def acceptance(x_likelihood, x_new_likelihood):
    """ Decides whether to accept new point or not

    Args:
        x_likelihood: likelihood of the old parameter point
        x_new_likelihood: likelihood of the new parameter point

    Returns:
         True if the new points is accepted

    @author: tpetrov
    """
    ## If likelihood of new point is higher (than likelihood of current point) accept the new point
    if x_new_likelihood > x_likelihood:
        return True
    else:
        ## Chance to accept even if the likelihood of the new point is lower (than likelihood of current point)
        accept = np.random.uniform(0, 1)
        return accept < (np.exp(x_new_likelihood - x_likelihood))


def metropolis_hastings(likelihood_computer, prior_rule, transition_model, param_init, iterations, space, observations,
                        acceptance_rule, parameter_intervals, functions, eps, progress=False, timeout=-1, debug=False):
    """ The core method of the Metropolis Hasting

        likelihood_computer (function(space, theta, functions, observation/data, eps)): function returning the likelihood that functions in theta point generated the data
        prior_rule (function(theta, eps)): prior function
        transition_model (function(theta)): a function that draws a sample from a symmetric distribution and returns it
        param_init  (pair of numbers): starting parameter point
        iterations (int): number of accepted to generated
        observations (list of numbers): observations that we wish to model
        acceptance_rule (function(theta, theta_new)): decides whether to accept or reject the new sample
        parameter_intervals (list of pairs): parameter domains
        progress (Tkinter_element or False): progress bar
        timeout (int): timeout in seconds
        debug (bool): if True extensive print will be used

    Returns:
        accepted, rejected (tuple of np.arrays): tuple of accepted and rejected parameter points

    @author: tpetrov
    @edit: xhajnal
    """
    theta = param_init
    both = []
    accepted = []
    rejected = []
    ## For each MCMC iteration do
    for iteration in range(iterations):
        ## Walk in parameter space - Get new parameter point from the current one
        theta_new = transition_model(theta, parameter_intervals)
        ## Estimate likelihood of current point
        ## TODO - check whether we can reuse the likelihood from the previous iteration
        ## (space, theta, functions, data, eps)
        theta_lik = likelihood_computer(space, theta, functions, observations, eps)
        # print("theta_lik", theta_lik)
        ## Estimate likelihood of new point
        theta_new_lik = likelihood_computer(space, theta_new, functions, observations, eps)
        # print("theta_new_lik", theta_new_lik)
        if debug:
            print("iteration:", iteration)
        # print("theta_lik + np.log(prior(theta, parameter_intervals))", theta_lik + np.log(prior(theta, parameter_intervals)))
        # print("theta_new_lik + np.log(prior(theta, parameter_intervals))", theta_new_lik + np.log(prior(theta_new, parameter_intervals)))

        ## If new point accepted
        if acceptance_rule(theta_lik + np.log(prior_rule(theta, parameter_intervals)), theta_new_lik + np.log(prior_rule(theta_new, parameter_intervals))):
            ## Go to the new point
            theta = theta_new
            accepted.append(theta_new)
            both.append([theta_new, True])
            if debug:
                print(f"new point: {theta_new} accepted")
        else:
            rejected.append(theta_new)
            both.append([theta_new, False])
            if debug:
                print(f"new point: {theta_new} rejected")
        if progress:
            progress(iteration/iterations, False, int(time() - globals()["start_time"]), timeout)

        ## Finish iterations after timeout
        if (time() - globals()["start_time"]) > timeout >= 0:
            globals()["results"].last_iter = iteration
            globals()["results"].time_it_took = time() - globals()["start_time"]
            break

    globals()["mh_results"].time_it_took = time() - globals()["start_time"]
    return np.array(both), np.array(accepted), np.array(rejected)


def manual_log_like_normal(space, theta, functions, observations, eps):
    """ Log likelihood of functions in point theta drawing the data

    Args:
        space (Refined space): supporting structure, defining parameters, their domains and types
        theta (list): parameter point
        functions (list of strings): functions to be evaluated in theta
        observations (list of ints): list of function indices which are being observed
        eps (number): very small value used as probability of non-feasible values in prior

    Returns:
         likelihood (float): P(data | functions(theta))

    @author: tpetrov
    @edit: xhajnal
    """
    res = 0
    # print("data", data)
    # print("functions", functions)

    ## Assignment of parameter values
    for index, param in enumerate(theta):
        locals()[space.get_params()[index]] = theta[index]

    ## Dictionary optimising performance - not evaluating the same functions again
    evaled_functions = {}

    for data_point in observations:  # observations:
        # print("data_point", data_point)
        # print("functions[data_point]", functions[data_point])

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
                        observations_samples_size: int, mh_sampling_iterations: int, eps, theta_init=False, where=False,
                        progress=False, not_burn_in=False, bins=20, timeout=False,
                        debug=False, metadata=True, draw_plot=False):
    """ Initialisation method for Metropolis Hastings

    Args:
        space (RefinedSpace): supporting structure, defining parameters, their domains and types
        observations (list of ints): either experiment or data(experiment result frequency)
        functions (list of strings):
        observations_count (int): total number of observations
        observations_samples_size (int): sample size from the observations
        mh_sampling_iterations (int): number of iterations/steps in searching in space
        eps (number): very small value used as probability of non-feasible values in prior
        theta_init (list of numbers): initial parameter point
        where (tuple/list): output matplotlib sources to output created figure
        progress (Tkinter element or False): progress bar
        not_burn_in (number): show last x percents of the accepted values (trim burn-in period)
        bins (int): number of segments in the plot
        timeout (int): timeout in seconds
        debug (bool): if True extensive print will be used
        metadata (bool): if True metadata will be plotted
        draw_plot (function): function showing intermidiate plots

    @author: tpetrov
    @edit: xhajnal
    """

    ## Internal settings
    start_time = time()
    globals()["start_time"] = start_time

    observations_samples_size = min(observations_count, observations_samples_size)
    ##                     HastingsResults ( params, theta_init, accepted, observations_count, observations_samples_count, MH_sampling_iterations, eps, not_burn_in,      pretitle, title, bins, last_iter,  timeout, time_it_took, rescale):
    globals()["mh_results"] = HastingsResults(space.params, theta_init, False, observations_count, observations_samples_size, mh_sampling_iterations, eps, not_burn_in=not_burn_in, title="", bins=bins, last_iter=0, timeout=timeout)

    ## TODO check this
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
    # If given observations or data
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
            samples.append(sample_functions(functions, data_means))
        print("samples", samples)

        observations = np.array(samples)[np.random.randint(0, observations_count, observations_samples_size)]
    print("observations", observations)

    if metadata:
        ## Plotting the distribution of observations
        Y = []
        for i in range(len(functions)):
            Y.append(observations.count(i))

        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(list(range(len(functions))), Y)
        # plt.xticks(range(len(functions)), range(len(functions) + 1))
        ax.set_xlabel("Function index")
        ax.set_ylabel("Number of observations")
        ax.set_title(f"Distribution of {observations_samples_size} observations (from full sample= {observations_count})")
        if not where:
            plt.show()
        else:
            draw_plot(fig)

    print("Initial parameter point: ", theta_init)

    ##                                      (likelihood_computer,    prior, transition_model,   param_init, iterations,             space,    data, acceptance_rule, parameter_intervals, functions, eps, progress,          timeout,         debug):
    both, accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model_a, theta_init, mh_sampling_iterations, space, observations, acceptance, parameter_intervals, functions, eps, progress=progress, timeout=timeout, debug=debug)

    globals()["mh_results"].accepted = accepted

    print("accepted.shape", accepted.shape)
    if len(accepted) == 0:
        return False, False

    print(f"Set of accepted and rejected points is stored here: {tmp_dir}/[accepted/rejected]_{observations_samples_size}.p")
    pickle.dump(accepted, open(os.path.join(tmp_dir, f"accepted_{observations_samples_size}.p"), 'wb'))
    pickle.dump(rejected, open(os.path.join(tmp_dir, f"rejected_{observations_samples_size}.p"), 'wb'))

    print(f"Whole class is stored here: {tmp_dir}/mh_class.p")
    pickle.dump(globals()["mh_results"], open(os.path.join(tmp_dir, f"mh_class.p"), 'wb'))

    # print("accepted", accepted)
    # to_show = accepted.shape[0]
    # print("accepted[100:to_show, 1]", accepted[100:to_show, 1])
    # print("rejected", rejected)

    if metadata:
        ## Create Scatter plot showing accepted and rejected points in its given order
        fig = Figure(figsize=(10, 10))
        for index, param in enumerate(space.get_params()):
            ax = fig.add_subplot(len(space.get_params()), 1, index+1)
            ## New code adding information how the accepted and rejected points are connected
            ## TODO probably can be optimised
            X_accept, X_reject, Y_accept, Y_reject = [], [], [], []
            for point_index, point in enumerate(both):
                if point[1] is False:
                    X_reject.append(point_index)
                    Y_reject.append(point[0][index])
                else:
                    X_accept.append(point_index)
                    Y_accept.append(point[0][index])
            ax.scatter(X_reject, Y_reject, marker='x', c="r", label='Rejected', alpha=0.5)
            ax.scatter(X_accept, Y_accept, marker='.', c="b", label='Accepted', alpha=0.5)
            ax.set_xlabel("MH Iteration")
            ## Previous code before that information
            # ax.plot(rejected[:, index], 'rx', label='Rejected', alpha=0.5)
            # ax.plot(accepted[:, index], 'b.', label='Accepted', alpha=0.5)
            # ax.set_xlabel("Index")
            ax.set_ylabel(f"${param}$")
            ax.set_title(f"Accepted and Rejected values of ${param}$.")
            ax.grid()
            ax.legend()
        if not where:
            plt.show()
        else:
            draw_plot(fig)

    globals()["mh_results"].not_burn_in = not_burn_in

    if metadata:
        ## Trace and histogram of accepted points
        if not_burn_in is False:
            not_burn_in = int(-0.75 * accepted.shape[0])
        elif 0 < not_burn_in < 1:
            not_burn_in = int(-not_burn_in * accepted.shape[0])
        else:
            not_burn_in = int(-not_burn_in / 100 * accepted.shape[0])

        fig = Figure(figsize=(20, 10))
        gs = gridspec.GridSpec(len(space.get_params()), 2, figure=fig)
        for index, param in enumerate(space.get_params()):
            ax = fig.add_subplot(gs[index, 0])
            ax.plot(accepted[:, index])
            ax.set_title(f"Trace of accepted points for ${param}$")
            ax.set_xlabel(f"Index")
            ax.set_ylabel(f"${param}$")
            ax = fig.add_subplot(gs[index, 1])

            # X = sorted(list(set(accepted[:, index])))
            # Y = []
            # for i in X:
            #     Y.append(list(accepted[:, index]).count(i))
            # ax.bar(X, Y)
            # plt.xticks(range(len(functions)), range(len(functions) + 1))
            bins = 20
            ax.hist(accepted[:, index], bins=bins, density=True)
            ax.set_ylabel("Occurrence")
            ax.set_xlabel(f"${param}$")
            ax.set_title(f"Histogram of  accepted points for ${param}$, {bins} bins.")
            fig.tight_layout()
        if not where:
            plt.show()
        else:
            draw_plot(fig)

    ## TODO make a option to set to see the whole space, not zoomed - freaking hard
    ## "Currently hist2d calculates it's own axis limits, and any limits previously set are ignored." (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.hist2d.html)
    ## Option 2 - https://stackoverflow.com/questions/29175093/creating-a-log-linear-plot-in-matplotlib-using-hist2d
    ## No scale

    if where:
        return globals()["mh_results"]
    else:
        globals()["mh_results"].show_mh_heatmap(where=where)
