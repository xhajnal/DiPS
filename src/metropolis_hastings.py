import os
from platform import system
from time import time
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure

# from termcolor import colored

## Importing my code
from termcolor import colored

from common.files import pickle_dump
from space import RefinedSpace
from common.config import load_config
from common.document_wrapper import DocumentWrapper
from common.convert import niceprint

import warnings

spam = load_config()
# results_dir = spam["results"]
tmp_dir = spam["tmp"]
del spam

wrapper = DocumentWrapper(width=75)


def maximize_plot():
    try:
        mng = plt.get_current_fig_manager()
        if "wind" in system().lower():
            mng.window.state('zoomed')
        else:
            mng.resize(*mng.window.maxsize())
    except:
        pass


class HastingsResults:
    """ Class to represent Metropolis Hastings results"""
    def __init__(self, params, theta_init, accepted, rejected, observations_count: int, observations_samples_count: int,
                 mh_sampling_iterations: int, eps, burn_in=0.25, pretitle="", title="", bins=20, last_iter=0, timeout=0, time_it_took=0):
        """
        Args:
            params (list of strings): parameter names
            accepted (np.array): accepted points with iteration index
            rejected (np.array): rejected points with iteration index
            observations_count (int): total number of observations
            observations_samples_count (int): sample size from the observations
            mh_sampling_iterations (int): number of iterations/steps in searching in space
            eps (number): very small value used as probability of non-feasible values in prior
            burn_in (number): fraction or count of how many samples will be trimmed from beginning
            pretitle (string): title to be put in front of title
            title (string): title of the plot
            bins (int): number of segments in the heatmap plot (used only for 2D param space)
        """
        ## Inside variables
        self.params = params
        self.theta_init = theta_init

        ## Results
        self.accepted = accepted
        self.rejected = rejected

        ## Results setting
        self.observations_count = observations_count
        self.observations_samples_count = observations_samples_count
        self.mh_sampling_iterations = mh_sampling_iterations
        self.eps = eps

        self.burn_in = burn_in
        # try:  ## backward compatibility
        #     self.not_burn_in = not_burn_in
        # except AttributeError as exx:
        #     if "'HastingsResults' object has no attribute 'not_burn_in'" in exx:
        #         self.not_burn_in = show
        self.title = title
        self.pretitle = pretitle

        self.bins = bins

        self.last_iter = last_iter
        self.timeout = timeout
        self.time_it_took = time_it_took

    def get_burn_in(self):
        """ Returns fraction of the burned-in part"""
        if 0 < self.burn_in < 1:
            return self.burn_in
        elif len(self.accepted):
            return min(1, self.burn_in / len(self.accepted))
        else:
            return None

    def merge_acc_and_rej(self):
        """ Returns both, accepted and rejected samples, in a single list"""
        spam = np.empty([len(self.accepted) + len(self.rejected), len(self.params) + 1])
        acc_index = 0
        rej_index = 0
        for i in range(len(spam)):
            if acc_index < len(self.accepted):
                if int(self.accepted[acc_index][-1]) == i:
                    spam[i] = np.append(self.accepted[acc_index][:-1], True)
                    acc_index = acc_index + 1
                else:
                    spam[i] = np.append(self.rejected[rej_index][:-1], False)
                    rej_index = rej_index + 1
            else:
                spam[i] = np.append(self.rejected[rej_index][:-1], False)
                rej_index = rej_index + 1
        return spam

        # spam = np.empty([len(self.accepted) + len(self.rejected), len(self.params) + 1])
        # acc_index = 0
        # rej_index = 0
        # for i in range(len(self.accepted)):
        #     if int(self.accepted[acc_index][-1]) == i:
        #         spam[i] = np.append(self.accepted[acc_index][:-1], True)
        #         acc_index = acc_index + 1
        #     else:
        #         spam[i] = np.append(self.rejected[rej_index][:-1], False)
        #         rej_index = rej_index + 1
        # for i in range(acc_index + rej_index, len(spam)):
        #     spam[i] = np.append(self.rejected[rej_index][:-1], False)
        #     rej_index = rej_index + 1
        # return spam

        # spam = []
        # acc_index = 0
        # rej_index = 0
        # for i in range(len(self.accepted)):
        #     if int(self.accepted[acc_index][-1]) == i:
        #         spam.append(np.append(self.accepted[acc_index][:-1], True))
        #         acc_index = acc_index + 1
        #     else:
        #         spam.append(np.append(self.rejected[rej_index][:-1], False))
        #         rej_index = rej_index + 1
        # for i in range(rej_index, len(self.rejected)):
        #     spam.append(np.append(self.rejected[rej_index][:-1], False))
        # return spam

    def get_not_burn_in(self):
        """ Returns fraction of not burned-in part"""
        if self.get_burn_in() is not None:
            return 1 - self.get_burn_in()
        else:
            return None

    def set_accepted(self, accepted):
        """ Sets the accepted points"""
        self.accepted = accepted

    def set_rejected(self, rejected):
        """ Sets rejected points"""
        self.rejected = rejected

    def get_acc_as_a_list(self):
        """ Returns accepted points in a list"""
        return self.accepted.tolist()

    def get_rej_as_a_list(self):
        """ Returns rejected points in a list"""
        return self.rejected.tolist()

    def show_mh_heatmap(self, where=False, bins=False, burn_in=None, as_scatter=False, debug=False):
        """ Visualises the result of Metropolis Hastings as a heatmap

        Args:
            where (tuple/list): output matplotlib sources to output created figure
            bins (int): number of segments in the plot (used only for heatmap - 2D param space)
            burn_in (number or None): discards the fraction/number of accepted points, None - use class value (to trim burn-in period)
            as_scatter (bool): Sets the plot to scatter plot even for 2D output
            debug (bool): if True extensive print will be used

        @author: xhajnal
        @edit: denis
        """
        # import matplotlib as mpl
        # mpl.rcParams.update(mpl.rcParamsDefault)
        # plt.style.use('default')

        ## Backwards compatibility
        if len(self.params) == len(self.accepted[0]):
            print("old data")
            indices = np.linspace(1, len(self.accepted), num=len(self.accepted))
            indices = np.array(list(map(lambda x: [x], indices)))
            self.accepted = np.hstack((self.accepted, indices))
            # for index, item in enumerate(self.accepted):
            #     self.accepted[index] = self.accepted[index]

        if debug:
            print("burn-in", burn_in)
            print("self.accepted", self.accepted)

        if burn_in is None:
            try:  ## Backward compatibility
                burn_in = self.burn_in
            except AttributeError as exx:
                if "'HastingsResults' object has no attribute 'burn_in'" in str(exx):
                    try:
                        burn_in = (100 - self.not_burn_in)/100  ## Backward compatibility
                        self.burn_in = burn_in
                    except:
                        pass
                    try:
                        if 0 <= self.show <= 1:  ## Backward compatibility
                            burn_in = 1 - self.show  ## Backward compatibility
                        elif 0 <= self.show <= 100:  ## Backward compatibility
                            burn_in = 1 - self.show/100  ## Backward compatibility
                        elif self.show < 0:  ## Backward compatibility
                            burn_in = len(self.accepted) + self.show  ## Backward compatibility
                        else:
                            burn_in = len(self.accepted) - self.show  ## Backward compatibility
                        self.burn_in = burn_in
                    except:
                        pass
        try:  ## backward compatibility
            if self.mh_sampling_iterations < 0:
                pass
        except AttributeError as exx:
            self.mh_sampling_iterations = self.MH_sampling_iterations  ## Backward compatibility

        if burn_in < 0:
            raise Exception("MH - wrong burn-in setting.")
        if burn_in > len(self.accepted):
            raise Exception("MH - Burn-in values set higher than accepted point. Nothing to show.")

        ## Convert fraction to show into exact number
        if 0 < burn_in < 1:
            keep_index = int(burn_in * self.accepted.shape[0]) + 1
        else:
            keep_index = int(burn_in) + 1
            burn_in = round(burn_in / self.accepted.shape[0], 2)

        if self.last_iter > 0:
            self.title = f'Estimate of MH algorithm, {niceprint(self.last_iter)} iterations, sample size = {self.observations_count}, \n trimming first {burn_in * 100}% of {niceprint(self.accepted.shape[0])} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'
        else:
            self.title = f'Estimate of MH algorithm, {niceprint(self.mh_sampling_iterations)} iterations, sample size = {self.observations_count}, \n trimming first {burn_in * 100}% of {niceprint(self.accepted.shape[0])} acc points, init point: {self.theta_init}, \n It took {gethostname()} {round(self.time_it_took, 2)} second(s)'

        if debug:
            print("self.accepted[keep_index:, 0]", self.accepted[keep_index:, 0])

        if bins is not False:
            self.bins = bins

        ## Multidimensional case
        if len(self.accepted[0]) > 3 or as_scatter:
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
            egg = self.accepted[keep_index:].T
            egg = egg[:-1]
            ax.plot(egg, '.-', markersize=15)  ## TODO Check 10 bees default -> ZeroDivisionError: integer division or modulo by zero

            # for sample in self.accepted[not_burn_in:]:
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
                maximize_plot()
                plt.show()
        else:
            if where:
                plt.hist2d(self.accepted[keep_index:, 0], self.accepted[keep_index:, 1], bins=self.bins)
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title("\n".join(wrapper.wrap(self.title)))
                where[1] = plt.colorbar()
                where[1].set_label('# of accepted points per bin', rotation=270, labelpad=20)
                return where[0], where[1]
            else:
                plt.figure(figsize=(12, 6))
                plt.hist2d(self.accepted[keep_index:, 0], self.accepted[keep_index:, 1], bins=self.bins)
                figure = plt.colorbar()
                plt.xlabel(self.params[0])
                plt.ylabel(self.params[1])
                plt.title(self.title)
                figure.ax.set_ylabel('# of accepted points per bin', rotation=270, labelpad=20)
                maximize_plot()
                plt.show()

    def show_iterations(self, where=False):
        """ Create Scatter plot showing accepted and rejected points in its given order

        Args:
           where (bool or callable): method to forward the figure
        """
        if where:
            fig = Figure(figsize=(10, 10))
        else:
            fig = plt.figure()
        if len(self.params) == 2:
            plots = 3
        else:
            plots = len(self.params)
        for index, param in enumerate(self.params):
            ax = fig.add_subplot(plots, 1, index + 1)
            borderline_index = self.accepted[:, -1][int(self.get_burn_in() * len(self.accepted))-1]
            ax.axvline(x=borderline_index + 0.5, color='black', linestyle='-', label="burn-in threshold")

            ax.scatter(self.rejected[:, -1], self.rejected[:, index], marker='x', c="r", label='Rejected', alpha=0.5)
            ax.scatter(self.accepted[:, -1], self.accepted[:, index], marker='.', c="b", label='Accepted', alpha=0.5)

            ## TODO probably can be optimised
            # ax.axvline(x=self.get_burn_in() * self.mh_sampling_iterations, color='black', linestyle='-',
            #            label="burn-in threshold")
            ## New code adding information how the accepted and rejected points are connected
            # X_accept, X_reject, Y_accept, Y_reject = [], [], [], []
            # for point_index, point in enumerate(self.both):
            #     if point[1] is False:
            #         X_reject.append(point_index)
            #         Y_reject.append(point[0][index])
            #     else:
            #         X_accept.append(point_index)
            #         Y_accept.append(point[0][index])
            # ax.scatter(X_reject, Y_reject, marker='x', c="r", label='Rejected', alpha=0.5)
            # ax.scatter(X_accept, Y_accept, marker='.', c="b", label='Accepted', alpha=0.5)

            ## TODO calculate how many burned samples from burned accepted
            # borderline_index = X_accept[int(self.get_burn_in() * len(self.accepted))-1]
            # ax.axvline(x=borderline_index + 0.5, color='black', linestyle='-', label="burn-in threshold")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("MH Iteration")
            ## Previous code before that information
            # ax.plot(rejected[:, index], 'rx', label='Rejected', alpha=0.5)
            # ax.plot(accepted[:, index], 'b.', label='Accepted', alpha=0.5)
            # ax.set_xlabel("Index")
            ax.set_ylabel(f"${param}$")
            ax.set_title(f"Accepted and Rejected values of ${param}$.")
            ax.grid()
            ax.legend()
        ## Plane plot, phase space
        if len(self.params) == 2:
            ax = fig.add_subplot(plots, 1, 3)
            both = self.merge_acc_and_rej()
            ax.plot(both[:, 0], both[:, 1], label="Path", c="grey", alpha=0.1)
            ax.plot(both[:, 0][0], both[:, 1][0], 'g+', label="Start")
            del both
            ax.plot(self.accepted[:, 0], self.accepted[:, 1], 'b.', label='Accepted', alpha=0.3)
            ax.plot(self.rejected[:, 0], self.rejected[:, 1], 'rx', label='Rejected', alpha=0.3)
            ax.set_xlabel(self.params[0])
            ax.set_ylabel(self.params[1])
            ax.legend()
            ax.set_title("Trace of Accepted and Rejected points in a plane.")
        if not where:
            maximize_plot()
            plt.show()
        else:
            where(fig)

    def show_accepted(self, where=False):
        """ Trace and histogram of accepted points

        Args:
           where (bool or callable): method to forward the figure
        """
        if where:
            fig = Figure(figsize=(20, 10))
        else:
            fig = plt.figure()
        if len(self.params) == 2:
            gs = gridspec.GridSpec(3, 2, figure=fig)
        else:
            gs = gridspec.GridSpec(len(self.params), 2, figure=fig)
        for index, param in enumerate(self.params):
            ## Trace of accepted points for respective parameter
            ax = fig.add_subplot(gs[index, 0])
            ax.plot(self.accepted[:, index])
            ax.axvline(x=int(self.get_burn_in() * len(self.accepted[:, index])) + 0.5, color='black', linestyle='-',
                       label="burn-in threshold")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"Trace of accepted points for ${param}$")
            ax.set_xlabel(f"Index")
            ax.set_ylabel(f"${param}$")

            ## Histogram of accepted points for respective parameter
            ax = fig.add_subplot(gs[index, 1])
            # X = sorted(list(set(accepted[:, index])))
            # Y = []
            # for i in X:
            #     Y.append(list(accepted[:, index]).count(i))
            # ax.bar(X, Y)
            # plt.xticks(range(len(functions)), range(len(functions) + 1))
            bins = 20
            ax.hist(self.accepted[:, index], bins=bins, density=True)
            ax.set_ylabel("Occurrence")
            ax.set_xlabel(f"${param}$")
            ax.set_title(f"Histogram of accepted points for ${param}$, {bins} bins.")
            fig.tight_layout()
        ## Plane plot, phase space
        if len(self.params) == 2:
            ax = fig.add_subplot(gs[2, :])
            ax.plot(self.accepted[:, 0], self.accepted[:, 1],  label="Path", c="grey", alpha=0.1)
            ax.plot(self.accepted[:, 0], self.accepted[:, 1], 'b.', label='Accepted', alpha=0.3)
            ax.plot(self.accepted[:, 0][0], self.accepted[:, 1][0], 'g+', label="Start")
            ax.set_xlabel(self.params[0])
            ax.set_ylabel(self.params[1])
            ax.legend()
            ax.set_title("Trace of Accepted points in a plane.")

        if not where:
            maximize_plot()
            plt.show()
        else:
            where(fig)


## Now unused
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
    """" Defines how to walk around the parameter space, set a new point,
        using normal distribution around the old point

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
        ## Store only if the param value inside the domains
        theta_new[index] = temp

    return theta_new


## Now unused
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
    """ Decides whether to accept new point, x_new, or not, based on its likelihood

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


## Now unused
def acceptance_default(x_likelihood, x_new_likelihood):
    """ Decides whether to accept new point, x_new, or not, based on its likelihood

    Args:
        x_likelihood: likelihood of the old parameter point
        x_new_likelihood: likelihood of the new parameter point

    Returns:
         True if the new points is accepted

    @author: xhajnal
    """
    ## If likelihood of new point is higher (than likelihood of current point) accept the new point
    if x_new_likelihood > x_likelihood:
        return True


def manual_log_like_normal(space, theta, functions, data, sample_size, eps, debug=False):
    """ Log likelihood of functions in parameter point theta drawing the data, P(functions(theta)| data)

    Args:
        space (Refined space): supporting structure, defining parameters, their domains and types
        theta (list): parameter point
        functions (list of strings): functions to be evaluated in theta
        data (list): list of function indices which are being observed
        sample_size (int): number of samples
        eps (number): very small value used as probability of non-feasible values in prior
        debug (bool): if True extensive print will be used

    Returns:
         likelihood (float): P(data | functions(theta))

    @author: tpetrov
    @edit: xhajnal
    """
    warnings.filterwarnings("error")
    res = 0
    # print("data", data)
    # print("functions", functions)

    ## Assignment of parameter values
    for index, param in enumerate(theta):
        locals()[space.get_params()[index]] = theta[index]

    ## Dictionary optimising performance - not evaluating the same functions again
    evaled_functions = {}

    ## TODO represent observations as data
    res = 0

    ## Via wiki https://en.wikipedia.org/wiki/Maximum_likelihood_estimation Discrete distribution, continuous parameter space
    for index, data_point in enumerate(data):
        point = eval(functions[index])
        # lik = C(n,k) * p**k * (1-p)**(n-k)  ## formula
        # lik = nCr(sample_size, data_point*sample_size) * point**(data_point*sample_size) * (1-point)**(sample_size-data_point*sample_size)  ## Our representation
        ## log_lik = np.log(nCr(data_point * sample_size, sample_size)) + (data_point * sample_size * np.log(point) + (1 - data_point) * sample_size * np.log(1 - point))  ## Original log likelihood, but the C(n,k) does not change that the one loglik is greater and it strikes out in subtraction part
        try:
            pseudo_log_lik = (data_point * sample_size * np.log(point) + (1 - data_point) * sample_size * np.log(1 - point))
        except RuntimeWarning as warn:
            if debug:
                print("index", index)
                print("theta", theta)
                print("functions[index]", point)
                print("data[index]", data_point)
            pseudo_log_lik = float("-inf")
            # show_message(2, "MH", f"function value {point} is invalid for log")
        res = res + pseudo_log_lik
        if debug:
            print(f"param point {theta}")
            print(f"data_point {data_point}")
            print(f"function {eval(functions[index])}")
            print(colored(f"log-likelihood {pseudo_log_lik}", "blue"))
            print()
    warnings.filterwarnings("default")

    # for index, data_point in enumerate(data):
    #     sigma = np.sqrt((data_point - eval(functions[index])) ** 2 / sample_size)
    #     res = res - sample_size * np.log(sigma * np.sqrt(2 * np.pi)) - ((data_point - eval(functions[index])) ** 2 / (2 * sigma ** 2))

    ## OLD CODE - using observation - problem how to encode overlapping observation of events
    # for data_point in observations:
    #     # print("data_point", data_point)
    #     # print("functions[data_point]", functions[data_point])
    #
    #     if data_point in evaled_functions.keys():
    #         temp = evaled_functions[data_point]
    #     else:
    #         evaled_functions[data_point] = eval(functions[data_point])
    #         temp = evaled_functions[data_point]
    #
    #     # print(temp)
    #     # print(np.log(temp))
    #
    #     if temp < eps:
    #         temp = eps
    #     if temp > 1. - eps:
    #         temp = 1. - eps
    #
    #     # print(res)
    #     res = res + np.log(temp)  # +np.log(prior(x))

    # print(res)
    return res


def metropolis_hastings(likelihood_computer, prior_rule, transition_model, param_init, iterations, space, data, sample_size,
                        acceptance_rule, parameter_intervals, functions, eps, progress=False, timeout=-1, debug=False):
    """ The core method of the Metropolis Hasting

        likelihood_computer (function(space, theta, functions, observation/data, eps)): function returning the likelihood that functions in theta point generated the data
        prior_rule (function(theta, eps)): prior function
        transition_model (function(theta)): a function that draws a sample from a symmetric distribution and returns it
        param_init  (pair of numbers): starting parameter point
        iterations (int): number of accepted to generated
        space (Refined space): supporting structure, defining parameters, their domains and types
        data (list of numbers): data that we wish to model
        sample_size (int): sample size / number of observations
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
    accepted = []
    ## Setting the initial point as rejected so it will be shown in plots
    ## Even though it is never compared and hence should not be acc nor rej
    rejected = [np.append(theta, 0)]
    ## For each MCMC iteration do
    has_moved = True
    theta_lik = 0
    for iteration in range(1, iterations + 1):
        ## Walk in parameter space - Get new parameter point from the current one
        theta_new = transition_model(theta, parameter_intervals)
        ## Estimate likelihood of current point
        ## (space, theta, functions, data, eps)

        ## Not recalculating the likelihood if we did not move
        if has_moved:
            theta_lik = likelihood_computer(space, theta, functions, data, sample_size, eps, debug=debug)
        # print("theta_lik", theta_lik)
        ## Estimate likelihood of new point
        theta_new_lik = likelihood_computer(space, theta_new, functions, data, sample_size, eps, debug=debug)
        # print("theta_new_lik", theta_new_lik)
        if debug:
            print("iteration:", iteration)
        # print("theta_lik + np.log(prior(theta, parameter_intervals))", theta_lik + np.log(prior(theta, parameter_intervals)))
        # print("theta_new_lik + np.log(prior(theta, parameter_intervals))", theta_new_lik + np.log(prior(theta_new, parameter_intervals)))

        ## If new point accepted
        ## old acceptance rule checking using uniform distribution ## this is done by the walker
        # if acceptance_rule(theta_lik + np.log(prior_rule(theta, parameter_intervals)), theta_new_lik + np.log(prior_rule(theta_new, parameter_intervals))):
        if acceptance_rule(theta_lik, theta_new_lik):
            ## Go to the new point
            has_moved = True
            theta = theta_new
            accepted.append(np.append(theta_new, iteration))
            if debug:
                print(f"new point: {theta_new} accepted")
        else:
            has_moved = False
            rejected.append(np.append(theta_new, iteration))
            if debug:
                print(f"new point: {theta_new} rejected")
        if progress:
            progress(iteration/iterations, False, int(time() - globals()["start_time"]), timeout)

        ## Finish iterations after timeout
        if (time() - globals()["start_time"]) > timeout > 0:
            globals()["mh_results"].last_iter = iteration
            globals()["mh_results"].time_it_took = time() - globals()["start_time"]
            break

    globals()["mh_results"].time_it_took = time() - globals()["start_time"]
    return np.array(accepted), np.array(rejected)


def initialise_sampling(space: RefinedSpace, data, functions, sample_size: int,  mh_sampling_iterations: int, eps,
                        theta_init=False, where=False, progress=False, burn_in=False, bins=20, timeout=False,
                        debug=False, metadata=True, draw_plot=False):
    """ Initialisation method for Metropolis Hastings

    Args:
        space (RefinedSpace): supporting structure, defining parameters, their domains and types
        data (list of numbers): data that we wish to model
        functions (list of strings):
        sample_size (int): total number of observations
        mh_sampling_iterations (int): number of iterations/steps in searching in space
        eps (number): very small value used as probability of non-feasible values in prior
        theta_init (list of numbers): initial parameter point
        where (tuple/list): output matplotlib sources to output created figure
        progress (Tkinter element or False): progress bar
        burn_in (number): fraction or count of how many samples will be trimmed from beginning
        bins (int): number of segments in the plot
        timeout (int): timeout in seconds
        debug (bool): if True extensive print will be used
        metadata (bool): if True metadata will be plotted
        draw_plot (Callable): function showing intermediate plots

    @author: tpetrov
    @edit: xhajnal
    """

    ## Internal settings
    start_time = time()
    globals()["start_time"] = start_time

    ##                     HastingsResults ( params, theta_init, accepted, rej observations_count, observations_samples_count, MH_sampling_iterations, eps, burn_in,      pretitle, title, bins, last_iter,  timeout, time_it_took, rescale):
    globals()["mh_results"] = HastingsResults(space.params, theta_init, [], [], sample_size, sample_size, mh_sampling_iterations, eps, burn_in=burn_in, title="", bins=bins, last_iter=0, timeout=timeout)

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

    # ## Maintaining observations
    # # If given observations or data
    # if data:
    #     ## Checking the type of observations (experiment/data)
    #     if len(data) > len(functions):
    #         ## Already given observations
    #         observations_count = len(data)
    #     else:
    #         ## Changing the data into observations
    #         spam = []
    #         index = 0
    #         for observation in data:
    #             for times in range(int(observation * float(sample_size))):
    #                 spam.append(index)
    #             index = index + 1
    #         data = spam
    # else:
    #     data_means = [eval(fi) for fi in functions]
    #     print("data means", data_means)
    #     samples = []
    #     for i in range(sample_size):
    #         samples.append(sample_functions(functions, data_means))
    #     print("samples", samples)
    #
    #     data = np.array(samples)[np.random.randint(0, sample_size, sample_size)]
    print("data", data)
    print("Initial parameter point: ", theta_init)

    ## MAIN LOOP
    ##                                      (likelihood_computer,    prior, transition_model,   param_init,       iterations,       space, data, sample_size, acceptance_rule, parameter_intervals, functions, eps, progress,          timeout,         debug):
    accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model_a, theta_init, mh_sampling_iterations, space, data, sample_size, acceptance, parameter_intervals, functions, eps, progress=progress, timeout=timeout, debug=debug)

    globals()["mh_results"].set_accepted(accepted)
    globals()["mh_results"].set_rejected(rejected)

    print("accepted.shape", accepted.shape)
    if len(accepted) == 0:
        print("Metropolis-Hastings, no accepted point found")
        return False

    ## Dumping results
    print(f"Set of accepted points is stored here: {tmp_dir}/accepted.p")
    pickle_dump(accepted, os.path.join(tmp_dir, f"accepted.p"))
    print(f"Set of rejected points is stored here: {tmp_dir}/rejected.p")
    pickle_dump(rejected, os.path.join(tmp_dir, f"rejected.p"))
    print(f"Whole class is stored here: {tmp_dir}/mh_class.p")
    pickle_dump(globals()["mh_results"], os.path.join(tmp_dir, f"mh_class.p"))

    ## Showing metadata visualisations
    if metadata:
        if not where:
            globals()["mh_results"].show_iterations()
            globals()["mh_results"].show_accepted()
        else:
            globals()["mh_results"].show_iterations(where=draw_plot)
            globals()["mh_results"].show_accepted(where=draw_plot)

    ## TODO make a option to set to see the unzoomed space - freaking hard
    ## "Currently hist2d calculates it's own axis limits, and any limits previously set are ignored." (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.hist2d.html)
    ## Option 2 - https://stackoverflow.com/questions/29175093/creating-a-log-linear-plot-in-matplotlib-using-hist2d
    ## No scale

    if where:
        return globals()["mh_results"]
    else:
        globals()["mh_results"].show_mh_heatmap(where=where)
