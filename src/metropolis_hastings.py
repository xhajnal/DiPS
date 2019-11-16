import os
import configparser
from time import time
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
import pickle

from common.document_wrapper import DocumentWrapper
from space import RefinedSpace


config = configparser.ConfigParser()
print(os.getcwd())

workspace = os.path.dirname(__file__)
# print("workspace", workspace)
cwd = os.getcwd()
os.chdir(workspace)


config.read(os.path.join(workspace, "../config.ini"))
tmp_dir = config.get("paths", "tmp")
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

os.chdir(cwd)
wrapper = DocumentWrapper(width=75)


def sample(functions, data_means):
    """ Will sample according to the pdf as given by the polynomials

    Returns:
         ## TODO

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
    ## So far for 2 parameters
    sd = 0.3  ## Standard deviation of the normal distribution
    theta_new = np.zeros(2)

    temp = parameter_intervals[0][0] - 1  ## Lower bound of first parameter - 1
    while (temp <= parameter_intervals[0][0]) or (temp >= parameter_intervals[0][1]):
        temp = np.random.normal(theta[0], sd)
    theta_new[0] = temp

    temp = parameter_intervals[1][0] - 1  ## Lower bound of second parameter - 1
    while (temp <= parameter_intervals[1][0]) or (temp >= parameter_intervals[1][1]):
        temp = np.random.normal(theta[1], sd)
    theta_new[1] = temp

    return theta_new


def prior(x, eps):
    """ TODO
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
        return accept < (np.exp(x_new - x))


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, space, data,
                        acceptance_rule, parameter_intervals, functions, eps, progress=False, debug=False):
    """ TODO
    the main method

        likelihood_computer (function(x, data)): returns the likelihood that these parameters generated the data
        prior (function(x, eps)): prior function
        transition_model (function(x)): a function that draws a sample from a symmetric distribution and returns it
        param_init  (pair of numbers): a starting sample
        iterations (int): number of accepted to generated
        data (list of numbers):: the data that we wish to model
        acceptance_rule (function(x, x_new)): decides whether to accept or reject the new sample
        parameter_intervals (list of pairs): boundaries of parameters
        progress (Tkinter element): progress bar
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
        x_lik = likelihood_computer(space, x, functions, data, eps)
        x_new_lik = likelihood_computer(space, x_new, functions, data, eps)
        if debug:
            print("iteration:", iteration)

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

    return np.array(accepted), np.array(rejected)


def manual_log_like_normal(space, theta, functions, observations, eps):
    """ TODO

    Args:
        space (Refined space):
        theta (list): parameter values
        functions (list of strings):
        observations (list of numbers):
        eps (number): very small value used as probability of non-feasible values in prior

    Returns:
         ## TODO

    @author: tpetrov
    @edit: xhajnal
    """
    res = 0
    # print("data", data)
    # print("functions", functions)

    locals()[space.get_params()[0]] = theta[0]
    locals()[space.get_params()[1]] = theta[1]

    for data_point in observations:  # observations:
        # print("data_point", data_point)
        # print("functions[data_point]", functions[data_point])
        # print("x=", globals()["x"])
        # print("y=", globals()["y"])
        # print(eval("x+y"))
        temp = eval(functions[data_point])

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


def initialise_sampling(space: RefinedSpace, observations, functions, N: int, N_obs: int, MH_samples: int, eps,
                        theta_init=False, where=False, progress=False, debug=False):
    """ Initialisation method for Metropolis Hastings
    Args:
        space (RefinedSpace):
        observations (list of numbers): either experiment or data(experiment result frequency)
        functions (list of strings):
        N (number): total data amount
        N_obs (number): number of samples
        MH_samples (number): number of iterations
        eps (number): very small value used as probability of non-feasible values in prior
        theta_init (list of numbers): initial point in parameter space
        where (tuple/list): output matplotlib sources to output created figure
        progress (Tkinter element): progress bar
        debug (bool): if True extensive print will be used

    @author: tpetrov
    @edit: xhajnal
    """
    ## Internal settings
    start_time = time()

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
        theta_init = [(parameter_intervals[0][0] + parameter_intervals[0][1])/2, (parameter_intervals[1][0] + parameter_intervals[1][1])/2]  ## Middle of the intervals # np.ones(10)*0.1

    for index, param in enumerate(space.get_params()):
        globals()[param] = theta_init[index]
        print(f"{param} = {theta_init[index]}")

    ## Maintaining observations
    if observations:
        ## Checking the type of observations (experiment/data)
        if len(observations) > len(functions):
            ## Already given observations
            N = len(observations)
        else:
            ## Changing the data into observations
            spam = []
            index = 0
            for observation in observations:
                for times in range(int(observation * float(N))):
                    spam.append(index)
                index = index + 1
            observations = spam
    else:
        data_means = [eval(fi) for fi in functions]
        print("data means", data_means)
        samples = []
        for i in range(N):
            samples.append(sample(functions, data_means))
        print("samples", samples)

        observations = np.array(samples)[np.random.randint(0, N, N_obs)]
    print("observations", observations)
    N_obs = min(N, N_obs)

    if not where:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(observations, bins=range(len(functions)))
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Figure 1: Distribution of {N_obs} observations (from full sample= {N})")
        plt.show()

    # theta_new = transition_model_a(theta_true, parameter_intervals)     ## apparently just a print call
    # r = prior(theta_true, eps)
    # np.log(r)  ## another print call
    # res = manual_log_like_normal(space, theta_true, functions, np.array(observations), eps)

    print("Initial parameter point: ", theta_init)

    ##                                      (likelihood_computer,    prior, transition_model,   param_init, iterations, space, data,    acceptance_rule,parameter_intervals, functions, eps):
    accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model_a, theta_init, MH_samples, space, observations, acceptance, parameter_intervals, functions, eps, progress=progress, debug=debug)

    print(f"Set of accepted and rejected points is stored here: {tmp_dir}")
    pickle.dump(accepted, open(os.path.join(tmp_dir, f"accepted_{N_obs}.p"), 'wb'))
    pickle.dump(rejected, open(os.path.join(tmp_dir, f"rejected_{N_obs}.p"), 'wb'))

    # accepted = pickle.load(open(os.path.join(tmp_dir, f"accepted_{N_obs}.p"), "rb"))
    # rejected = pickle.load(open(os.path.join(tmp_dir, f"rejected_{N_obs}.p"), "rb"))

    # print("accepted", accepted)
    # to_show = accepted.shape[0]
    # print("accepted[100:to_show, 1]", accepted[100:to_show, 1])
    # print("rejected", rejected)

    if not where:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 1, 1)
        to_show = accepted.shape[0]
        ax.plot(rejected[(0, 100)[len(rejected) > 200]:to_show, 1], 'rx', label='Rejected', alpha=0.5)
        ax.plot(accepted[(0, 100)[len(accepted) > 200]:to_show, 1], 'b.', label='Accepted', alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("Figure 2: MCMC sampling for N=" + str(N_obs) + " with Metropolis-Hastings. First " + str(to_show) + " samples are shown.")
        ax.grid()
        ax.legend()
        plt.show()

    show = int(-0.75 * accepted.shape[0])  ## TODO check this line
    if not where:
        hist_show = int(-0.75 * accepted.shape[0])
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(accepted[show:, 0])
        ax.set_title(f"Figure 4: Trace for {space.get_params()[0]}")
        ax.set_ylabel(f"${space.get_params()[0]}$")
        ax = fig.add_subplot(1, 2, 2)
        ax.hist(accepted[hist_show:, 0], bins=20, density=True)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(f"${space.get_params()[0]}$")
        ax.set_title(f"Fig.5: Histogram of ${space.get_params()[0]}$")
        fig.tight_layout()

    ## Create the heat map
    print("accepted.shape", accepted.shape)
    if len(accepted) == 0:
        return False, False
    heatmap_title = f'{space.get_params()[0]}, {space.get_params()[1]} estimate with MH algorithm, {MH_samples} iterations, sample size = {N_obs} \n It took {gethostname()} {round(time() - start_time)} second(s)'
    if where:
        plt.hist2d(accepted[show:, 0], accepted[show:, 1], bins=20)
        plt.xlabel(space.get_params()[0])
        plt.ylabel(space.get_params()[1])
        plt.title("\n".join(wrapper.wrap(heatmap_title)))
        where[1] = plt.colorbar()
        return where[0], where[1]
    else:
        plt.figure(figsize=(12, 6))
        plt.hist2d(accepted[show:, 0], accepted[show:, 1], bins=20)
        plt.colorbar()
        plt.xlabel(space.get_params()[0])
        plt.ylabel(space.get_params()[1])
        plt.title(heatmap_title)
        plt.show()
