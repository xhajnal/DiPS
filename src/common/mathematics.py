import math
from collections.abc import Iterable
import scipy.stats as st
from sympy import Interval
import numpy as np
from mpmath import mpi
from numpy import prod


def nCr(n, k):
    """ Returns combinatorial number n take k

    Args:
        n (int):
        k (int):
    """
    if n < k:
        return 0
    f = math.factorial
    return f(n) / f(k) / f(n - k)


def catch_data_error(data, minimum, maximum):
    """ Corrects all data value to be in range min,max

    Args:
        data (dict or list): structure of data
        minimum (float): minimal value in data to be set to
        maximum (float): maximal value in data to be set to
    """
    if isinstance(data, dict):
        for n in data.keys():
            for i in range(len(data[n])):
                if data[n][i] < minimum:
                    data[n][i] = minimum
                if data[n][i] > maximum:
                    data[n][i] = maximum
    else:
        for i in range(len(data)):
            if data[i] < minimum:
                data[i] = minimum
            if data[i] > maximum:
                data[i] = maximum


def create_intervals(confidence, n_samples, data):
    """ Returns intervals of data_point +- margin

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data (list of floats): values to be margined
    """
    intervals = []
    if not isinstance(data, Iterable):
        return [create_interval(confidence, n_samples, data)]
    for data_point in data:
        intervals.append(create_interval(confidence, n_samples, data_point))
    return intervals


def create_interval_NEW(confidence, samples=False, n_samples=False, mean=False, s=False, true_std=False, is_prob=False, is_normal=False, side="both"):
    """ Returns confidence interval of mean based on samples

        Args:
            confidence (float): confidence level, C
            samples (list of numbers): samples
            n_samples (int): number of samples (not necessary if the samples are provided)
            mean (float): mean of samples (not necessary if the samples are provided)
            s (float): standard deviation of sample
            true_std (float): population standard deviation (sigma) (False if not known)
            is_prob (Bool): marks that the we estimate CI for probability values
            is_normal (Bool): marks that population follows normal distribution
            side (string): choose which side estimate you want ("both"/"left"/"right")

        Returns:
            confidence interval of mean
        """
    h = 0

    if samples:
        n_samples = len(samples)
        mean = float(np.mean(samples))
        s = float(np.std(samples))
        std_err = float(st.sem(samples))
    else:
        if s is False and n_samples < 30:
            raise Exception("confidence intervals", "Missing standard deviation to estimate mean with less than 30 samples.")
        else:
            std_err = s / math.sqrt(n_samples)

    if side == "both":
        alpha = (1 - confidence) / 2
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        t = st.t.ppf((1 + confidence) / 2, n_samples - 1)
    else:
        alpha = (1 - confidence)
        z = st.norm.ppf(1 - (1 - confidence))
        t = st.t.ppf((1 + confidence), n_samples - 1)

    if is_prob:  ## CI for probabilities
        if mean == 0:  ## Rule of three
            return Interval(0, 3/n_samples)
        elif mean == 1:  ## Rule of three
            return Interval(1 - 3/n_samples, 1)
        elif n_samples >= 30:  ##  Binomial proportion confidence interval: Normal/Gaussian distribution of the proportion: https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
            h = z * math.sqrt((mean * (1 - mean)) / n_samples)
        elif n_samples < 30:
            interval = st.bayes_mvs(samples, confidence)[0][1]  ## 0 is the mean, 1 is the interval estimate
            return Interval(interval[0], interval[1])
            ## h = t * math.sqrt((mean * (1 - mean)) / n_samples) ## TODO, check this
    else:      ## CI for usual values
        if (n_samples >= 30 or is_normal) and true_std is not False:  ## use Normal Distribution
            h = z * true_std / math.sqrt(n_samples)
        elif is_normal:  ## use Student distribution
            # h = t * s / math.sqrt(n_samples)
            h = t * std_err
        else:
            interval = st.bayes_mvs(samples, confidence)[0][1]  ## 0 is the mean, 1 is the interval estimate
            return Interval(interval[0], interval[1])

    h = float(h)
    if side == "both":
        return Interval(mean - h, mean + h)
    elif side == "right":
        if is_prob:
            return Interval(0, mean + h)
        else:
            return Interval(float('-inf'), mean + h)
    else:
        if is_prob:
            return Interval(mean - h, 1)
        else:
            return Interval(mean - h, float('inf'))


def create_interval(confidence, n_samples, data_point):
    """ Returns interval of probabilistic data_point +- margin

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined from interval [0,1]
    """
    delta = margin(confidence, n_samples, data_point)
    return Interval(float(max(data_point - delta, 0)), float(min(data_point + delta, 1)))


## TODO shortly describe this type of margin
def margin(confidence, n_samples, data_point):
    """ Estimates expected interval with respect to parameters

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined
    """
    try:
        return st.norm.ppf(1 - (1 - confidence) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples) + 0.5 / n_samples
    except ValueError as error:
        raise Exception("Unable to compute the margins. Please, check whether each data point in domain [0,1]")


def margin_experimental(confidence, n_samples, data_point):
    """ Estimates expected interval with respect to parameters
    This margin was used to produce the visual outputs for hsb19

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float):, the value to be margined
    """
    return st.norm.ppf(1 - (1 - confidence) / 2) * math.sqrt(
        data_point * (1 - data_point) / n_samples) + 0.5 / n_samples + 0.005


def cartesian_product(*arrays):
    """ Returns a product of given list of arrays
    """

    la = len(arrays)
    if la == 0:
        return np.array([])
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def is_in(region1, region2):
    """ Returns True if the region1 is in the region2, returns False otherwise

    Args:
        region1 (list of pairs): (hyper)space defined by the regions
        region2 (list of pairs): (hyper)space defined by the regions
    """
    if len(region1) is not len(region2):
        print("The intervals does not have the same size")
        return False

    for dimension in range(len(region1)):
        if mpi(region1[dimension]) not in mpi(region2[dimension]):
            return False
    return True


def get_rectangle_volume(rectangle):
    """ Computes the volume of the given (hyper)rectangle

    Args:
        rectangle:  (list of intervals) defining the (hyper)rectangle
    """
    intervals = []
    ## If there is empty rectangle
    if not rectangle:
        raise Exception("empty rectangle has no volume")
    for interval in rectangle:
        intervals.append(interval[1] - interval[0])

    product = prod(intervals)
    if isinstance(product, np.float64):
        product = float(product)

    return product


def create_matrix(sample_size, dim):
    """ Return **dim** dimensional array of length **sample_size** in each dimension

    Args:
        sample_size (int): number of samples in dimension
        dim (int): number of dimensions

    """
    return np.array(private_create_matrix(sample_size, dim, dim))


def private_create_matrix(sample_size, dim, n_param):
    """ Return **dim** dimensional array of length **sample_size** in each dimension

    Args:
        sample_size (int): number of samples in dimension
        dim (int): number of dimensions
        n_param (int): dummy parameter

    @author: xtrojak, xhajnal
    """
    if dim == 0:
        point = []
        for i in range(n_param):
            point.append(0)
        return [point, 9]
    return [private_create_matrix(sample_size, dim - 1, n_param) for _ in range(sample_size)]
