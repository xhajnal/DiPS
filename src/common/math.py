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
    f = math.factorial
    return f(n) / f(k) / f(n - k)


def catch_data_error(data, minimum, maximum):
    """ Corrects all data value to be in range min,max

    Args:
        data (dict): structure of data
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


def create_intervals(alpha, n_samples, data):
    """ Returns intervals of data_point +- margin

    Args:
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        data (list of floats): values to be margined
    """
    intervals = []
    if not isinstance(data, Iterable):
        return [create_interval(alpha, n_samples, data)]
    for data_point in data:
        intervals.append(create_interval(alpha, n_samples, data_point))
    return intervals


def create_interval(alpha, n_samples, data_point):
    """ Returns interval of data_point +- margin

    Args:
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined
    """
    delta = margin(alpha, n_samples, data_point)
    return Interval(data_point - delta, data_point + delta)


## TODO shortly describe this type of margin
def margin(alpha, n_samples, data_point):
    """ Estimates expected interval with respect to parameters

    Args:
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined
    """
    return st.norm.ppf(1 - (1 - alpha) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples) + 0.5 / n_samples


def margin_experimental(alpha, n_samples, data_point):
    """ Estimates expected interval with respect to parameters
    This margin was used to produce the visual outputs for hsb19

    Args:
        alpha (float): confidence interval to compute margin
        n_samples (int): number of samples to compute margin
        data_point (float):, the value to be margined
    """
    return st.norm.ppf(1 - (1 - alpha) / 2) * math.sqrt(
        data_point * (1 - data_point) / n_samples) + 0.5 / n_samples + 0.005


def cartesian_product(*arrays):
    """ Returns a product of given list of arrays
    """

    la = len(arrays)
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