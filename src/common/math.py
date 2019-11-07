import math
from collections.abc import Iterable
import scipy.stats as st
from sympy import Interval
import numpy as np
from mpmath import mpi


def nCr(n, k):
    """ Return combinatorial number n take k

    Args
    ----------
    n: (int)
    k: (int)
    """

    f = math.factorial
    return f(n) / f(k) / f(n - k)


def catch_data_error(data, minimum, maximum):
    """ Corrects all data value to be in range min,max

    Args
    ----------
    data: (dictionary) structure of data
    minimum: (float) minimal value in data to be set to
    maximum: (float) maximal value in data to be set to

    """
    for n in data.keys():
        for i in range(len(data[n])):
            if data[n][i] < minimum:
                data[n][i] = minimum
            if data[n][i] > maximum:
                data[n][i] = maximum


def create_intervals(alpha, n_samples, data):
    """ Returns intervals of data_point +- margin

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data: (list of floats), values to be margined
    """
    intervals = []
    if not isinstance(data, Iterable):
        return [create_interval(alpha, n_samples, data)]
    for data_point in data:
        intervals.append(create_interval(alpha, n_samples, data_point))
    return intervals


def create_interval(alpha, n_samples, data_point):
    """ Returns interval of data_point +- margin

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data_point: (float), the value to be margined
    """
    delta = margin(alpha, n_samples, data_point)
    return Interval(data_point - delta, data_point + delta)


def margin(alpha, n_samples, data_point):
    """ Estimates expected interval with respect to parameters
    TBA shortly describe this type of margin

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data_point: (float), the value to be margined
    """
    return st.norm.ppf(1 - (1 - alpha) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples) + 0.5 / n_samples


def margin_experimental(alpha, n_samples, data_point):
    """ Estimates expected interval with respect to parameters
    This margin was used to produce the visual outputs for hsb19

    Args
    ----------
    alpha: (float) confidence interval to compute margin
    n_samples: (int) number of samples to compute margin
    data_point: (float), the value to be margined
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

    Args
    ----------
    region1: (list of pairs) (hyper)space defined by the regions
    region2: (list of pairs) (hyper)space defined by the regions
    """
    if len(region1) is not len(region2):
        print("The intervals does not have the same size")
        return False

    for dimension in range(len(region1)):
        if mpi(region1[dimension]) not in mpi(region2[dimension]):
            return False
    return True

