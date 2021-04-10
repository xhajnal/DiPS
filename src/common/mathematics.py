import math
import multiprocessing
import warnings
from collections.abc import Iterable
from statsmodels.stats.proportion import proportion_confint
import scipy.stats as st
from sympy import Interval, factor
import numpy as np


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


def correct_data_values(data, minimum, maximum):
    """ Corrects all data value to be in range [min,max]

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


def create_intervals_hsb(confidence, n_samples, data):
    """ Returns intervals of data_point +- margin

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data (list of floats): values to be margined
    """
    print(data)
    intervals = []
    if not isinstance(data, Iterable):
        assert isinstance(data, float)
        return [create_interval_hsb(confidence, n_samples, data)]
    for data_point in data:
        try:
            assert isinstance(data_point, float)
        except AssertionError:
            data_point = float(data_point)
        intervals.append(create_interval_hsb(confidence, n_samples, data_point))
    return intervals


def create_interval_NEW(confidence, samples=False, n_samples=False, sample_mean=False, sd=False, true_std=False, is_prob=False, is_normal=False, side="both"):
    """ Returns confidence interval of mean based on samples

        Args:
            confidence (float): confidence level, C
            samples (list of numbers): samples
            n_samples (int): number of samples (not necessary if the samples are provided)
            sample_mean (float): mean of samples (not necessary if the samples are provided)
            sd (float): standard deviation of sample
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
        sample_mean = float(np.mean(samples))
        sd = float(np.std(samples))
        std_err = float(st.sem(samples))
    else:
        if sd is False and n_samples < 30:
            raise Exception("confidence intervals", "Missing standard deviation to estimate mean with less than 30 samples.")
        else:
            std_err = sd / math.sqrt(n_samples)

    if side == "both":
        alpha = (1 - confidence) / 2
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        t = st.t.ppf((1 + confidence) / 2, n_samples - 1)
    else:
        alpha = (1 - confidence)
        z = st.norm.ppf(1 - (1 - confidence))
        t = st.t.ppf((1 + confidence), n_samples - 1)

    if is_prob:  ## CI for probabilities
        if sample_mean == 0:  ## Rule of three
            return Interval(0, 3/n_samples)
        elif sample_mean == 1:  ## Rule of three
            return Interval(1 - 3/n_samples, 1)
        elif n_samples >= 30:  ##  Binomial proportion confidence interval: Normal/Gaussian distribution of the proportion: https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
            h = z * math.sqrt((sample_mean * (1 - sample_mean)) / n_samples)
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
        return Interval(sample_mean - h, sample_mean + h)
    elif side == "right":
        if is_prob:
            return Interval(0, sample_mean + h)
        else:
            return Interval(float('-inf'), sample_mean + h)
    else:
        if is_prob:
            return Interval(sample_mean - h, 1)
        else:
            return Interval(sample_mean - h, float('inf'))


def create_proportions_interval(confidence, n_samples, data_point, method="AC"):
    """ Returns confidence interval of given point, n
        using 6 methods:
            3/N             - rule of three
            CLT             - Central Limit Theorem confidence intervals / Wald method
            AC              - Agresti-Coull method (default)
            Wilson          - Wilson Score method
            Clopper_pearson - Clopper-Pearson interval based on Beta distribution
            Jeffreys        - Jeffreys Bayesian Interval
        plus a backward compatibility for hsb19's CLT method with a correction term

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples
        data_point (float): the value to be margined from interval [0,1]
        method (string): method to compute the confidence intervals with 3/N, CLT, AC (default), Wilson, Clopper-Pearson, Jeffreys, hsb
    """
    if data_point > 1 or data_point < 0:
        raise Exception("create_proportions_interval cannot be used for value outside of range [0,1].")

    if method.lower() == "clt" or method.lower == "wald":
        clt_margin = st.norm.ppf(1 - (1 - confidence) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples)
        clt = float(max(data_point - clt_margin, 0)), float(min(data_point + clt_margin, 1))
        return Interval(*clt)
    elif "3" in method or "three" in method:
        rule_of_three_margin = 3/n_samples
        rule_of_three = Interval(float(max(data_point - rule_of_three_margin, 0)), float(min(data_point + rule_of_three_margin, 1)))
        return rule_of_three
    elif method.lower() == "ac" or "agresti" in method.lower():
        AC = proportion_confint(round(data_point*n_samples), n_samples, alpha=1 - confidence, method="agresti_coull")
        return Interval(*AC)
    elif method.lower() == "wilson":
        wilson = proportion_confint(round(data_point*n_samples), n_samples, alpha=1 - confidence, method="wilson")
        return Interval(*wilson)
    elif "clop" in method.lower() or "pear" in method.lower():
        clopper_pearson = proportion_confint(round(data_point*n_samples), n_samples, alpha=1 - confidence, method="beta")
        return Interval(*clopper_pearson)
    elif "jef" in method.lower():
        jeffreys = proportion_confint(round(data_point*n_samples), n_samples, alpha=1 - confidence, method="jeffreys")
        return Interval(*jeffreys)
    elif "hsb" in method.lower():
        return create_interval_hsb(confidence, n_samples, data_point)
    else:
        raise Exception("Method mot found.")


def create_broadest_interval(confidence, n_samples, data_point):
    """ Returns broadest interval of probabilistic data_point +- margin
        using 6 methods:
            3/N             - rule of three
            CLT             - Central Limit Theorem confidence intervals, Wald
            AC              - Agresti-Coull method
            wilson          - Wilson Score method
            clopper_pearson - Clopper-Pearson interval based on Beta distribution
            jeffreys        - Jeffreys Bayesian Interval

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined from interval [0,1]
    """
    clt = create_proportions_interval(confidence, n_samples, data_point, "CLT")
    rule_of_three = create_proportions_interval(confidence, n_samples, data_point, "3")

    if clt.is_proper_subset(rule_of_three):
        print(f"clt {clt} in rule_of_three {rule_of_three}")
        biggest = rule_of_three
        biggest_name = "rule_of_three"
    elif rule_of_three.is_proper_subset(clt):
        print(f"rule_of_three {rule_of_three} in clt {clt}")
        biggest = clt
        biggest_name = "clt"
    else:
        raise Exception(f"Cannot make the biggest interval, rule of three {rule_of_three} is not in clt {clt} or vise-versa")

    AC = create_proportions_interval(confidence, n_samples, data_point, "AC")

    if AC.is_proper_subset(biggest):
        print(f"AC {AC} in the biggest interval so far {biggest}, skipping")
    elif biggest.is_proper_subset(AC):
        print(f"the biggest interval so far {biggest} in AC {AC}, using it instead")
        biggest = AC
        biggest_name = "AC"
    else:
        raise Exception(
            f"Cannot make the biggest interval, AC {AC} is not in the biggest interval so far {biggest} or vise-versa")

    wilson = create_proportions_interval(confidence, n_samples, data_point, "wilson")

    if wilson.is_proper_subset(biggest):
        print(f"wilson {wilson} in the biggest interval so far {biggest}, skipping")
    elif biggest.is_proper_subset(wilson):
        print(f"the biggest interval so far {biggest} in wilson {wilson}, using it instead")
        biggest = wilson
        biggest_name = "wilson"
    else:
        raise Exception(
            f"Cannot make the biggest interval, wilson {wilson} is not in the biggest interval so far {biggest} or vise-versa")

    clopper_pearson = create_proportions_interval(confidence, n_samples, data_point, "clop")

    if clopper_pearson.is_proper_subset(biggest):
        print(f"clopper_pearson {clopper_pearson} in the biggest interval so far {biggest}, skipping")
    elif biggest.is_proper_subset(clopper_pearson):
        print(f"the biggest interval so far {biggest} in clopper_pearson {clopper_pearson}, using it instead")
        biggest = clopper_pearson
        biggest_name = "Clopper-Pearson"
    else:
        raise Exception(f"Cannot make the biggest interval, clopper_pearson {clopper_pearson} is not in the biggest interval so far {biggest} or vise-versa")

    jeffreys = create_proportions_interval(confidence, n_samples, data_point, "jef")

    if jeffreys.is_proper_subset(biggest):
        print(f"jeffreys {jeffreys} in the biggest interval so far {biggest}, skipping")
    elif biggest.is_proper_subset(jeffreys):
        print(f"the biggest interval so far {biggest} in jeffreys {jeffreys}, using it instead")
        biggest = jeffreys
        biggest_name = "Jeffreys"
    else:
        raise Exception(
            f"Cannot make the biggest interval, jeffreys {jeffreys} is not in the biggest interval so far {biggest} or vise-versa")

    print(f"Given point, {data_point} broadest interval is {biggest} using {biggest_name} method")
    return biggest


def create_interval_hsb(confidence, n_samples, data_point):
    """ Returns interval of probabilistic data_point +- margin using margin function

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined from interval [0,1]
    """
    if n_samples == 0:
        return Interval(0, 1)
    delta = margin_hsb(confidence, n_samples, data_point)
    return Interval(float(max(data_point - delta, 0)), float(min(data_point + delta, 1)))


def margin_hsb(confidence: float, n_samples: int, data_point: float):
    """ Confidence intervals with CLT/Wald method with fixing term

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float): the value to be margined
    """
    assert isinstance(confidence, float)
    assert isinstance(n_samples, int)
    try:
        assert isinstance(data_point, float)
    except AssertionError:
        data_point = float(data_point)
    try:
        return st.norm.ppf(1 - (1 - confidence) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples) + 0.5 / n_samples
    except ValueError as error:
        raise Exception("Unable to compute the margins. Please, check whether each data point in domain [0,1]")


def margin_experimental(confidence, n_samples, data_point):
    """ Confidence intervals with CLT/Wald method with two fixing terms
    This margin was used to produce the visual outputs for hsb19

    Args:
        confidence (float): confidence level, C
        n_samples (int): number of samples to compute margin
        data_point (float):, the value to be margined
    """
    return st.norm.ppf(1 - (1 - confidence) / 2) * math.sqrt(data_point * (1 - data_point) / n_samples) + 0.5 / n_samples + 0.005


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


def weight_list(spam, weights, warn=True):
    """ Returns weighted list

    Args:
        spam(list): list to multiply with weights
        weights (list): of weights to multiply the respective distance with
        warn (bool): if warn, it will warn instead of raising error

    Returns:
        (list): weighted list
    """

    if warn:
        if len(weights) > len(spam):
            warnings.warn("The list of weights is longer than the list, last weights are not used!!", RuntimeWarning)

        if len(weights) > len(spam):
            warnings.warn("The list of weights is shorter than the list, last items are not weighted!!", RuntimeWarning)

    try:
        for index, item in enumerate(spam):
            spam[index] = float(spam[index]) * float(weights[index])
    except IndexError:
        pass
    return spam


def bar(expression, index, return_dict):
    """ Private method of simplify_functions_parallel """
    return_dict[index] = factor(expression)
    print(index)


def simplify_functions_parallel(expressions):
    """ Factorise the given list of expressions

    Args
        expressions: (list) of expressions to factorise in parallel
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    factorised_expressions = []
    for index, item in enumerate(expressions):
        # print("item ", item, "index", index)
        processes.append(multiprocessing.Process(target=bar, args=(item, index, return_dict)))
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    ## Showing the difference in an example:
    # print(return_dict)
    for i in sorted(return_dict.keys()):
        factorised_expressions.append(return_dict[i])
    return factorised_expressions
