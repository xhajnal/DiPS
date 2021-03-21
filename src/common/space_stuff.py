import copy
import operator
from fractions import Fraction
from functools import reduce
from typing import Iterable

import numpy as np
from mpmath import mpi

from common.mathematics import cartesian_product
from rectangle import My_Rectangle


def rectangular_hull(points):
    """ Creates a single smallest (hyper)rectangle wrapping all the points"""
    assert isinstance(points, Iterable)
    min_point = []
    max_point = []
    ## TODO can be optimised
    for index in range(len(points[0])):
        min_point.append(min(list(map(lambda x: x[index], points))))
        max_point.append(max(list(map(lambda x: x[index], points))))
    return points_to_rectangle([min_point, max_point])


def points_to_rectangle(points):
    """ Converts set of endpoint into (hyper)rectangle"""
    rectangle = []
    for dimension in range(len(points[0])):
        spam = []
        for point in points:
            spam.append(point[dimension])
        rectangle.append(spam)

    return rectangle


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
    if isinstance(rectangle, My_Rectangle):
        rectangle = rectangle.region

    ## If there is empty rectangle
    if not rectangle:
        raise Exception("Empty rectangle has no volume")
    for interval in rectangle:
        try:
            intervals.append(Fraction(str(interval[1])) - Fraction(str(interval[0])))
        except Exception as err:
            print(interval)
            raise err

    ## Python 3.8+
    product = reduce(operator.mul, intervals, 1)
    ## Python 3.1-7
    # product = prod(intervals)

    # if isinstance(product, np.float64):
    #     product = float(product)

    return product


def split_by_longest_dimension(region):
    """ Splits given region, (hyper)rectangle, into two by splitting the longest dimension in halves """
    index, maximum = 0, 0
    for i in range(len(region)):
        value = region[i][1] - region[i][0]
        if value > maximum:
            index = i
            maximum = value
    low = region[index][0]
    high = region[index][1]

    ## Compute the half of region
    threshold = low + (high - low) / 2

    ##  Update space
    model_low = copy.deepcopy(region)
    model_low[index] = [low, threshold]
    model_high = copy.deepcopy(region)
    model_high[index] = [threshold, high]
    return [model_low, model_high, index, threshold]


def split_by_all_dimensions(region):
    """ Splits given hyper rectangle into set of rectangles by splitting in each dimension"""
    thresholds = []
    for index, dimension in enumerate(region):
        ## Compute the half of dimension
        low = dimension[0]
        high = dimension[1]
        thresholds.append(low + (high - low) / 2)

    rectangles = []
    # print("len", "{0:b}".format(2**(len(region)-1)))
    for index in range(2**(len(region))):
        # print("index", index)
        rectangle = []
        # print("binary", "{0:b}".format(index + 2**(len(region)))[1:])
        for dim_index, dimension in enumerate("{0:b}".format(index + 2**(len(region)))[1:]):
            # print("dim_index", dim_index)
            if dimension == "0":
                rectangle.append([region[dim_index][0], thresholds[dim_index]])
            else:
                rectangle.append([thresholds[dim_index], region[dim_index][1]])
        # print("rectangle", rectangle)
        rectangles.append(rectangle)
    return rectangles


def split_by_samples(region, sat_list, unsat_list, sample_size, debug=False):
    """ Splits given rectangle into smaller ones such as rectangles contain only sat or unsat samples
        exception is type III, when rectangular hulls of sat and unsat samples are equal,
        we split by all dimensions in halves

        type I: rectangular hulls have no Intersections
        type II: one of the rectangular hull is a true subspace
        type III: hulls are equal (sat and unsat points are "well mixed")

    Args:
        region:  (list of intervals) defining the (hyper)rectangle
        sat_list: (list of points) list of sat points within the region
        unsat_list: (list of points) list of unsat points within the region
        sample_size: (int) number of samples per dimension
        debug (bool): if True extensive print will be used

    :return: (list of rectangles) rectangle to which the region was split into
    """

    rectangle_of_sats = rectangular_hull(sat_list)
    print("rectangle_of_sats", rectangle_of_sats) if debug else None
    rectangle_of_sats = expand_rectangle(rectangle_of_sats, region, [sample_size] * len(region))
    print("expanded rectangle_of_sats", rectangle_of_sats) if debug else None
    rectangle_of_unsats = rectangular_hull(unsat_list)
    print("rectangle_of_unsats", rectangle_of_unsats) if debug else None
    rectangle_of_unsats = expand_rectangle(rectangle_of_unsats, region, [sample_size] * len(region))
    print("expanded rectangle_of_unsats", rectangle_of_unsats) if debug else None

    ## TODO this can be improved by second splitting of the smaller rectangle
    if rectangle_of_unsats == rectangle_of_sats:
        regions = split_by_all_dimensions(region)
        if len(regions) == 1:
            print(f"Type III: region: {region}: \n sat samples: {sat_list}, \n unsat samples: {unsat_list} \n {rectangle_of_sats}, {rectangle_of_unsats}")
    elif is_in(rectangle_of_sats, rectangle_of_unsats):
        regions = refine_by(rectangle_of_unsats, rectangle_of_sats, debug=False)
        if len(regions) == 1:
            print(f"Type II: region: {region}: \n sat samples: {sat_list}, \n unsat samples: {unsat_list} \n {rectangle_of_sats}, {rectangle_of_unsats}")
    elif is_in(rectangle_of_unsats, rectangle_of_sats):
        regions = refine_by(rectangle_of_sats, rectangle_of_unsats, debug=False)
        if len(regions) == 1:
            print(f"Type II: {rectangle_of_sats}, {rectangle_of_unsats}")
    else:
        regions = [rectangle_of_sats, rectangle_of_unsats]
        if len(regions) == 1:
            print(f"Type I: {rectangle_of_sats}, {rectangle_of_unsats}")

        ## TODO this can happen when e.g. all sat are on left and all unsat on right side
        # raise NotImplementedError(f'Splitting for this "weird" sampling result not implemented so far with region {region}, rectangle of sat points {rectangle_of_sats}, rectangle of unsat points {rectangle_of_unsats}')
    print(f"By sampling we split the region into regions: {regions}") if debug else None
    return regions


def expand_rectangle(rectangle, region, sample_sizes):
    """ Expands the rectangle half of the size of sampling size respecting boundaries """
    for index, dimension in enumerate(rectangle):
        ## half of the rectangle size * dimension size / number of rectangles in dimension (sample size minus 1)
        delta = 1/2*(region[index][1]-region[index][0])/(sample_sizes[index]-1)
        if dimension[0] > region[index][0]:
            ## New value of rectangle's lower value of dimension
            rectangle[index][0] = rectangle[index][0] - delta
        if dimension[1] < region[index][1]:
            ## New value of rectangle's upper value of dimension
            rectangle[index][1] = rectangle[index][1] + delta
    return rectangle


def refine_by(region1, region2, debug=False):
    """ Returns the first (hyper)space refined/spliced by the second (hyperspace) into orthogonal subspaces

    Args:
        region1 (list of pairs): (hyper)space defined by the regions
        region2 (list of pairs): (hyper)space defined by the regions
        debug (bool): if True extensive print will be used
    """

    if not is_in(region2, region1):
        raise Exception(f"The first region {region1} is not within the second {region2}, it cannot be refined/spliced properly")

    region1 = copy.deepcopy(region1)
    regions = []
    ## For each dimension trying to cut of the space
    for dimension in range(len(region2)):
        ## LEFT
        if region1[dimension][0] < region2[dimension][0]:
            sliced_region = copy.deepcopy(region1)
            sliced_region[dimension][1] = region2[dimension][0]
            if debug:
                print("left ", sliced_region)
            regions.append(sliced_region)
            region1[dimension][0] = region2[dimension][0]
            if debug:
                print("new intervals", region1)

        ## RIGHT
        if region1[dimension][1] > region2[dimension][1]:
            sliced_region = copy.deepcopy(region1)
            sliced_region[dimension][0] = region2[dimension][1]
            if debug:
                print("right ", sliced_region)
            regions.append(sliced_region)
            region1[dimension][1] = region2[dimension][1]
            if debug:
                print("new intervals", region1)

    # print("region1 ", region1)
    regions.append(region1)
    return regions


def refine_into_rectangles(sampled_space, silent=True):
    """ Refines the sampled space into hyperrectangles such that rectangle is all sat or all unsat

    Args:
        sampled_space: (space.RefinedSpace): space
        silent (bool): if silent printed output is set to minimum

    Returns:
        Hyperectangles of length at least 2 (in each dimension)
    """
    sample_size = len(sampled_space[0])
    dimensions = len(sampled_space.shape) - 1
    if not silent:
        print("\n refine into rectangles here ")
        print(type(sampled_space))
        print("shape", sampled_space.shape)
        print("space:", sampled_space)
        print("sample_size:", sample_size)
        print("dimensions:", dimensions)
    # find_max_rectangle(sampled_space, [0, 0])

    if dimensions == 2:
        parameter_indices = []
        for param in range(dimensions):
            parameter_indices.append(np.asarray(range(0, sample_size)))
        parameter_indices = cartesian_product(*parameter_indices)
        if not silent:
            print(parameter_indices)
        spam = []
        for point in parameter_indices:
            # print("point", point)
            result = find_max_rectangle(sampled_space, point, silent=silent)
            if result is not None:
                spam.append(result)
        if not silent:
            print(spam)
        return spam
    else:
        print(f"Sorry, {dimensions} dimensions TBD")


def find_max_rectangle(sampled_space, starting_point, silent=True):
    """ Finds the largest hyperrectangles such that rectangle is all sat or all unsat from starting point in positive direction

    Args:
        sampled_space: sampled space
        starting_point (list of floats): a point in the space to start search in
        silent (bool): if silent printed output is set to minimum

    Returns:
        (triple) : (starting point, end point, is_sat)
    """
    assert isinstance(sampled_space, (np.ndarray, np.generic))
    sample_size = len(sampled_space[0])
    dimensions = len(sampled_space.shape) - 1
    if dimensions == 2:
        index_x = starting_point[0]
        index_y = starting_point[1]
        length = 2
        start_value = sampled_space[index_x][index_y][1]
        if not silent:
            print("Dealing with 2D space at starting point", starting_point, "with start value", start_value)
        if start_value == 2:
            if not silent:
                print(starting_point, "already added, skipping")
            return
        if index_x >= sample_size - 1 or index_y >= sample_size - 1:
            if not silent:
                print(starting_point, "is at the border, skipping")
            sampled_space[index_x][index_y][1] = 2
            return

        ## While other value is found
        while True:
            ## print(index_x+length)
            ## print(sampled_space[index_x:index_x+length, index_y:index_y+length])
            values = list(map(lambda x: [y[1] for y in x], sampled_space[index_x:index_x + length, index_y:index_y + length]))
            # print(values)
            foo = []
            for x in values:
                for y in x:
                    foo.append(y)
            values = foo
            if not silent:
                print("Values found: ", values)
            if (not start_value) in values:
                length = length - 1
                if not silent:
                    print(f"rectangle [[{index_x},{index_y}],[{index_x + length},{index_y + length}]] does not satisfy all sat not all unsat")
                break
            elif index_x + length > sample_size or index_y + length > sample_size:
                if not silent:
                    print(f"rectangle [[{index_x},{index_y}],[{index_x + length},{index_y + length}]] is out of box, using lower value")
                length = length - 1
                break
            else:
                length = length + 1

        ## Mark as seen (only this point)
        sampled_space[index_x][index_y][1] = 2
        length = length - 1

        ## Skip if only this point safe/unsafe
        if length == 0:
            if not silent:
                print("Only single point found, skipping")
            return

        ## print((sampled_space[index_x, index_y], sampled_space[index_x+length-2, index_y+length-2]))

        # print(type(sampled_space))
        # place(sampled_space, sampled_space==False, 2)
        # print("new sampled_space: \n", sampled_space)

        # print("the space to be marked: \n", sampled_space[index_x:(index_x + length - 1), index_y:(index_y + length - 1)])
        if not silent:
            print("length", length)

        ## old result (in corner points format)
        # result = (sampled_space[index_x, index_y], sampled_space[index_x + length - 1, index_y + length - 1])

        ## new result (in region format)
        result = ([[sampled_space[index_x, index_y][0][0], sampled_space[index_x + length, index_y][0][0]],
                   [sampled_space[index_x, index_y][0][1], sampled_space[index_x, index_y + length][0][1]]])
        print(f"adding rectangle [[{index_x},{index_y}],[{index_x + length},{index_y + length}]] with value [{sampled_space[index_x, index_y][0]},{sampled_space[index_x + length, index_y + length][0]}]")

        ## OLD seen marking (setting seen for all searched points)
        # place(sampled_space[index_x:(index_x + length), index_y:(index_y + length)],
        #      sampled_space[index_x:(index_x + length), index_y:(index_y + length)] == False, 2)
        # place(sampled_space[index_x:(index_x + length), index_y:(index_y + length)],
        #      sampled_space[index_x:(index_x + length), index_y:(index_y + length)] == True, 2)

        print("new sampled_space: \n", sampled_space)
        ## globals()["que"].enqueue([[index_x, index_x+length-2],[index_y, index_y+length-2]],start_value)
        return result
    else:
        print(f"Sorry, {dimensions} dimensions TBD")
