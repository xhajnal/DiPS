import itertools
from copy import copy
from termcolor import colored

from common.space_stuff import refine_by, get_intersection, is_in
from space import RefinedSpace


def estimate_dimensions_width_2D(refinement, params, param_intervals):
    """ Counts number of horizontal and vertical pieces of 2D visualised Storm refinement

    Args:
        refinement (list of strings): array of rectangles represented by a symbol "S" (safe), "-" unsafe(), and " " (unknown)
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter
    Return:
        tuples (number of horizontal pieces, number of vertical pieces)
    """
    # Get number of pieces in each dimension
    vertical_pieces_count = len(refinement)
    horizontal_pieces_count = len(refinement[0])

    # Get length of individual piece in each dimension
    vertical_piece_length = (param_intervals[1][1] - param_intervals[1][0]) / vertical_pieces_count
    horizontal_piece_length = (param_intervals[0][1] - param_intervals[0][0]) / horizontal_pieces_count

    ## Compute vertical intervals
    spam = param_intervals[1][0]
    vertical_pieces = []
    for item in range(vertical_pieces_count):
        vertical_pieces.append([spam, spam + vertical_piece_length])
        spam = spam + vertical_piece_length

    ## Compute horizontal intervals
    spam = param_intervals[0][0]
    horizontal_pieces = []
    for item in range(horizontal_pieces_count):
        horizontal_pieces.append([spam, spam + horizontal_piece_length])
        spam = spam + horizontal_piece_length

    return horizontal_pieces, vertical_pieces


def parse_2D_refinement_into_space(refinement, params, param_intervals):
    """ Converts 2D visualised Storm refinement into true rectangles

    Args:
        refinement (list of strings): array of rectangles represented by a symbol "S" (safe), "-" unsafe(), and " " (unknown)
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter

    Returns:
        triple [safe, unsafe, unknown rectangles]
    """
    horizontal_pieces, vertical_pieces = estimate_dimensions_width_2D(refinement, params, param_intervals)

    safe_rectangles = []
    unsafe_rectangles = []
    unknown_rectangles = []

    for h_index, h_item in enumerate(horizontal_pieces):
        for v_index, v_item in enumerate(vertical_pieces):
            value = refinement[v_index][h_index]
            if value == "S":
                safe_rectangles.append([h_item, v_item])
            elif value == " ":
                unsafe_rectangles.append([h_item, v_item])
            elif value == "-":
                unknown_rectangles.append([h_item, v_item])
            else:
                raise Exception(f"Unexpected symbol found in the refinement by Storm at {v_index+1}:{h_index}.")

    return [safe_rectangles, unsafe_rectangles, unknown_rectangles]


## TODO I guess this method is not necessary
def parse_multidim_refinement_into_space(refinement, params, param_intervals):
    """ Converts multidimensional Storm refinement into true rectangles

    Args:
        refinement (list of strings): array of rectangles represented by AllSat, AllViolated, ExistsSat, ExistsViolated, Unknown
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter

    Returns:
        triple [safe, unsafe, unknown rectangles]
    """
    raise NotImplementedError()


def merge_2D_refinements(refinements, params, param_intervals):
    """ Merges several 2D visualised Storm refinements with conjunction
     The refinements have to share parameters, their order and domains.

    Args:
        refinements (list of list of strings): list of array of rectangles represented by a symbol "S" (safe), "-" unsafe(), and " " (unknown)
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter

    Returns:
        triple [safe, unsafe, unknown rectangles]
     """
    horizontal_pieces, vertical_pieces = estimate_dimensions_width_2D(refinements[0], params, param_intervals)

    safe_rectangles = []
    unsafe_rectangles = []
    unknown_rectangles = []

    for h_index, h_item in enumerate(horizontal_pieces):
        for v_index, v_item in enumerate(reversed(vertical_pieces)):
            ## TODO this can be optimised but it runs for fixed number of rectangles
            is_sat = True
            is_unknown = True
            for ref_index, refinement in enumerate(refinements):
                value = refinements[ref_index][v_index][h_index]
                if value == "S":
                    continue
                elif value == " ":
                    unsafe_rectangles.append([h_item, v_item])
                    is_sat = False
                    is_unknown = False
                    break
                elif value == "-":
                    is_sat = False
                else:
                    raise Exception(f"Unexpected symbol, {value}, found in the refinement by Storm at {v_index+1}:{h_index} of refinement number {ref_index + 1}.")
            if is_sat:
                safe_rectangles.append([h_item, v_item])
            elif is_unknown:
                unknown_rectangles.append([h_item, v_item])

    return [safe_rectangles, unsafe_rectangles, unknown_rectangles]


def merge_multidim_refinements(refinements, params, param_intervals):
    """ Merges several multidimensional refinements with conjunction
     The refinements have to share parameters, their order and domains.

    Args:
        refinements (list of list of intervals): list of [[sat], [unsat], [unknown]] rectangles
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter

    Returns:
        triple [safe, unsafe, unknown rectangles]
     """

    ## Sort refinements
    for refinement_index, refinement in enumerate(refinements):
        for type_index, type in enumerate(refinement):
            refinements[refinement_index][type_index] = sorted(refinements[refinement_index][type_index])

    ## Initial unsat subspace, we will add each new rectangle to it
    unsafe = refinements[0][1]

    ## UNSAFE
    to_add = False
    for index_space, space in enumerate(refinements[1:]):  ## skip the first refinement
        for index_rectangle, unsat_rectangle in enumerate(space[1]):
            # if unsat_rectangle in unsat:  ## if the unsat rectangles is the list of unsat rectangles skip it
            #     continue
            # print(colored(unsat_rectangle, "yellow"))
            for rect_already_in in unsafe:
                if unsat_rectangle[0][1] < rect_already_in[0][0]:
                    # print(colored("hello break", "yellow"))
                    # print(unsat_rectangle)
                    # print(rect_already_in)
                    to_add = True
                    break
                if is_in(unsat_rectangle, rect_already_in):
                    to_add = False
                    # print("break 2")
                    break

                intersection = get_intersection(unsat_rectangle, rect_already_in)
                if intersection is False:
                    # print("continue")
                    to_add = True
                    continue
                # print(colored(f"intersection of {unsat_rectangle} and {rect} chosen as {intersection}", "yellow"))
                egg = refine_by(unsat_rectangle, intersection)
                egg.remove(intersection)
                # egg.extend(refine_by(rect_already_in, intersection))
                # print(egg)
                # print(refinements[index_space + 1][1])
                refinements[index_space + 1][1].extend(egg)
                # print(refinements[index_space + 1][1])
                to_add = False
                # print("break 3")
                break
            if to_add:
                # print(f"adding {unsat_rectangle}")
                unsafe.append(unsat_rectangle)
                unsafe = sorted(unsafe)
                to_add = False

    ## Remove duplicates
    unsafe = sorted(unsafe)
    for i in range(len(unsafe)):
        dedup = list(item for item, _ in itertools.groupby(unsafe))
    try:
        unsafe = dedup
    except UnboundLocalError:
        unsafe = []

    ## UNKNOWN
    non_green = copy(unsafe)
    unknwon = []
    to_add = False
    for index_space, space in enumerate(refinements):
        for index_rectangle, unknown_rectangle in enumerate(space[2]):
            for index, rect_already_in in enumerate(non_green):
                if unknown_rectangle[0][1] < rect_already_in[0][0]:
                    to_add = True
                    break
                intersection = get_intersection(unknown_rectangle, rect_already_in)
                if intersection is False:
                    to_add = True
                    continue
                else:
                    spam = refine_by(unknown_rectangle, intersection)
                    spam.remove(intersection)
                    refinements[index_space][2].extend(spam)
                    to_add = False
                    break
            if to_add:
                non_green.append(unknown_rectangle)
                non_green = sorted(non_green)
                unknwon.append(unknown_rectangle)
                to_add = False

    ## Remove duplicates
    for i in range(len(unknwon)):
        dedup1 = list(item for item, _ in itertools.groupby(unknwon))
    try:
        unknwon = dedup1
    except UnboundLocalError:
        unknwon = []

    ## SAFE
    safe = []
    all = copy(non_green)
    to_add = False
    for index_space, space in enumerate(refinements):
        for index_rectangle, safe_rectangle in enumerate(space[0]):  ## iterate through safe rectangles
            for index, rect_already_in in enumerate(all):
                if safe_rectangle[0][1] < rect_already_in[0][0]:
                    to_add = True
                    break
                intersection = get_intersection(safe_rectangle, rect_already_in)
                if intersection is False:
                    to_add = True
                    continue
                else:
                    spam = refine_by(safe_rectangle, intersection)
                    spam.remove(intersection)
                    refinements[index_space][0].extend(spam)
                    to_add = False
                    break
            if to_add:
                all.append(safe_rectangle)
                all = sorted(all)
                safe.append(safe_rectangle)
                to_add = False

    ## Remove duplicates
    for i in range(len(safe)):
        dedup2 = list(item for item, _ in itertools.groupby(safe))
    try:
        safe = dedup2
    except UnboundLocalError:
        safe = []

    ## TEST
    # kkk = copy(safe)
    # kkk.extend(unknwon)
    # space = RefinedSpace([[0, 1], [0, 1], [0, 1]], ["a", "b", "c"], rectangles_sat=kkk, rectangles_unknown=[], rectangles_unsat=unsafe)
    # print(space.get_coverage())
    #
    # space = RefinedSpace([[0, 1], [0, 1], [0, 1]], ["a", "b", "c"], rectangles_sat=safe, rectangles_unknown=unknwon, rectangles_unsat=unsafe)
    # print(space.get_green_volume())
    # print(space.get_red_volume())
    # print(space.get_white_volume())
    #
    # for unknown_rectangle in unknwon:
    #     for unsafe_rectangle in unsafe:
    #         intersection = get_intersection(unknown_rectangle, unsafe_rectangle)
    #         if intersection is not False:
    #             print(colored(f"1 {unknown_rectangle} /// {unsafe_rectangle}", "red"))
    # print()
    # for safe_rectangle in safe:
    #     for unsafe_rectangle in unsafe:
    #         intersection = get_intersection(safe_rectangle, unsafe_rectangle)
    #         if intersection is not False:
    #             print(colored(f"2 {safe_rectangle} /// {unsafe_rectangle}", "red"))
    # print()
    # for safe_rectangle in safe:
    #     for unknown_rectangle in unknwon:
    #         intersection = get_intersection(safe_rectangle, unknown_rectangle)
    #         if intersection is not False:
    #             print(colored(f"3 {safe_rectangle} /// {unknown_rectangle}", "red"))
    #
    # print()
    # for safe_rectangle in safe:
    #     for unknown_rectangle in non_green:
    #         intersection = get_intersection(safe_rectangle, unknown_rectangle)
    #         if intersection is not False:
    #             print(colored(f"4 {safe_rectangle} /// {unknown_rectangle}", "red"))
    #
    # print()
    # for index, rectangle in enumerate(safe):
    #     for another_rectangle in safe[index+1:]:
    #         intersection = get_intersection(rectangle, another_rectangle)
    #         if intersection is not False:
    #             print(colored(f"5 {rectangle} /// {another_rectangle}", "red"))
    #
    # print()
    # for index, rectangle in enumerate(unsafe):
    #     for another_rectangle in unsafe[index + 1:]:
    #         intersection = get_intersection(rectangle, another_rectangle)
    #         if intersection is not False:
    #             print(colored(f"6 {rectangle} /// {another_rectangle}", "red"))
    #
    # print()
    # for index, rectangle in enumerate(unknwon):
    #     for another_rectangle in unknwon[index + 1:]:
    #         intersection = get_intersection(rectangle, another_rectangle)
    #         if intersection is not False:
    #             print(colored(f"7 {rectangle} /// {another_rectangle}", "red"))

    return [safe, unsafe, unknwon]
