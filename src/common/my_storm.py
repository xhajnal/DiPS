def cut_space(refinement, params, param_intervals):
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


def parse_refinement_into_space(refinement, params, param_intervals):
    """ Converts 2D visualised Storm refinement into true rectangles

    Args:
        refinement (list of strings): array of rectangles represented by a symbol "S" (safe), "-" unsafe(), and " " (unknown)
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter

    Returns:
        triple [safe, unsafe, unknown rectangles]
    """
    horizontal_pieces, vertical_pieces = cut_space(refinement, params, param_intervals)

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


def merge_refinements(refinements, params, param_intervals):
    """ Merges several 2D visualised Storm refinements with conjunction
     The refinements have to share parameters, their order and domains.

    Args:
        refinements (list of list of strings): list of array of rectangles represented by a symbol "S" (safe), "-" unsafe(), and " " (unknown)
        params (list of strings): list of parameter names
        param_intervals (list of tuples): domains of each parameter

    Returns:
        triple [safe, unsafe, unknown rectangles]
     """
    horizontal_pieces, vertical_pieces = cut_space(refinements[0], params, param_intervals)

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
