
def solve_puzzle(edges):
    '''Find a solution to a puzzle.

    This function takes puzzle pieces and fits them into a rectangular grid.

    '''

    # list of pieces, which are to be represented by their indices in this list
    pieces = [] # not defined yet, need to figure out how to do so

    # build the dictionary to store edge patterns
    # format: {(0, 1, -1) : {(piece_1_idx, rotation), (piece_2_idx, rotation)}, ...}
    edge_map = {}
    for edge in edges:
        pass