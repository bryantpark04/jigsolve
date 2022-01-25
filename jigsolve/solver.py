import numpy as np

from jigsolve.utils import edge_rotate, grid
from jigsolve.vision.piece import EDGE_DOWN, EDGE_FLAT, EDGE_LEFT, EDGE_RIGHT, EDGE_UP

def puzzle_dimensions(pieces):
    area = len(pieces)
    perimeter = sum(piece.edges.count(0) for piece in pieces)
    c = int((perimeter + np.sqrt(perimeter**2 - 16 * area)) / 4)
    return c, area // c

def preprocess_edges(pieces):
    '''Generates edge rotation and prefix dictionary.

    Parameters
    ----------
    pieces : list[PuzzlePiece]
        List of puzzle pieces.

    Returns
    -------
    lookup : dict[tuple[int], set[tuple[int]]]
        deet
    all : set[tuple[int]]
        all the pieces
    '''
    lookup = {}
    all = set()
    for i, piece in enumerate(pieces):
        for r in range(4):
            edges = edge_rotate(piece.edges, r)
            all.add((i, r))
            for key in enumerate(edges):
                if key not in lookup:
                    lookup[key] = set()
                lookup[key].add((i, r))

    return lookup, all

def solve_puzzle(pieces):
    '''Find a solution to a puzzle.

    This function takes puzzle pieces and fits them into a rectangular grid.

    Parameters
    ----------
    pieces : list[PuzzlePiece]
        List of puzzle pieces.

    Returns
    -------
    solution : something lmao
        idk bruh
    '''
    lookup, all = preprocess_edges(pieces)
    w, h = puzzle_dimensions(pieces)
    solution = {}
    possible = {}
    for r, c in grid(h, w):
        possible[r, c] = all.copy()
    for c in range(w):
        possible[0, c] &= lookup[EDGE_UP, EDGE_FLAT]
        possible[h-1, c] &= lookup[EDGE_DOWN, EDGE_FLAT]
    for r in range(h):
        possible[r, 0] &= lookup[EDGE_LEFT, EDGE_FLAT]
        possible[r, w-1] &= lookup[EDGE_RIGHT, EDGE_FLAT]
    used = set()
    # solve puzzle
    return solution
