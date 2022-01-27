import cv2
import functools
import numpy as np

from jigsolve.utils import adjacent, edge_rotate, grid_iter
from jigsolve.vision.piece import EDGE_DIRECTIONS, EDGE_DOWN, EDGE_FLAT, EDGE_LEFT, EDGE_RIGHT, EDGE_UP

def puzzle_dimensions(pieces):
    area = len(pieces)
    perimeter = sum(piece.edges.count(0) for piece in pieces)
    c = int((perimeter + np.sqrt(perimeter**2 - 16 * area)) / 4)
    return area // c, c

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
    h, w = puzzle_dimensions(pieces)
    solutions = []
    grid = {}
    possible = {}
    for r, c in grid_iter(h, w):
        grid[r, c] = None
        possible[r, c] = all.copy()
    for c in range(w):
        possible[0, c] &= lookup[EDGE_UP, EDGE_FLAT]
        possible[h-1, c] &= lookup[EDGE_DOWN, EDGE_FLAT]
    for r in range(h):
        possible[r, 0] &= lookup[EDGE_LEFT, EDGE_FLAT]
        possible[r, w-1] &= lookup[EDGE_RIGHT, EDGE_FLAT]
    used = set()

    def get_least():
        best = w * h + 1
        bp = None
        for r, c in grid_iter(h, w):
            if grid[r, c] is not None: continue
            inspect = possible[r, c]
            if len(inspect) < 2:
                return (r, c)
            if len(inspect) < best:
                best = len(inspect)
                bp = (r, c)
        return bp

    def solve_puzzle_util():
        if len(used) == w * h:
            solutions.append(grid.copy())
            return
        r, c = get_least()
        # loop through possible pieces
        for pi, pr in possible[r, c]:
            if pi in used: continue
            used.add(pi)
            grid[r, c] = (pi, pr)
            old = {}
            edges = edge_rotate(pieces[pi].edges, pr)
            # loop through adjacent edges
            # no need for bounds checking because edges have been taken care of already
            for i, e in enumerate(edges):
                if e == EDGE_FLAT: continue
                # the opposite edge
                opp = (i + 2) % 4
                ak = (opp, -e)
                # make sure it's possible
                if ak not in lookup: break
                # save and set adjacent cell
                dr, dc = EDGE_DIRECTIONS[i]
                key = r + dr, c + dc
                old[key] = possible[key]
                possible[key] = possible[key] & lookup[ak]
            else:
                # recur
                solve_puzzle_util()
            # restore
            used.remove(pi)
            grid[r, c] = None
            for k, v in old.items():
                possible[k] = v
        return

    solve_puzzle_util()
    return (h, w), solutions

def eval_solution(h, w, pieces, solution):
    @functools.lru_cache(maxsize=None)
    def hist_compare(pe1, pe2):
        p1, e1 = pe1
        p2, e2 = pe2
        return cv2.compareHist(pieces[p1].hist[e1], pieces[p2].hist[e2], cv2.HISTCMP_BHATTACHARYYA)
    score = 0
    for r, c in grid_iter(h, w):
        p1, r1 = solution[r, c]
        e1 = edge_rotate(pieces[p1].edges, r1)
        for i, e in enumerate(e1):
            if e == EDGE_FLAT: continue
            # save and set adjacent cell
            dr, dc = EDGE_DIRECTIONS[i]
            ar, ac = r + dr, c + dc
            p2, r2 = solution[ar, ac]
            pe1 = (p1, (i + r1) % 4)
            pe2 = (p2, (i + r2 + 2) % 4)
            score += hist_compare(*sorted([pe1, pe2]))
    return score
