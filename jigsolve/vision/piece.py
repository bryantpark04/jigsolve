import cv2
import numpy as np

# edge types
EDGE_INDENT = -1
EDGE_TAB = 1
EDGE_FLAT = 0

def edge_types(contours):
    '''Determine edge types of a piece.

    This function takes a contour of an oriented piece and determines the edge
    types. Edges start at the top edge and move in a clockwise direction.
    Edges are represented as a list of numbers, where each is either
    `EDGE_INDENT`, `EDGE_TAB`, or `EDGE_FLAT`.

    Parameters
    ----------
    contours : np.ndarray
        The contour of an oriented puzzle piece found by `cv2.findContours`.

    Returns
    -------
    edges : list
        A list of edges, clockwise starting from the top edge.
    '''
    pass

def color_distribution(img, contour):
    '''Determine color distribution of a piece.

    This function takes an image of a piece and returns a histogram of the
    colors in the piece.

    Parameters
    ----------
    img : np.ndarray
        A BGR image containing a single puzzle piece.
    contour : np.ndarray
        The contour of a puzzle piece found by `cv2.findContours`.

    Returns
    -------
    hist : np.ndarray
        A histogram of the colors inside the piece.
    '''
    pass
