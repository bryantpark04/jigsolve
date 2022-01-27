import cv2
import numpy as np

from jigsolve.utils import dist

# edge types
EDGE_INDENT = -1
EDGE_TAB = 1
EDGE_FLAT = 0

# edge locations
EDGE_UP = 0
EDGE_RIGHT = 1
EDGE_DOWN = 2
EDGE_LEFT = 3

# edge directions
EDGE_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def edge_types(piece, indent=30, indent_max=25, tab=2, tab_length=20, tab_width=30):
    '''Determine edge types of a piece.

    This function takes a contour of an oriented piece and determines the edge
    types. Edges start at the top edge and move in a clockwise direction.
    Edges are represented as a list of numbers, where each is either
    `EDGE_INDENT`, `EDGE_TAB`, or `EDGE_FLAT`.

    Parameters
    ----------
    piece : np.ndarray
        A binarized image of an oriented piece.

    Returns
    -------
    edges : list
        A list of edges, clockwise starting from the top edge.
    '''
    show = np.zeros_like(piece)
    show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)

    piece = cv2.medianBlur(piece, 9)
    contours = cv2.findContours(image=piece, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(show, [contour], 0, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imwrite('deet0.png', show)

    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(show, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    # cv2.imwrite('deet1.png', show)

    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    edges = [None]*4
    for s, e, f, d in defects[:,0]:
        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        dx = np.abs(start[0] - end[0])
        dy = np.abs(start[1] - end[1])
        ox = np.abs(far[0] - cx)
        oy = np.abs(far[1] - cy)
        cv2.line(show, start, end, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(show, far, 5, (0, 255, 255), cv2.FILLED)
        if d > 256 * indent:
            if edges[0] is None and far[1] < cy and ox < oy and dx > dy and dy < indent_max:
                edges[0] = EDGE_INDENT
            elif edges[1] is None and far[0] > cx and ox > oy and dx < dy and dx < indent_max:
                edges[1] = EDGE_INDENT
            elif edges[2] is None and far[1] > cy and ox < oy and dx > dy and dy < indent_max:
                edges[2] = EDGE_INDENT
            elif edges[3] is None and far[0] < cx and ox > oy and dx < dy and dx < indent_max:
                edges[3] = EDGE_INDENT

    for s, e, f, d in defects[:,0]:
        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        ox = np.abs(far[0] - cx)
        oy = np.abs(far[1] - cy)
        if d < 256 * tab and dist(start, end) < tab_length:
            if edges[0] is None and far[1] < cy and ox < oy and ox < tab_width:
                edges[0] = EDGE_TAB
            elif edges[1] is None and far[0] > cx and ox > oy and oy < tab_width:
                edges[1] = EDGE_TAB
            elif edges[2] is None and far[1] > cy and ox < oy and ox < tab_width:
                edges[2] = EDGE_TAB
            elif edges[3] is None and far[0] < cx and ox > oy and oy < tab_width:
                edges[3] = EDGE_TAB

    for i, e in enumerate(edges):
        if e is None: edges[i] = EDGE_FLAT

    return tuple(edges)

def color_distribution(img, mask):
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
    return cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
