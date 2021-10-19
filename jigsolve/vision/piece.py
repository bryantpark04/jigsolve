# edge types
EDGE_INDENT = -1
EDGE_TAB = 1
EDGE_FLAT = 0

def features(img):
    '''Extract features from a single piece.

    This function takes an image of a piece, and returns an object containing
    important features of the piece.

    Parameters
    ----------
    img : np.array
        An image containing a single puzzle piece, in BGR.

    Returns
    -------
    features : Features
        An object containing the piece's features.
    '''
    pass

def orientation(binarized):
    '''Determine the orientation of a piece.

    This function takes a binarized image of a piece, and returns the angle
    it is rotated, in a counterclockwise direction.

    Parameters
    ----------
    binarized : np.array
        A binarized image containing a single puzzle piece.

    Returns
    -------
    angle : float
        The angle the piece is rotated, counterclockwise in degrees.
    '''
    pass

def edge_types(contours):
    '''Determine edge types of a piece.

    This function takes a contour of an oriented piece and determines the edge
    types. Edges start at the top edge and move in a clockwise direction.
    Edges are represented as a list of numbers, where each is either
    `EDGE_INDENT`, `EDGE_TAB`, or `EDGE_FLAT`.

    Parameters
    ----------
    contours : np.array
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
    img : np.array
        A BGR image containing a single puzzle piece.
    contour : np.array
        The contour of a puzzle piece found by `cv2.findContours`.

    Returns
    -------
    hist : np.array
        A histogram of the colors inside the piece.
    '''
    pass
