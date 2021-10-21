import cv2
import numpy as np

from jigsolve.utils import dist

def get_aruco(img):
    '''Find ArUco markers in an image.

    This function takes an image and detects ArUco markers in it. The ArUco
    dictionary used is `DICT_4X4_50`.

    Parameters
    ----------
    img : np.ndarray
        A BGR image with ArUco markers.

    Returns
    -------
    corners : np.ndarray
        Detected corners.
    '''
    aruco = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    param = cv2.aruco.DetectorParameters_create()
    return cv2.aruco.detectMarkers(img, aruco, parameters=param)[0]

def rect_from_corners(corners):
    '''Create a rectangle using ArUco corners.

    This function takes corners outputted from ArUco detection and orders them
    starting from the upper-left corner and in a clockwise fashion. The
    vertices are the midpoints of each ArUco marker. The returned points will
    not be orthogonal; rather, they represent the pre-image of a rectangle
    before a perspective transformation.

    Parameters
    ----------
    corners : np.ndarray
        Detected ArUco corners.

    Returns
    -------
    rect : np.ndarray
        An array of four corners.
    '''
    # find the center of each corner
    pts = np.array([np.mean(corner[0], axis=0) for corner in corners])
    rect = np.zeros((4, 2), dtype='float32')

    # build rectangle
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perspective_transform(img, rect, border=[0, 0, 0, 0]):
    '''Perform a perspective transformation.

    This function takes an image and an array of four corner points, then
    performs a perspective transform. The corner points should start with the
    upper-left corner and continue in a clockwise fashion. The image is then
    transformed so that the image of the given set of points is mapped to a
    proper rectangle with orthogonal sides.

    Each element in the `border` list offsets and edge by that amount. The
    first edge is the top edge, continuing in a clockwise fashion.

    Parameters
    ----------
    img : np.ndarray
        A BGR image.
    rect : np.ndarray
        An array of four corner points to map to a rectangle.
    border : list
        Border offsets.
    '''
    tl, tr, br, bl = rect
    # find maximum dimensions
    w = max(int(dist(tl, tr)), int(dist(bl, br))) + border[1] + border[3]
    h = max(int(dist(tl, bl)), int(dist(tr, br))) + border[0] + border[2]
    # create a destination rectangle
    dst = np.array([
        [border[3], border[0]],
        [w - 1 - border[1], border[0]],
        [w - 1 - border[1], h - 1 - border[2]],
        [border[3], h - 1 - border[2]]
    ], dtype='float32')
    # transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # do transformation
    return cv2.warpPerspective(img, M, (w, h))

def binarize(img, threshold=70):
    '''Binarize an image.

    This function takes an image and binarizes it.

    Parameters
    ----------
    img : np.ndarray
        A BGR image.
    threshold : int
        The threshold, from 0 to 255.

    Returns
    -------
    binarized : np.ndarray
        A binarized version of the input image.
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(img_gray, threshold, 255, cv2.THRESH_OTSU)[1]

def find_contours(binarized, min_area=20000):
    '''Find contours in a binarized image.

    This function takes a binarized image and finds contours in it. Any
    contours found that do not have an area greater than `min_area` are
    removed.

    Parameters
    ----------
    binarized : np.ndarray
        A binarized image.
    min_area : int
        The minimum area for contours.

    Returns
    -------
    contours : list
        A list of contours.
    '''
    contours = cv2.findContours(image=binarized, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    return list(filter(lambda c: cv2.contourArea(c) > min_area, contours))

def get_pieces(img, contours, padding=10):
    '''Split an image into segments with contours.

    This function takes an image and a list of contours, and yields tuples
    that represent bounding boxes around each piece.

    Parameters
    ----------
    img : np.ndarray
        A BGR image containing multiple pieces.
    contours : list
        A list of contours.
    padding : int
        Additional space to add around each piece.

    Yields
    -------
    x : int
        x coordinate of upper-left corner.
    y : int
        y coordinate of upper-left corner.
    w : int
        width of bounding box.
    h : int
        height of bounding box.
    '''
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        yield x - padding, y - padding, w + 2 * padding, h + 2 * padding

def get_mask(img, contours):
    '''Get a mask from contours.

    This function takes an image and a list of contours and returns a mask.
    This mask is suitable for removing the entire background or quickly
    producing a binarized image of any part of the image.

    The mask is the same shape as the image, except the last dimension is
    reduced to 1. Each point contained in a contour is set to 255, and the
    rest of the mask is 0.

    Parameters
    ----------
    img : np.ndarray
        A BGR image.
    contours : list
        A list of contours.

    Returns
    -------
    mask : np.ndarray
        A mask including only the regions inside the contours.
    '''
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    return cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

def orientation(binarized):
    '''Determine the orientation of a piece.

    This function takes a binarized image of a piece, and returns the angle
    it is rotated, in a counterclockwise direction.

    Parameters
    ----------
    binarized : np.ndarray
        A binarized image containing a single puzzle piece.

    Returns
    -------
    angle : float
        The angle the piece is rotated, counterclockwise in degrees.
    '''
    edges = cv2.Canny(binarized, 10, 50)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 40)
    thetas = lines[:,0,1]
    thetas.sort()
    split = np.argmax(np.diff(thetas)) + 1
    theta = np.mean(thetas[:split])
    return 90 - theta * 180 / np.pi
