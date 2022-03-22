import cv2
import numpy as np

from jigsolve.utils import crop, dist

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

def perspective_transform(img, rect, aspect=4/3):
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
    '''
    tl, tr, br, bl = rect
    # find maximum dimensions
    w = max(dist(tl, tr), dist(bl, br))
    h = max(dist(tl, bl), dist(tr, br))
    if w / h < aspect:
        h = w / aspect
    else:
        w = h * aspect
    w = int(w)
    h = int(h)
    # h = int(0.75 * w) # may not work when there's a border
    # create a destination rectangle
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
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
    return cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)[1]

def find_contours(binarized, min_area=0):
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
    contours = cv2.findContours(image=binarized, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[0]
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
    box : tuple
        Piece bounding box.
    piece : np.ndarray
        Image of piece.
    mask : np.ndarray
        Mask for piece.
    '''
    for c in contours:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(image=mask, contours=[c], contourIdx=0, color=255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(c)
        box = (x - padding, y - padding, w + 2 * padding, h + 2 * padding)
        isolated = np.zeros_like(img)
        cv2.bitwise_and(img, img, dst=isolated, mask=mask)
        yield box, crop(isolated, box), crop(mask, box)

def orientation(binarized, bin_width=0.2):
    '''Determine the orientation of a piece.

    This function takes a binarized image of a piece, and returns the angle
    it is rotated, in a counterclockwise direction. Lines are collected into
    bins ranging from 0 to pi radians, and the bin with the most lines is
    averaged to find the orientation.

    Parameters
    ----------
    binarized : np.ndarray
        A binarized image containing a single puzzle piece.
    bin_width : float
        Number of radians for each bin.

    Returns
    -------
    angle : float
        The angle the piece is rotated, counterclockwise in degrees.
    '''
    edges = cv2.Canny(binarized, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 30)
    thetas = lines[:,0,1]
    # print(len(thetas))
    # split = np.argmax(np.diff(thetas)) + 1
    # theta = np.mean(thetas[:split])

    # freq, bins = np.histogram(thetas, bins=num_bins, range=(0, np.pi))
    # most = np.argmax(freq)
    # big_bin = np.logical_and(thetas >= bins[most], thetas <= bins[most+1])
    # theta = np.mean(thetas[big_bin])

    bins = []
    max_bins = 4

    def pick_bin(theta):
        for i, b in enumerate(bins):
            if np.abs(theta - np.average(b)) < bin_width: return i
        if len(bins) < max_bins:
            bins.append([])
            return len(bins) - 1
        return -1

    for theta in thetas:
        i = pick_bin(theta)
        if i == -1: continue
        bins[i].append(theta)

    theta = np.average(max(bins, key=len))
    return (360 - theta * 180 / np.pi) % 90
