import numpy as np
import cv2

def dist(p1, p2):
    '''Distance between two points

    Parameters
    ----------
    p1 : np.ndarray
        Point 1.
    p2 : np.ndarray
        Point 2.

    Returns
    -------
    d : float
        Distance between two points.
    '''
    return np.linalg.norm(p1 - p2)

def crop(img, box):
    '''Crop an image with a bounding box.

    The box is defined by a 4-tuple. The first two define the x and y
    coordinates of the upper-left corner respectively, and the last two define the
    width and height of the box respectively.

    Parameters
    ----------
    img : np.ndarray
        An image.
    box : tuple
        Coordinates of the box.

    Returns
    -------
    cropped : np.ndarray
        The cropped image.
    '''
    x, y, w, h = box
    return img[y:y+h,x:x+w]

def rotate(img, angle):
    '''Rotate an image.

    This function takes an image and rotates it counterclockwise.

    Parameters
    ----------
    img : np.ndarray
        The image to rotate.
    angle : float
        The angle, in degrees, to rotate counterclockwise.

    Returns
    -------
    rotated : np.ndarray
        The rotated image.
    '''
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_h = int(h * abs_sin + w * abs_cos)
    new_w = int(h * abs_cos + w * abs_sin)
    M[0,2] += new_w / 2 - w / 2
    M[1,2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, M, (new_h, new_w))
