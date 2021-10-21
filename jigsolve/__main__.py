from pathlib import Path

import cv2
import numpy as np

from jigsolve.vision.image import binarize, find_contours, get_aruco, get_mask, get_pieces, orientation, perspective_transform, rect_from_corners
from jigsolve.utils import crop, rotate

def main():
    test_image = Path(__file__) / '../../test_images/good.jpg'
    img = cv2.imread(str(test_image.resolve()))

    # perspective transform
    corners = get_aruco(img)
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # mild cropping (not necessary in final product)
    img = img[:-150,100:-100,:]

    # find piece contours
    img_bw = binarize(img)
    contours = find_contours(img_bw)
    mask = get_mask(img, contours)
    img = cv2.bitwise_and(img, img, mask=mask)
    for box in get_pieces(img, contours):
        piece = crop(img, box)
        piece_bw = crop(mask, box)
        angle = orientation(piece_bw)
        piece = rotate(piece, -angle)
        cv2.imshow('piece', piece)
        cv2.waitKey(0)

if __name__ == '__main__': main()
