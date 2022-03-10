from pathlib import Path

import cv2
import numpy as np
from imutils import rotate_bound, resize

import requests
import time

from jigsolve.vision.image import binarize, find_contours, get_aruco, get_pieces, orientation, perspective_transform, \
    rect_from_corners
from jigsolve.vision.piece import edge_types


def main():
    ###
    requests.get("http://192.168.69.1/cmd_pipe.php", params={"cmd": "im"})
    time.sleep(1)
    url = "http://192.168.69.1/media/image.jpg"
    data = requests.get(url).content
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    ###

    wd = Path(__file__).resolve().parent
    cal = np.load(wd / 'calibration/calibration.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # cv2.imwrite('undistort.png', img)

    cv2.imshow("cam", resize(img, width=800))

    cv2.waitKey(0)

    # perspective transform
    corners = get_aruco(img)
    print(len(corners))
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # mild cropping (not necessary in final product)
    img = img[70:-67]
    # cv2.imwrite('transform.png', img)

    # find piece contours
    img_bw = binarize(img, threshold=10)
    # cv2.imwrite('binarize.png', img_bw)
    contours = find_contours(img_bw, min_area=18000)

    # img_copy = img.copy()
    # cv2.drawContours(img_copy, contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)
    # cv2.imwrite('contours.png', img_copy)

    # pieces = list(get_pieces(img, contours))
    # box, piece, mask = pieces[9]
    for box, piece, mask in get_pieces(img, contours, padding=20):
        angle = orientation(mask, num_bins=16)
        if angle > 45: angle -= 90
        piece = rotate_bound(piece, angle)
        mask = rotate_bound(mask, angle)
        edges = edge_types(mask)
        # cv2.imwrite('piece.png', piece)
        # cv2.imshow('piece', piece)
        # cv2.waitKey(0)


if __name__ == '__main__': main()
