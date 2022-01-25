from pathlib import Path

import cv2
import numpy as np
from imutils import rotate_bound

from jigsolve.models import PuzzlePiece
from jigsolve.solver import solve_puzzle
from jigsolve.vision.image import binarize, find_contours, get_aruco, get_pieces, orientation, perspective_transform, rect_from_corners
from jigsolve.vision.piece import edge_types

def main():
    wd = Path(__file__).resolve().parent
    test_image = wd / 'test_images/turck2.jpg'
    img = cv2.imread(str(test_image.resolve()))

    cal = np.load(wd / 'calibration/calibration.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # perspective transform
    corners = get_aruco(img)
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # mild cropping (not necessary in final product)
    img = img[70:-67]

    # find piece contours
    img_bw = binarize(img, threshold=10)
    contours = find_contours(img_bw, min_area=18000)

    pieces = []
    for box, piece, mask in get_pieces(img, contours, padding=20):
        angle = orientation(mask, num_bins=16)
        if angle > 45: angle -= 90
        piece = rotate_bound(piece, angle)
        mask = rotate_bound(mask, angle)
        edges = edge_types(mask)
        pieces.append(PuzzlePiece(piece, mask, angle, box, edges, ()))

    solution = solve_puzzle(pieces)
    print(solution)

if __name__ == '__main__': main()
