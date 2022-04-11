from pathlib import Path
from functools import partial

import cv2
import numpy as np
from imutils import rotate_bound, resize

import requests
import time
from jigsolve.models import PuzzlePiece
from jigsolve.solver.approx import eval_solution, solve_puzzle
from jigsolve.solver.fit import piece_displacements
from jigsolve.utils import grid_iter, rotate_piece, split_combined

from jigsolve.vision.image import binarize, find_contours, get_aruco, get_pieces, orientation, perspective_transform, \
    rect_from_corners
from jigsolve.vision.piece import color_distribution, edge_types, get_origin

def main():
    ###
    # requests.get("http://192.168.69.1/cmd_pipe.php", params={"cmd": "im"})
    # time.sleep(2)
    # url = "http://192.168.69.1/media/image.jpg"
    # data = requests.get(url).content
    # open('source.jpg', 'wb').write(data)
    # arr = np.asarray(bytearray(data), dtype=np.uint8)
    # img = cv2.imdecode(arr, -1)
    ###

    # Test using image source.jpg
    img = cv2.imread("source.jpg")

    wd = Path(__file__).resolve().parent
    cal = np.load(wd / 'calibration/calibration.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # perspective transform
    corners = get_aruco(img)

    print(len(corners))
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # find piece contours
    img_bw = binarize(img, threshold=25)
    contours = find_contours(img_bw, min_area=20000, max_area=50000)
    print(len(contours))

    pieces = []
    for box, combined in get_pieces(img, contours, padding=0):
        # find origin of piece
        _, mask, _ = split_combined(combined)
        origin = get_origin(mask)
        # store origin in separate layer
        origin_mat = np.zeros_like(mask)
        origin_mat[origin[1], origin[0]] = 255
        origin = (origin[0] + box[0], origin[1] + box[1])
        combined = np.dstack((combined, origin_mat))
        # find piece rotation
        angle = orientation(mask, bin_width=0.25)
        if angle > 45: angle -= 90
        combined = rotate_bound(combined, angle)
        # identify edge types
        _, mask, _ = split_combined(combined)
        edges = edge_types(mask)
        # calculate edge color distributions
        hist = tuple(color_distribution(combined))
        # construct piece object
        pieces.append(PuzzlePiece(combined, hist, origin, angle, box, edges))

    # get approximate solution
    solutions = solve_puzzle(pieces)
    print(len(solutions))
    solution = min(solutions, key=partial(eval_solution, pieces))

    # find piece displacements
    solution_origin = (300, 1400)
    disp = piece_displacements(pieces, solution, solution_origin)

    # add pieces to image
    for r, c in grid_iter(*solution.shape):
        temp = np.zeros_like(img)
        pi, pr = solution[r, c]
        pimg, _, porigin = split_combined(rotate_piece(pieces[pi].combined, pr))
        xd, yd = disp[r, c]
        ih, iw, _ = pimg.shape
        temp[yd:yd + ih, xd:xd + iw] = pimg
        img = cv2.add(img, temp)

        src_point = pieces[pi].origin
        dst_point = cv2.minMaxLoc(porigin)[3]
        dst_point = (dst_point[0] + xd, dst_point[1] + yd)

        cv2.line(img, src_point, dst_point, (0, 255, 0), 10, cv2.LINE_AA)
        cv2.circle(img, src_point, 20, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, dst_point, 20, (255, 0, 0), cv2.FILLED)

    cv2.imwrite('solution.png', img)

if __name__ == '__main__': main()
