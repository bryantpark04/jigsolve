from pathlib import Path

import cv2
import numpy as np
from imutils import rotate_bound, resize

import requests
import time
from jigsolve.models import PuzzlePiece
from jigsolve.solver.approx import eval_solution, solve_puzzle
from jigsolve.solver.fit import piece_displacements
from jigsolve.utils import grid_iter, rotate_piece

from jigsolve.vision.image import binarize, find_contours, get_aruco, get_pieces, orientation, perspective_transform, \
    rect_from_corners
from jigsolve.vision.piece import color_distribution, edge_types


def main():
    ###
    requests.get("http://192.168.69.1/cmd_pipe.php", params={"cmd": "im"})
    time.sleep(2)
    url = "http://192.168.69.1/media/image.jpg"
    data = requests.get(url).content
    open('source.jpg', 'wb').write(data)
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    ###

    wd = Path(__file__).resolve().parent
    cal = np.load(wd / 'calibration/calibration.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # perspective transform
    corners = get_aruco(img)

    print(len(corners))
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    cv2.imwrite('perspective.png', img)

    # mild cropping (not necessary in final product)
    img = img[120:-140]

    cv2.imwrite('test.png', img)

    # find piece contours
    img_bw = binarize(img, threshold=45)
    cv2.imwrite('bin.png', img_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours = find_contours(img_bw, min_area=18000)
    print(len(contours))

    pieces = []
    for box, piece, mask in get_pieces(img, contours, padding=0):
        angle = orientation(mask)
        if angle > 45: angle -= 90
        piece = rotate_bound(piece, angle)
        mask = rotate_bound(mask, angle)
        edges = edge_types(mask)
        hist = tuple(color_distribution(piece, mask))
        pieces.append(PuzzlePiece(piece, mask, hist, angle, box, edges))

    solutions = solve_puzzle(pieces)
    # solutions = list(filter(lambda s: s[0, 0] == (9, 3), solutions))
    print(len(solutions))
    scores = [eval_solution(pieces, solution) for solution in solutions]

    solution = solutions[np.argsort(scores)[0]]
    # test piece alignment
    disp = piece_displacements(pieces, solution)

    h, w = solution.shape
    canvas = np.zeros((h * 500, w * 500, 3), np.uint8)
    for r, c in grid_iter(h, w):
        temp = np.zeros_like(canvas)
        pi, pr = solution[r, c]
        img = rotate_piece(pieces[pi].img, pr)
        xd, yd = disp[r, c]
        ih, iw, _ = img.shape
        temp[yd:yd + ih, xd:xd + iw] = img
        canvas = cv2.add(canvas, temp)

    cv2.imwrite('solution.png', canvas)
    small = resize(canvas, width=800)
    cv2.imshow('test', small)
    cv2.waitKey(0)


if __name__ == '__main__': main()
