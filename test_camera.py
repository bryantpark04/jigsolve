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

    # cv2.imwrite('perspective.png', img)
    # persp_img = img.copy() # this img used for internal coordinate system

    # mild cropping
    # img = img[120:-140]
    # crop_margin = 120
    # img = img[crop_margin:-crop_margin, crop_margin:-crop_margin]

    # cv2.imwrite('test.png', img)

    # find piece contours
    img_bw = binarize(img, threshold=25)
    # cv2.imwrite('bin.png', img_bw)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours = find_contours(img_bw, min_area=20000, max_area=50000)
    print(len(contours))

    pieces = []
    for box, piece, mask in get_pieces(img, contours, padding=0):
        origin = get_origin(mask)
        origin += box[:2]
        origin = tuple(origin)
        angle = orientation(mask, bin_width=0.25)
        if angle > 45: angle -= 90
        piece = rotate_bound(piece, angle)
        mask = rotate_bound(mask, angle)
        edges = edge_types(mask)
        hist = tuple(color_distribution(piece, mask))
        pieces.append(PuzzlePiece(piece, mask, hist, origin, angle, box, edges))

    solutions = solve_puzzle(pieces)
    print(len(solutions))
    scores = [eval_solution(pieces, solution) for solution in solutions]

    solution = solutions[np.argsort(scores)[0]]
    solution_origin = (300, 1400)
    disp = piece_displacements(pieces, solution, solution_origin)

    # add pieces to canvas
    h, w = solution.shape
    # canvas = np.zeros_like(img, dtype=np.uint8)
    for r, c in grid_iter(h, w):
        temp = np.zeros_like(img)
        pi, pr = solution[r, c]
        pImg = rotate_piece(pieces[pi].img, pr)
        xd, yd = disp[r, c]
        ih, iw, _ = pImg.shape
        temp[yd:yd + ih, xd:xd + iw] = pImg
        img = cv2.add(img, temp)

        src_point = pieces[pi].origin
        dst_point = get_origin(rotate_piece(pieces[pi].mask, pr)) + disp[r, c]
        cv2.line(img, src_point, dst_point, (0, 255, 0), 10, cv2.LINE_AA)
        cv2.circle(img, src_point, 20, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, dst_point, 20, (255, 0, 0), cv2.FILLED)

    cv2.imwrite('solution.png', img)

if __name__ == '__main__': main()
