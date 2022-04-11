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
from jigsolve.robot.puzzle_robot import piece_pick_point


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

    cv2.imwrite('perspective.png', img)
    persp_img = img.copy() # this img used for internal coordinate system

    # mild cropping
    # img = img[120:-140]
    crop_margin = 120
    img = img[crop_margin:-crop_margin, crop_margin:-crop_margin]

    cv2.imwrite('test.png', img)

    # find piece contours
    img_bw = binarize(img, threshold=40)
    cv2.imwrite('bin.png', img_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours = find_contours(img_bw, min_area=18000)
    print(len(contours))

    pieces = []
    init_positions_image_coords = [] # initial position of the top left corner of the bounding box for each puzzle piece, in perp_image coords
    for box, piece, mask, pos in get_pieces(img, contours, padding=0):
        angle = orientation(mask)
        if angle > 45: angle -= 90
        piece = rotate_bound(piece, angle)
        mask = rotate_bound(mask, angle)
        edges = edge_types(mask)
        hist = tuple(color_distribution(piece, mask))
        init_pos = np.array([pos[0] + crop_margin, pos[1] + crop_margin])
        pieces.append(PuzzlePiece(piece, mask, hist, angle, box, edges))
        init_positions_image_coords.append(init_pos)

    # Find optimal location for robot to pick up a piece
    # test_piece = pieces[2]
    # pick_point = piece_pick_point(test_piece.mask)
    # print(pick_point)

    solutions = solve_puzzle(pieces)
    # solutions = list(filter(lambda s: s[0, 0] == (9, 3), solutions))
    print(len(solutions))
    scores = [eval_solution(pieces, solution) for solution in solutions]

    solution = solutions[np.argsort(scores)[0]]
    disp = piece_displacements(pieces, solution)

    # add pieces to canvas
    h, w = solution.shape
    canvas = np.zeros((h * 300, w * 300, 3), np.uint8)
    for r, c in grid_iter(h, w):
        temp = np.zeros_like(canvas)
        pi, pr = solution[r, c]
        pImg = rotate_piece(pieces[pi].img, pr)
        xd, yd = disp[r, c]
        ih, iw, _ = pImg.shape
        temp[yd:yd + ih, xd:xd + iw] = pImg
        canvas = cv2.add(canvas, temp)
    
    # find a place for the solved puzzle on original image (hardcoded for now)
    img_shape = persp_img.shape
    canvas_pos = np.array([int(0.45 * img_shape[0]), 0]) # top left corner of canvas on perspective image, (r, c)
    print(img.shape, canvas.shape)
    persp_img[canvas_pos[0]:canvas_pos[0] + canvas.shape[0], canvas_pos[1]:canvas_pos[1] + canvas.shape[1]] = canvas
    cv2.imshow("test 1", resize(persp_img, width=800))
    cv2.waitKey(0)

    # find optimal point for picking up each piece, relative to piece image's bounding box
    # pick_points = np.ndarray((len(pieces), 2))
    # for piece in pieces:
    #     pick_point = piece_pick_point(piece.mask)
    
    # find final piece positions and rotations in perp_image coordinates
    # TODO: Use piece.img_rot, canvas_pos, and disp
    final_positions_image_coords = np.ndarray((len(pieces), 2)) # final piece positions in image coords
    rotations = np.ndarray(len(pieces))
    h, w = solution.shape
    for r, c in grid_iter(h, w):
        pi, pr = solution[r, c]
        final_positions_image_coords[pi] = canvas_pos + disp[r, c]
        rotations[pi] = pieces[pi].img_rot

    # transform (initial and final) piece positions and rotations into real (robot) coordinates
    # TODO

    cv2.imwrite('solution.png', canvas)
    small = resize(canvas, width=800)
    cv2.imshow('test', small)
    cv2.waitKey(0)


if __name__ == '__main__': main()
