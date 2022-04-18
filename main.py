from functools import partial
from pathlib import Path

import cv2
import numpy as np
from imutils import rotate_bound

from jigsolve.models import PuzzlePiece
from jigsolve.robot.arm import Arm
from jigsolve.robot.calibrate import calibrate_arm
from jigsolve.robot.coords import get_transformer
from jigsolve.solver.approx import eval_solution, solve_puzzle
from jigsolve.solver.fit import piece_displacements
from jigsolve.utils import grid_iter, rotate_piece, split_combined
from jigsolve.vision.camera import capture_image
from jigsolve.vision.image import binarize, find_contours, get_aruco, get_pieces, orientation, perspective_transform, rect_from_corners
from jigsolve.vision.piece import color_distribution, edge_types, get_origin

def main(arm):
    arm.use_absolute(True)
    arm.go_home()
    arm.move_to(x=-300, y=0, z=0, mode='G0')
    input('capture image')
    img = capture_image('http://192.168.69.1')

    # Test using image source.jpg
    # img = cv2.imread("source.jpg")

    wd = Path(__file__).resolve().parent
    cal = np.load(wd / 'calibration/camera.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # perspective transform
    corners = get_aruco(img)
    assert len(corners) == 4
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # find piece contours
    img_bw = binarize(img, threshold=30)
    cv2.imwrite('img/out/bin.png', img_bw)
    contours = find_contours(img_bw, min_area=20000, max_area=110000)
    print(f"{len(contours) = }")

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
    print(f"{len(solutions) = }")
    solution = min(solutions, key=partial(eval_solution, pieces))

    # find piece displacements
    solution_origin = (1440, 900)
    disp = piece_displacements(pieces, solution, solution_origin)

    # add pieces to image and prepare tool paths
    # a path is ((src_x, src_y), (dst_x, dst_y), cw_rot)
    paths = []
    for r, c in grid_iter(*solution.shape):
        temp = np.zeros_like(img)
        pi, pr = solution[r, c]
        pimg, _, porigin = split_combined(rotate_piece(pieces[pi].combined, pr))
        xd, yd = disp[r, c]
        ih, iw, _ = pimg.shape
        temp[yd:yd + ih, xd:xd + iw] = pimg
        img = cv2.add(img, temp)

        dst_point = cv2.minMaxLoc(porigin)[3]
        dst_point = (dst_point[0] + xd, dst_point[1] + yd)

        # pieces[pi].rot is cw, pr is multiples of 90 ccw
        prot = (pieces[pi].rot - 90 * pr) % 360
        # rotate ccw if more than half-turn
        if prot > 180:
            prot -= 360

        paths.append((pieces[pi].origin, dst_point, prot))

    cv2.imwrite('img/out/solution.png', img)

    dst_pts = cal = np.load(wd / 'calibration/coords.npy') # why are we storing this in cal?
    transformer = get_transformer(img, dst_pts)
    # transformer(img_x, img_y) -> (robot_x, robot_y)

    # draw paths
    for (src_x, src_y), (dst_x, dst_y), cw_rot in paths:
        # call robot?? just draw lines for now
        # path line
        cv2.line(img, (src_x, src_y), (dst_x, dst_y), (0, 0, 255), 1)
        # src and dst point circles
        cv2.circle(img, (src_x, src_y), 20, (255, 0, 0), 5)
        cv2.circle(img, (dst_x, dst_y), 20, (255, 0, 0), 5)
        # rotation lines
        cv2.line(img, (src_x, src_y), (src_x - int(40 * np.sin(cw_rot * np.pi / 180)), src_y - int(40 * np.cos(cw_rot * np.pi / 180))), (255, 0, 255), 3)
        cv2.line(img, (dst_x, dst_y), (dst_x, dst_y - 40), (255, 0, 255), 3)
    cv2.imwrite('img/out/solution.png', img)

    # execute paths
    for (src_x, src_y), (dst_x, dst_y), cw_rot in paths:
        # re-calibrate robot for each path
        dst_pts = calibrate_arm(dst_pts)
        transformer = get_transformer(img, dst_pts)

        arm.go_home()
        rx, ry = transformer(src_x, src_y)
        arm.move_to(x=rx, y=ry)
        arm.move_to(z=-58)
        input('pick up piece')
        src_angle = arm.get_current_position()[4]
        arm.air_picker_pick()
        arm.move_to(z=-25)
        arm.go_home()
        rx, ry = transformer(dst_x, dst_y)
        arm.move_to(x=rx, y=ry)
        input('rotate piece')
        dst_angle = arm.get_current_position()[4]
        arm.rotate_relative(src_angle - dst_angle + cw_rot)
        arm.move_to(z=-58)
        input('place piece')
        arm.move_to(z=-55) # makes robot quieter when letting piece go, but also moves the piece slightly
        arm.air_picker_place()
        arm.move_to(z=-25)
        arm.air_picker_neutral()
    arm.go_home()
    arm.close()

if __name__ == '__main__':
    arm = Arm('COM4')
    try:
        main(arm)
    except:
        arm.use_absolute(True)
        arm.air_picker_neutral()
        arm.go_home()
        arm.close()
