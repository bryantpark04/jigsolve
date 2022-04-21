import curses
from pathlib import Path

import cv2
import numpy as np

from jigsolve.robot.arm import Arm
from jigsolve.vision.camera import capture_image
from jigsolve.vision.image import get_aruco, perspective_transform, rect_from_corners

def main(stdscr):
    # -------------get src_pts, the locations of aruco markers in the image-----------------
    num_calib_markers = 4 # number of aruco markers used for calibration, including two bottom corners
    arm = Arm(port="COM4")

    # capture image
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

    # get image coords of aruco markers
    markers = get_aruco(img)
    assert len(markers) == num_calib_markers + 2

    # perspective transform using corners
    markers_sorted = markers[markers[:, 1].argsort()] # sort columns by y-value of marker
    corners = np.concatenate((markers_sorted[:2], markers_sorted[-2:]))
    # corners: two points with the highest image y-vals, and two points with the lowest image y-vals
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # get and save source points for calibration
    src_pts = get_aruco(img) # new marker points from perspective-transformed image (excludes corners)
    assert src_pts.shape()[0] == num_calib_markers - 2
    p = Path(__file__) / '../calibration/src_pts_exc_corners.npy' # source points exclusing corners
    np.save(p, src_pts)

    # ---------------------get dst_pts, the robot coord points of markers----------------------
    arm.use_absolute(False)
    markers = []
    names = ('lower-left', 'lower-right')

    stdscr.clear()
    stdscr.addstr(2, 0, 'WASD: movement')
    stdscr.addstr(3, 0, 'UP/DOWN: z-axis')
    stdscr.addstr(4, 0, 'SHIFT: finer movement')
    stdscr.addstr(5, 0, 'H: go home')
    stdscr.addstr(6, 0, 'Q: quit')
    while len(markers) < num_calib_markers:
        stdscr.addstr(0, 0, f'Move to the next marker (left to right), then press SPACE.')
        c = stdscr.getkey()
        if c in 'WASD':
            amount = 1
            c = c.lower()
        else:
            amount = 10
        if c == 'h':
            arm.go_home()
        elif c == 'q':
            break
        elif c == 'w':
            arm.move_to(y=amount)
        elif c == 's':
            arm.move_to(y=-amount)
        elif c == 'a':
            arm.move_to(x=-amount)
        elif c == 'd':
            arm.move_to(x=amount)
        elif c == 'KEY_UP':
            arm.move_to(z=5)
        elif c == 'KEY_DOWN':
            arm.move_to(z=-5)
        elif c == ' ':
            x, y, *_ = arm.get_current_position()
            markers.append((x, y))
    arm.go_home()
    arm.close()

    if len(markers) != num_calib_markers: 
        return
    dst_pts = np.array(markers)
    p = Path(__file__) / '../calibration/dst_pts.npy'
    np.save(p, dst_pts)

curses.wrapper(main)
