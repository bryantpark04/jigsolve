from pathlib import Path

import cv2
import numpy as np
from imutils import resize

from jigsolve.robot.pydexarm import Dexarm
from jigsolve.vision.camera import capture_image
from jigsolve.vision.image import get_aruco, perspective_transform, rect_from_corners

def main():
    img = capture_image('http://192.168.69.1')

    wd = Path(__file__).resolve().parent
    cal = np.load(wd / 'calibration/calibration.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # perspective transform
    corners = get_aruco(img)

    print(len(corners))
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    img = resize(img, width=800)
    cv2.imwrite('img/out/perspective.png', img)
    # todo: instantiate a dexarm at some point or pass it in
    dexarm = Dexarm(port="COM4")
    dexarm._send_cmd('G90\r')
    dexarm.move_to(x=-290, y=30, z=0, mode='G0')
    h, w, _ = img.shape
    src_pts = np.array([[0, 0], [w - 1, 0]])
    dst_pts = np.array([[-314.0, 71.0], [303.0, 86.0]])
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    def click_event(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN: return
        v = np.array([x, h - 1 - y, 1])
        rx, ry = M @ v
        dexarm.move_to(x=rx, y=ry, z=-30)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', click_event)
    cv2.waitKey(0)
    cv2.destroyWindow('img')
    dexarm.move_to(x=-290, y=30, z=0, mode='G0')

if __name__ == "__main__":
    main()
