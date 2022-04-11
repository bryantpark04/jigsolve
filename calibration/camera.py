import cv2
import numpy as np
from tqdm import tqdm

from pathlib import Path

board = (6, 8)

obj_points = []
img_points = []
obj = np.zeros((board[0] * board[1], 3), np.float32)
obj[:,:2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = Path(__file__) / '../../img/calibration'
for path in tqdm(images.glob('*.jpg')):
  img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  ret, corners = cv2.findChessboardCorners(img, board, None)
  if ret:
    obj_points.append(obj)
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    assert len(corners) == board[0] * board[1]
    img_points.append(corners)

h, w = img.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (h, w), None, None)
newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

np.savez_compressed('camera.npz', mtx=mtx, dist=dist, newmtx=newmtx)
