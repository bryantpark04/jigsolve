import cv2
import numpy as np

def get_transformer(img, src_pts_exc_corners, dst_pts):
  h, w = img.shape[:2]
  src_pts_corners = np.array([[0, 0], [w - 1, 0]])
  src_pts = np.concatenate((src_pts_corners, src_pts_exc_corners))
  src_pts = src_pts[src_pts[:, 0].argsort()] # sort by x-value
  M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
  def transform(x, y):
    v = np.array([x, h - 1 - y, 1])
    rx, ry = M @ v
    return rx, ry
  return transform
