import cv2
import numpy as np

def get_transformer(img, dst_pts):
  h, w = img.shape[:2]
  src_pts = np.array([[0, 0], [w - 1, 0]])
  M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
  def transform(x, y):
    v = np.array([x, h - 1 - y, 1])
    rx, ry = M @ v
    return rx, ry
  return transform
