import cv2
import numpy as np
import imutils
from collections import deque

def dist(p1, p2):
  return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rect_from_aruco_corners(corners):
  pts = np.array([np.mean(corner[0], axis=0) for corner in corners])
  rect = np.zeros((4, 2), dtype='float32')

  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  return rect

def get_aruco(img):
  aruco = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
  param = cv2.aruco.DetectorParameters_create()
  return cv2.aruco.detectMarkers(img, aruco, parameters=param)

def transform(img, rect):
  tl, tr, br, bl = rect
  border = 0
  w = max(int(dist(tl, tr)), int(dist(bl, br))) + 2*border
  h = max(int(dist(tl, bl)), int(dist(tr, br))) + 2*border
  dst = np.array([
    [border, border],
    [w - 1 - border, border],
    [w - 1 - border, h - 1 - border],
    [border, h - 1 - border]
  ], dtype='float32')
  M = cv2.getPerspectiveTransform(rect, dst)
  return cv2.warpPerspective(img, M, (w, h))

def find_contours(img):
  # transform image with aruco markers
  corners, ids, _ = get_aruco(img)
  rect = rect_from_aruco_corners(corners)
  img_transform = img = transform(img, rect)
  
  # binarize image
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  th, bw = cv2.threshold(img_gray, 50, 255, cv2.THRESH_OTSU)

  # contour detection
  contours, hierarchy = cv2.findContours(image=bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
  contours = list(filter(lambda c: cv2.contourArea(c) > 20000, contours)) # remove small contours
  
  return contours, img_transform


# read image from file - change later
img = cv2.imread('../test_images/IMG_1087_edited.jpg')

contours, img_transform = find_contours(img)

# display contours on the original image
cv2.drawContours(image=img_transform, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)
cv2.imwrite('contours_simple_image1.jpg', img_transform)
