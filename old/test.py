import cv2
import numpy as np
import imutils
from collections import deque

def show(name, img):
  out = imutils.resize(img, width=800)
  cv2.imshow(name, out)
  cv2.waitKey(0)
  cv2.destroyWindow(name)

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

def contained(p, s):
  for pc, sc in zip(p, s):
    if pc < 0 or pc >= sc:
      return False
  return True

def remove_background(img, p=(40, 40)):
  q = deque()
  q.append(p)
  visited = set()
  while q:
    x, y = p = q.popleft()
    if p in visited: continue
    visited.add(p)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      n = (x+dx, y+dy)
      if not contained(n, img.shape): continue
      if np.linalg.norm(img[n].astype(np.int8) - img[p].astype(np.int8)) < 5:
        q.append(n)
  for p in visited:
    img[p] = (0, 0, 0)
  return img

img = cv2.imread('20211014_132106.jpg')
# img = imutils.resize(img, width=800) #sdf
# img = cv2.medianBlur(img, 7)
# show('original', img)

corners, ids, _ = get_aruco(img) # fsdsadfasfsd
# copy = np.copy(img)
# cv2.aruco.drawDetectedMarkers(copy, corners, ids)
rect = rect_from_aruco_corners(corners) #  dsfsfdsf
# cv2.polylines(copy, [rect.astype(int)], True, (0, 255, 0), 5)
# show('markers', copy)
img = transform(img, rect) # dasfsdfs
# img = remove_background(img, (200, 200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th, bw = cv2.threshold(img_gray, 100, 255, cv2.THRESH_OTSU)
# img_blur = cv2.GaussianBlur(bw, (21, 21), 0)
# edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# draw contours on the original image
image_copy = img.copy()
print(type(contours))
contours = list(filter(lambda c: cv2.contourArea(c) > 20000, contours))
# for c in contours:
#   print(cv2.contourArea(c))
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)
# see the results
cv2.imwrite('contours_simple_image1.jpg', image_copy)

# cv2.imwrite('edges.png', bw)

