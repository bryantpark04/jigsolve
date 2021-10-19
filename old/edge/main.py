import cv2 as cv
import numpy as np
import imutils

def main():
    img = cv.imread('laptop.jpg')

    rect = [(557, 684), (4092, 652), (4076, 2600), (623, 2668)]
    rect = np.array([np.array(x, dtype='float32') for x in rect])

    transformed = transform(img, rect)

    cv.imshow('', imutils.resize(transformed, height=700))
    cv.waitKey(0)

def transform(img, rect):
    tl, tr, br, bl = rect
    border = 200
    w = max(int(dist(tl, tr)), int(dist(bl, br))) + 2*border
    h = max(int(dist(tl, bl)), int(dist(tr, br))) + 2*border
    dst = np.array([
        [border, border],
        [w - 1 - border, border],
        [w - 1 - border, h - 1 - border],
        [border, h - 1 - border]
    ], dtype='float32')
    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(img, M, (w, h))


def dist(p1, p2):
  return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



if __name__ == '__main__':
    main()