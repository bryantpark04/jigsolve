import numpy as np
import cv2
from jigsolve.vision.image import find_contours, binarize
from jigsolve.solver.fit import two_puzzle_piece_alignment

def main():
    # read test images
    img_0 = cv2.imread("puzzle_fit_images/img_0.png")
    h0, w0 = img_0.shape[:2]
    img_1 = cv2.imread("puzzle_fit_images/img_1.png")
    h1, w1 = img_1.shape[:2]

    # TEST: increase height of img_1
    canvas = np.zeros((450, 343, 3), np.uint8)
    x_diff = 0
    y_diff = 50
    canvas[y_diff:y_diff+img_1.shape[0], x_diff:x_diff+img_1.shape[1]] = img_1
    img_1 = canvas

    x_dis_final, y_dis_final = two_puzzle_piece_alignment(img_0, img_1)

    # show image with unaligned pieces
    canvas = np.zeros((700, 1000, 3), np.uint8)
    # draw img_0
    x_diff = 0
    y_diff = 100
    canvas[y_diff:y_diff+img_0.shape[0], x_diff:x_diff+img_0.shape[1]] = img_0
    # draw img_1
    x_diff = 400
    y_diff = 100
    canvas[y_diff:y_diff+img_1.shape[0], x_diff:x_diff+img_1.shape[1]] = img_1
    cv2.imshow("test", canvas)
    cv2.waitKey(0)

    # show image with aligned pieces
    canvas = np.zeros((700, 1000, 3), np.uint8)
    # draw img_0
    x_diff = 0
    y_diff = 100
    canvas[y_diff:y_diff+img_0.shape[0], x_diff:x_diff+img_0.shape[1]] = img_0
    # draw img_1
    x_diff = 400 + x_dis_final
    y_diff = 100 + y_dis_final
    canvas[y_diff:y_diff+img_1.shape[0], x_diff:x_diff+img_1.shape[1]] = img_1
    cv2.imshow("test", canvas)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
