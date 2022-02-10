import numpy as np
import cv2
from jigsolve.vision.image import find_contours, binarize

def get_side_contour(img, contour, side):
    '''Find a subsection of a contour.

    Returns the subsection of a contour that outlines the given
    side of the puzzle. Only works for right or left side (rotate the image
    for other sides as needed).

    Parameters
    ----------
    img : np.ndarray
        A cropped puzzle image
    contour : np.ndarray
        A single contour which outlines the piece in img
    side : int
        Which side to return the contour of. 0 is right,
        1 is left.

    Returns
    -------
    side_contour : np.ndarray
        A subsection of the given contour, outlining only the given
        side of the puzzle piece.
    '''
    contour = contour[:, 0].tolist()

    # find diagonal lines in img, with form x = my + b
    h, w = img.shape[:2]
    # line 1: top left to bottom right
    m1 = w/h
    b1 = 0
    # line 2: bottom left to top right
    m2 = -w/h
    b2 = w

    # find the row where the diagonal lines intersect
    intersect_row = int(h / 2)

    def eval_line(y, m, b):
        '''Evaluate x of given row for given line
        '''
        return m * y + b

    def point_in_side(point):
        '''Check whether the given point is part of the given side.
        '''
        if side == 0: # checking right side
            if point[1] < intersect_row:
                if eval_line(point[1], m2, b2) < point[0]:
                    return True
                else:
                    return False
            else:
                if eval_line(point[1], m1, b1) < point[0]:
                    return True
                else:
                    return False
        else: # side == 1, checking left side
            if point[1] < intersect_row:
                if eval_line(point[1], m1, b1) > point[0]:
                    return True
                else:
                    return False
            else:
                if eval_line(point[1], m2, b2) > point[0]:
                    return True
                else:
                    return False
    
    # find a point in contour outside of the side to be returned,
    # as the starting point for iterating contour points
    starting_idx = 0
    for i, point in enumerate(contour):
        if not point_in_side(point):
            starting_idx = i

    # beginning at starting_idx, iterate contour and find portion that
    # is in the given side
    side_contour = []
    contour_shifted = contour[starting_idx:] + contour[:starting_idx]
    for point in contour_shifted:
        if point_in_side(point):
            side_contour.append([point[0], point[1]])
    side_contour = np.array(side_contour)
    return side_contour

def find_x_on_side_contour(y, side_contour):
    '''Find the x value corresponding to a given y value in the side contour.
    '''
    # iterate over indices of points in side_contour
    for i in range(len(side_contour) - 1): # excludes last point
        # get point with lower y (p1) and higher y (p2)
        if side_contour[i][1] < side_contour[i + 1][1]:
            p1 = side_contour[i] # point 1
            p2 = side_contour[i + 1] # point 2
        else:
            p1 = side_contour[i + 1] # point 1
            p2 = side_contour[i] # point 2

        # check if y is between p1 and p2
        if y >= p1[1] and y <= p2[1]:
            # get x value at y using equation x = my + b
            m = (p2[0] - p1[0]) / (p2[1] - p1[1])
            b = p1[0] - m * p1[1]
            x = m * y + b
            return x
    return None

def puzzle_pieces_alignments(pieces, solution):
    '''Finds relative positions and rotations needed to fit pieces together.

    Parameters
    ----------
    pieces : list[PuzzlePiece]
        List of puzzle pieces.
    solution : dict[tuple[int], tuple[int]]
        Maps grid positions to piece indices and rotations.

    Returns
    -------
    idk, finish this later ig
    '''
    pass

def two_puzzle_piece_alignment(img_1, img_2):
    '''Finds relative positions and rotations needed to fit two pieces horizontally.

    Parameters
    ----------
    img_1 : np.ndarray
        Image of the left piece, isolated.
    img_2 : np.ndarray
        Image of the right piece, isolated. If the pieces fit vertically,
        the images should be rotated before being used as fields for this
        function.
    
    Returns
    -------
    idk
    '''

def main():
    # read test mask images
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

    # find contours
    bin_0 = binarize(img_0, threshold=10)
    bin_1 = binarize(img_1, threshold=10)
    contour_0 = max(find_contours(bin_0), key=cv2.contourArea)
    contour_1 = max(find_contours(bin_1), key=cv2.contourArea)

    # find side contours
    side_contour_0 = get_side_contour(img_0, contour_0, 0)
    side_contour_1 = get_side_contour(img_1, contour_1, 1)

    # find y ranges of side contours
    # for image 0
    min_y = h0
    max_y = 0
    for point in side_contour_0:
        if point[1] > max_y:
            max_y = point[1]
        if point[1] < min_y:
            min_y = point[1]
    y_range_0 = [min_y, max_y]
    # for image 1
    min_y = h1
    max_y = 0
    for point in side_contour_1:
        if point[1] > max_y:
            max_y = point[1]
        if point[1] < min_y:
            min_y = point[1]
    y_range_1 = [min_y, max_y]

    # Find y-displacement for image 1 that matches image 0. Starts with side
    # contours aligned on the top.

    # determine whether image 1 moves up or down
    move_up = False
    if y_range_1[1] - y_range_1[0] > y_range_0[1] - y_range_0[0]:
        # if side_contour_1 has more vertical range than side_contour_0
        move_up = True
    
    # iterate y-displacements
    reached_end = False
    y_dis = y_range_0[0] - y_range_1[0] # y displacement of image 1
    min_std = 9999
    y_dis_of_min_std = y_dis # IMPORTANT VARIABLE
    while not reached_end:
        # iterate over y values within y_range_0, with step 2 to find x displacements
        x_displacements = []
        for y in range(y_range_0[0], y_range_0[1], 2):
            # if y is also within y_range_1
            if y >= y_range_1[0] + y_dis and y <= y_range_1[1] + y_dis:
                # get x values on side_contours
                    x0 = find_x_on_side_contour(y, side_contour_0)
                    x1 = find_x_on_side_contour(y - y_dis, side_contour_1)
                    x_displacements.append(x1 - x0)
        
        # find standard deviation of x displacements
        std = np.std(x_displacements)
        print(std)
        # replace min_std if lower
        if std < min_std:
            min_std = std
            y_dis_of_min_std = y_dis

        # check for end of loop: if bottoms of contours aligned
        if y_range_0[1] == y_range_1[1] + y_dis:
            reached_end = True

        # increment
        if move_up:
            y_dis -= 1
        else: # move_up == False
            y_dis += 1

    # show image with unaligned pieces
    canvas = np.zeros((700, 800, 3), np.uint8)
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
    canvas = np.zeros((700, 800, 3), np.uint8)
    # draw img_0
    x_diff = 0
    y_diff = 100
    canvas[y_diff:y_diff+img_0.shape[0], x_diff:x_diff+img_0.shape[1]] = img_0
    # draw img_1
    x_diff = 400
    y_diff = 100 + y_dis_of_min_std
    canvas[y_diff:y_diff+img_1.shape[0], x_diff:x_diff+img_1.shape[1]] = img_1
    cv2.imshow("test", canvas)
    cv2.waitKey(0)

    # draw side contours TEST
    test_img = cv2.polylines(img_0, [side_contour_0], False, (0, 255, 0), 3)
    cv2.imshow("test", test_img)
    cv2.waitKey(0)
    test_img = cv2.polylines(img_1, [side_contour_1], False, (0, 255, 0), 3)
    cv2.imshow("test", test_img)
    cv2.waitKey(0)

    # draw contours TEST
    # test_0 = cv2.drawContours(img_0, [contour_0], -1, (0, 255, 0), 3)
    # test_0 = cv2.polylines(img_0, [contour_0], False, (0, 255, 0), 3)
    # cv2.imshow("test", test_0)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()