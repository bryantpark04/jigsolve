import cv2
import numpy as np
from scipy.interpolate import interp1d

from jigsolve.vision.image import find_contours, binarize
from jigsolve.utils import grid_iter, rotate_piece

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

def piece_displacements(pieces, solution):
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

    # find width and height of puzzle grid from solution
    h, w = solution.shape

    # make a data structure mapping grid position to piece image
    pieces_grid = np.empty((h, w), dtype=object)
    for r, c in grid_iter(h, w):
        pi, pr = solution[r, c]
        piece_img = rotate_piece(pieces[pi].img, pr)
        pieces_grid[r, c] = piece_img

    # Starting at the top left corner of the solution grid, move down while
    # aligning pieces. Then, start at the top of the second column and move
    # down. For each piece, take the average of the displacement found from
    # aligning with the piece above and the piece below.

    # set alignment of top left piece
    disp = np.zeros((h, w, 2), dtype=int)
    disp[0, 0] = (100, 100)

    # set alignment of left-most column.
    # iterate rest of the rows in the solution grid.
    for r in range(1, h):
        ndisp = piece_align_v(pieces_grid[r - 1, 0], pieces_grid[r, 0])
        disp[r, 0] = ndisp + disp[r - 1, 0]

    # iterate rest of the columns in the solution grid
    for c in range(1, w):
        # set alignment of top piece in column c
        ndisp = piece_align_h(pieces_grid[0, c - 1], pieces_grid[0, c])
        disp[0, c] = ndisp + disp[0, c - 1]

        # iterate rest of the pieces in column c
        for r in range(1, h):
            # get displacements from horizontal pieces
            hdisp = piece_align_h(pieces_grid[r, c - 1], pieces_grid[r, c])
            hdisp = hdisp + disp[r, c - 1]

            # get displacements from vertical pieces
            vdisp = piece_align_v(pieces_grid[r - 1, c], pieces_grid[r, c])
            vdisp = vdisp + disp[r - 1, c]

            # set piece displacement to average displacements
            disp_avg = np.average((hdisp, vdisp), axis=0).round().astype(np.uint)
            disp[r, c] = disp_avg

    return disp

def piece_align_h(img_0, img_1):
    '''Finds relative displacements of image 1 needed to fit two pieces horizontally.

    Parameters
    ----------
    img_0 : np.ndarray
        Image of the left piece, isolated.
    img_1 : np.ndarray
        Image of the right piece, isolated. If the pieces fit vertically,
        the images should be rotated before being used as fields for this
        function.

    Returns
    -------
    displacements : tuple[int]
        x displacement, y displacement of image 1 necessary for the puzzle
        pieces to align.
    '''
    # get image dimensions
    h0 = img_0.shape[0]
    h1 = img_1.shape[0]

    # find contours
    bin_0 = binarize(img_0, threshold=10)
    bin_1 = binarize(img_1, threshold=10)
    contour_0 = max(find_contours(bin_0), key=cv2.contourArea)
    contour_1 = max(find_contours(bin_1), key=cv2.contourArea)

    # find side contours
    side_contour_0 = get_side_contour(img_0, contour_0, 0)
    side_contour_1 = get_side_contour(img_1, contour_1, 1)

    # Find y-displacement for image 1 that matches image 0 (y_dis_final).
    # Start with side contours aligned on the top. Minimize standard
    # deviation of the horizontal distances between the side contours.
    # -------------------------------------------------------------------

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

    # determine whether image 1 moves up or down
    move_up = False
    if y_range_1[1] - y_range_1[0] > y_range_0[1] - y_range_0[0]:
        # if side_contour_1 has more vertical range than side_contour_0
        move_up = True

    # iterate y-displacements
    reached_end = False
    y_dis_test = y_range_0[0] - y_range_1[0] # y displacement of image 1 being tested
    min_std = 9999
    y_dis_final = y_dis_test # IMPORTANT VARIABLE
    while not reached_end:
        # iterate over y values within y_range_0, with step 2 to find x displacements
        x_displacements = []
        for y in range(y_range_0[0], y_range_0[1], 2):
            # if y is also within y_range_1
            if y >= y_range_1[0] + y_dis_test and y <= y_range_1[1] + y_dis_test:
                # get x values on side_contours
                    x0 = find_x_on_side_contour(y, side_contour_0)
                    x1 = find_x_on_side_contour(y - y_dis_test, side_contour_1)
                    x_displacements.append(x1 - x0)

        # find standard deviation of x displacements
        std = np.std(x_displacements)
        # replace min_std if lower
        if std < min_std:
            min_std = std
            y_dis_final = y_dis_test

        # check for end of loop: if bottoms of contours aligned
        if y_range_0[1] == y_range_1[1] + y_dis_test:
            reached_end = True

        # increment
        if move_up:
            y_dis_test -= 1
        else: # move_up == False
            y_dis_test += 1

    # Find x-displacement for image 1 that matches image 0 (x_dis_final).
    # Start with side contours lined up, such that img_0 is on the left
    # img_1 is on the right. Minimize mean of horizontal distances between
    # side contours, with y_dis_final applied to img_1.
    # --------------------------------------------------------------------
    x0, y0 = side_contour_0.T
    f0 = interp1d(y0, x0)
    x1, y1 = side_contour_1.T
    y1 = y1 + y_dis_final
    f1 = interp1d(y1, x1)
    mid = (np.min(y0) + np.max(y0)) / 2
    x_dis_final = int(f0(mid) - f1(mid))

    return np.array([x_dis_final, y_dis_final])

def piece_align_v(img_0, img_1):
    '''Finds relative displacements of image 1 needed to fit two pieces vertically.

    Parameters
    ----------
    img_0 : np.ndarray
        Image of the left piece, isolated.
    img_1 : np.ndarray
        Image of the right piece, isolated.

    Returns
    -------
    displacements : tuple[int]
        x displacement, y displacement of image 1 necessary for the puzzle
        pieces to align.
    '''
    # rotate images 90 deg counter-clockwise
    img_0_rot = img_0.transpose(1, 0, 2)
    img_1_rot = img_1.transpose(1, 0, 2)

    # find displacements in rotated coordinate plane
    x_dis_rot, y_dis_rot = piece_align_h(img_0_rot, img_1_rot)

    # rotate displacements back into original coordinate plane
    x_dis = y_dis_rot
    y_dis = x_dis_rot
    return np.array([x_dis, y_dis])