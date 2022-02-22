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

    # find width and height of puzzle grid from solution
    height = max(solution.keys(), key=lambda pos: pos[0])[0] + 1
    width = max(solution.keys(), key=lambda pos: pos[1])[1] + 1

    # make a data structure mapping grid position to piece image
    pieces_grid = [[0]*width]*height
    for r in range(height):
        for c in range(width):
            piece_idx, piece_rot = solution[(r, c)]
            piece_img_rot = pieces[piece_idx]
            piece_img = 0

            # rotate piece image
            if piece_rot == 0: # no rotation
                piece_img = piece_img_rot
            elif piece_rot == 1: # 90 deg ccw
                piece_img = cv2.rotate(piece_img_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif piece_rot == 2: # 180 deg
                piece_img = cv2.rotate(piece_img_rot, cv2.ROTATE_180)
            elif piece_rot == 3: # 270 deg ccw (or 90 deg cw)
                piece_img = cv2.rotate(piece_img_rot, cv2.ROTATE_90_CLOCKWISE)
            
            # add piece_img to pieces_grid
            pieces_grid[r][c] = piece_img

    # Starting at the top left corner of the solution grid, move down while
    # aligning pieces. Then, start at the top of the second column and move
    # down. For each piece, take the average of the displacement found from
    # aligning with the piece above and the piece below.
    
    # set alignment of top left piece
    displacements = [[(100, 100)]*width]*height # index of top left: [0][0]

    # set alignment of left-most column.
    # iterate rest of the rows in the solution grid.
    for r in range(1, height):
        displacements = two_puzzle_piece_align_vertical() # CONTINUE here

    # # iterate rest of the columns in the solution grid
    # for c in range(1, width):
    #     # iterate rest of the rows in the solution grid
    #     for r in range(1, height):

            
    print('test')

def two_puzzle_piece_align_horizontal(img_0, img_1):
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
    h0, w0 = img_0.shape[:2]
    h1, w1 = img_1.shape[:2]

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

    # find x ranges of side contours
    # for image 0
    min_x = h0
    max_x = 0
    for point in side_contour_0:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = point[0]
    x_range_0 = [min_x, max_x]
    # for image 1
    min_x = h1
    max_x = 0
    for point in side_contour_1:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = point[0]
    x_range_1 = [min_x, max_x]

    # iterate x-displacements
    reached_end = False
    x_dis_test = x_range_0[1] - x_range_1[0]
    min_avg_dist = 9999
    x_dis_final = x_dis_test # IMPORTANT VARIABLE
    while not reached_end:
        # iterate over y values within y_range_0, with step 2 to find x displacements
        x_displacements = []
        for y in range(y_range_0[0], y_range_0[1], 2):
            # if y is also within y_range_1
            if y >= y_range_1[0] + y_dis_final and y <= y_range_1[1] + y_dis_final:
                # get x values on side_contours
                    x0 = find_x_on_side_contour(y, side_contour_0)
                    x1 = find_x_on_side_contour(y - y_dis_final, side_contour_1) + x_dis_test
                    x_displacements.append(x1 - x0)
        
        # find average of x distances
        avg_dist = abs(np.mean(x_displacements))
        # replace min_avg_dist if lower
        if avg_dist < min_avg_dist:
            min_avg_dist = avg_dist
            x_dis_final = x_dis_test

        # check for end of loop: if left sides of image 0 and image 1 match
        if x_range_0[0] == x_range_1[0] + x_dis_test:
            reached_end = True
        
        # increment
        x_dis_test -= 1

    return -x_dis_final, y_dis_final

def two_puzzle_piece_align_vertical(img_0, img_1):
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
    img_0_rot = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_1_rot = cv2.rotate(img_1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # find displacements in rotated coordinate plane
    x_dis_rot, y_dis_rot = two_puzzle_piece_align_horizontal(img_0_rot, img_1_rot)

    # rotate displacements back into original coordinate plane
    x_dis = -y_dis_rot
    y_dis = x_dis_rot
    return x_dis, y_dis