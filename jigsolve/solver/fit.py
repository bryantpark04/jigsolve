import cv2
import numpy as np
from scipy.interpolate import interp1d

from jigsolve.vision.image import find_contours
from jigsolve.utils import grid_iter, rotate_piece

def get_side_contour(img, contour, left):
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
    left : bool
        If True, returns the left side. Otherwise, returns the right side.

    Returns
    -------
    side_contour : np.ndarray
        A subsection of the given contour, outlining only the given
        side of the puzzle piece.
    '''
    c = contour[:, 0]
    h, w = img.shape[:2]
    line = lambda m, b: lambda x: m * x + b
    lines = [line(-h/w, h), line(h/w, 0)]
    for i, p in enumerate(c):
        if p[1] <= lines[left](p[0]) and p[1] >= lines[not left](p[0]):
            c = np.concatenate((c[i:], c[:i]))
            break
    above = c[:,1] > lines[left](c[:,0])
    below = c[:,1] < lines[not left](c[:,0])
    return c[np.logical_and(above, below)]

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
        piece_img = rotate_piece(pieces[pi].mask, pr)
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
            # disp_avg = np.array([vdisp[0], hdisp[1]])
            disp[r, c] = disp_avg

    return disp

def smooth(x, length=12, discard=0.15):
    window = np.ones(length) / length
    x = np.convolve(x, window, 'valid')
    discard = int(discard * len(x))
    x = x[discard:-discard]
    return x

def get_critical(cont, max_diff=0.1):
    y = smooth(cont[:, 1])
    diff = np.abs(np.diff(y))
    points = y[np.where(diff < max_diff)]
    crit = np.sort(np.array([points[0], points[-1]]))
    return crit

def piece_align_h(left_img, right_img):
    '''Finds relative displacements of image 1 needed to fit two pieces horizontally.

    Parameters
    ----------
    left_img : np.ndarray
        Image of the left piece, isolated.
    right_img : np.ndarray
        Image of the right piece, isolated. If the pieces fit vertically,
        the images should be rotated before being used as fields for this
        function.

    Returns
    -------
    displacements : tuple[int]
        x displacement, y displacement of image 1 necessary for the puzzle
        pieces to align.
    '''

    # find contours
    con0 = max(find_contours(left_img), key=cv2.contourArea)
    con1 = max(find_contours(right_img), key=cv2.contourArea)

    # find side contours
    scon0 = get_side_contour(left_img, con0, False)
    scon1 = get_side_contour(right_img, con1, True)

    # Find y-displacement for image 1 that matches image 0 (y_dis_final).

    # wait, did I just invent convexity defects but worse? :facepalm:
    crit0 = get_critical(scon0)
    crit1 = get_critical(scon1)
    width0 = crit0[1] - crit0[0]
    width1 = crit1[1] - crit1[0]
    offset = int((np.abs(width0 - width1) / 2).round())
    if width0 < width1:
        # the left piece is smaller than the right piece
        y_dis_final = crit0[0] - offset - crit1[0]
    else:
        # the left piece is bigger than the right piece
        y_dis_final = crit0[0] + offset - crit1[0]

    # Find x-displacement for image 1 that matches image 0 (x_dis_final).
    # TODO: I hate this? do something else here
    x0, y0 = scon0.T
    f0 = interp1d(y0, x0)
    x1, y1 = scon1.T
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
    img_0_rot = img_0.T
    img_1_rot = img_1.T

    # find displacements in rotated coordinate plane
    x_dis_rot, y_dis_rot = piece_align_h(img_0_rot, img_1_rot)

    # rotate displacements back into original coordinate plane
    x_dis = y_dis_rot
    y_dis = x_dis_rot
    return np.array([x_dis, y_dis])
