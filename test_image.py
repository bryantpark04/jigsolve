from pathlib import Path

import cv2
import numpy as np
from imutils import resize, rotate_bound

from jigsolve.models import PuzzlePiece
from jigsolve.solver import eval_solution, puzzle_dimensions, solve_puzzle
from jigsolve.utils import crop, grid_iter, rotate
from jigsolve.vision.image import binarize, find_contours, get_aruco, get_pieces, orientation, perspective_transform, rect_from_corners
from jigsolve.vision.piece import color_distribution, edge_types

import matplotlib.pyplot as plt

def show_solution(idx, h, w, pieces, solution):
    fig, axs = plt.subplots(h, w)
    fig.suptitle(f'Solution {idx}')
    ax_iter = iter(axs.flat)
    for r, c in grid_iter(h, w):
        i, r = solution[r, c]
        img = rotate_bound(pieces[i].img, -90 * r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # find contour for the piece
        bin = binarize(img, threshold=10)
        contour = max(find_contours(bin), key=cv2.contourArea)
        
        # make background of each piece white
        fill_color = [255, 255, 255] # BGR
        mask_value = 255 # white
        stencil  = np.zeros(img.shape[:-1]).astype(np.uint8)
        cv2.fillPoly(stencil, [contour], mask_value)
        img[stencil != mask_value] = fill_color
        
        # crop image to fit the piece
        box = cv2.boundingRect(contour)
        img = crop(img, box)

        ax = next(ax_iter)
        ax.axis('off')
        ax.imshow(img)
    plt.savefig(f'solutions/{idx:04d}.png', dpi=300)
    plt.close()

def main():
    wd = Path(__file__).resolve().parent
    test_image = wd / 'test_images/turck2.jpg'
    img = cv2.imread(str(test_image.resolve()))

    cal = np.load(wd / 'calibration/calibration.npz')
    img = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])

    # perspective transform
    corners = get_aruco(img)
    rect = rect_from_corners(corners)
    img = perspective_transform(img, rect)

    # mild cropping (not necessary in final product)
    img = img[70:-67]

    # find piece contours
    img_bw = binarize(img, threshold=10)
    contours = find_contours(img_bw, min_area=18000)

    pieces = []
    for box, piece, mask in get_pieces(img, contours, padding=20):
        angle = orientation(mask, num_bins=16)
        if angle > 45: angle -= 90
        piece = rotate_bound(piece, angle)
        mask = rotate_bound(mask, angle)
        edges = edge_types(mask)
        hist = tuple(color_distribution(piece, mask))
        pieces.append(PuzzlePiece(piece, mask, hist, angle, box, edges))

    (h, w), solutions = solve_puzzle(pieces)
    # solutions = list(filter(lambda s: s[0, 0] == (9, 3), solutions))
    print(len(solutions))
    scores = [eval_solution(h, w, pieces, solution) for solution in solutions]
    for idx in np.argsort(scores):
        print(idx)
        print(solutions[idx])
        show_solution(idx, h, w, pieces, solutions[idx])
        break

if __name__ == '__main__': main()
