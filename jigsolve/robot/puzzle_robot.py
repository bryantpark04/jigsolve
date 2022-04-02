import cv2
import numpy as np

def pick_piece(dexarm):
    _, _, z_initial, _ = dexarm.get_current_position()
    dexarm.move_to(z=-56)
    dexarm.air_picker_pick()
    dexarm.move_to(z=z_initial)

def place_piece(dexarm):
    _, _, z_initial, _ = dexarm.get_current_position()
    dexarm.move_to(z=-56)
    dexarm.air_picker_neutral()
    dexarm.move_to(z=-54)
    dexarm.air_picker_place()
    dexarm.move_to(z=z_initial)
    dexarm.air_picker_stop()

def rotate_piece(deg):
    pass

def piece_pick_point(mask):
    '''Optimal point on a piece to pick it up

    Parameters
    ----------
    mask : np.ndarray
        Binary image, with puzzle piece white.

    Returns
    -------
    point : np.ndarray
        The optimal point of picking on the mask image. 
    '''
    # source: https://stackoverflow.com/questions/53646022/opencv-c-find-inscribing-circle-of-a-contour
    
    # Distance Transform
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # find max value
    _, max_val, _, max_loc = cv2.minMaxLoc(dt)

    # test image, to see whether circle makes sense
    # cv2.circle(mask, max_loc, int(max_val), (0, 255, 0), thickness=3)
    # cv2.imshow("test", mask)
    # cv2.waitKey(0)

    return np.array(max_loc)