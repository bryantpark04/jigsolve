import cv2


def capture_image():
    '''Captures a single frame from the webcam.

    This function creates a VideoCapture from the default webcam and returns a captured frame.

    Returns
    -------
    frame : np.ndarray
        An image from the default webcam.
    '''

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release() # how to avoid continously initializing and releasing the camera?
    return frame