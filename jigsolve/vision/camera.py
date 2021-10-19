import cv2

def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    return frame

cv2.imshow('frame', capture_image())
cv2.waitKey(0)