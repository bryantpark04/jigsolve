import numpy as np

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def crop(img, box):
    x, y, w, h = box
    return img[y:y+h,x:x+w]
