from time import sleep
from urllib.parse import urljoin

import cv2
import numpy as np
import requests

def capture_image(host, wait=2):
    '''Captures a single frame from the Pi Camera.

    Returns
    -------
    frame : np.ndarray
        An image from the Pi Camera.
    '''
    requests.get(urljoin(host, '/cmd_pipe.php'), params={'cmd': 'im'})
    sleep(wait)
    data = requests.get(urljoin(host, '/media/image.jpg')).content
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img
