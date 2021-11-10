import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

cal = np.load('calibration.npz')

directory = Path(__file__).parent

images = directory / 'images'
out = directory / 'out'
if not out.exists():
  out.mkdir()

for path in tqdm(images.glob('*.jpg')):
  img = cv2.imread(str(path))
  fixed = cv2.undistort(img, cal['mtx'], cal['dist'], None, cal['newmtx'])
  cv2.imwrite(str(out / path.name), fixed)
