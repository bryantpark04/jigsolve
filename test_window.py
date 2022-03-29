import cv2
from jigsolve.robot.pydexarm import Dexarm


def click_event(event, x, y, flags, params):
    pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{pt=}")
        # move robot arm to desired coordinates


def main():
    # todo: instantiate a dexarm at some point or pass it in
    img = cv2.imread("img/out/transformed.png")
    shape = img.shape[:2]
    print(f"{shape=}")
    while True:
        cv2.imshow('img', img)
        cv2.setMouseCallback('img', click_event)
        k = cv2.waitKeyEx(0)
        if k == 63232: # up
            pass
            # dexarm.move_to(z=5)
        elif k == 63233: # down
            pass
            # dexarm.move_to(z=-5)
        elif k == 63234: # left
            pass
            # dexarm._send_cmd(f'M2101 R-5\r')
        elif k == 63235: # right
            pass
            # dexarm._send_cmd(f'M2101 R5\r')
        elif k == ord('z'):
            pass
            # dexarm.air_picker_neutral()
        elif k == ord('x'):
            # dexarm.air_picker_pick()
            pass
        elif k == ord('c'):
            # dexarm.air_picker_place()
            pass
        else:
            break


if __name__ == "__main__":
    main()