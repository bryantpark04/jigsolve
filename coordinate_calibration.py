from jigsolve.robot.arm import Arm

import cv2
import numpy as np

from pathlib import Path
import curses

def main(stdscr):
    arm = Arm(port="COM4")
    arm.use_absolute(False)

    corners = []
    names = ('lower-left', 'lower-right')

    stdscr.clear()
    stdscr.addstr(2, 0, 'WASD: movement')
    stdscr.addstr(3, 0, 'UP/DOWN: z-axis')
    stdscr.addstr(4, 0, 'SHIFT: finer movement')
    stdscr.addstr(5, 0, 'H: go home')
    stdscr.addstr(6, 0, 'Q: quit')
    while len(corners) < 2:
        stdscr.addstr(0, 0, f'Move to the {names[len(corners)]} corner, then press SPACE.')
        c = stdscr.getkey()
        if c in 'WASD':
            amount = 1
            c = c.lower()
        else:
            amount = 10
        if c == 'h':
            arm.go_home()
        elif c == 'q':
            break
        elif c == 'w':
            arm.move_to(y=amount)
        elif c == 's':
            arm.move_to(y=-amount)
        elif c == 'a':
            arm.move_to(x=-amount)
        elif c == 'd':
            arm.move_to(x=amount)
        elif c == 'KEY_UP':
            arm.move_to(z=5)
        elif c == 'KEY_DOWN':
            arm.move_to(z=-5)
        elif c == ' ':
            x, y, *_ = arm.get_current_position()
            corners.append((x, y))
    arm.close()

    dst_pts = np.array(corners)
    p = Path(__file__) / '../calibration/coords.npy'
    np.save(p, dst_pts)

curses.wrapper(main)
