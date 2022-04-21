import curses
from pathlib import Path

import numpy as np


def get_calibrator(arm, old_dst_pts):
    def calibrator(stdscr):
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
            if old_dst_pts is not None: # move arm to the previous position of this corner to make calibration a little faster
                arm.use_absolute(True)
                old_x, old_y = old_dst_pts[len(corners)]
                arm.go_home()
                arm.move_to(x=old_x, y=old_y)
            arm.use_absolute(False)
            

            while True:
                c = stdscr.getkey()
                if c in 'WASD':
                    amount = 1
                    c = c.lower()
                else:
                    amount = 10
                if c == 'h':
                    arm.go_home()
                elif c == 'q':
                    raise ValueError('lmao')
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
                    break
        arm.go_home()
        assert len(corners) == 2 
        dst_pts = np.array(corners)
        p = Path(__file__) / '../../../calibration/coords.npy'
        np.save(p, dst_pts)
        return dst_pts

    return calibrator

def calibrate_arm(arm, old_dst_pts=None):
    return curses.wrapper(get_calibrator(arm, old_dst_pts)) # not sure if dst_pts will be passed through the curses.wrapper() call
