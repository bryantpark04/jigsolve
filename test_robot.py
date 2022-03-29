from jigsolve.robot.pydexarm import Dexarm
import curses
import cv2
import numpy as np

def find_transformation(src_pts, dst_pts):
    # source: https://stackoverflow.com/questions/33141310/estimate-2d-transformation-between-two-sets-of-points-using-ransac
    # # Find the transformation between points, standard RANSAC
    # transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Compute a rigid transformation (without depth, only scale + rotation + translation) and RANSAC
    transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return transformation_rigid_matrix

def main(stdscr):
    dexarm = Dexarm(port="COM4")    
    dexarm._send_cmd('G91\r')
    dexarm.set_module_type(2)

    stdscr.clear()
    while True:
        c = stdscr.getkey()
        stdscr.addstr(0, 0, c)
        if c in 'WASD':
            amount = 1
            c = c.lower()
        else:
            amount = 10
        if c == 'h':
            dexarm.go_home()
        elif c == 'q': 
            break
        elif c == 'w':
            dexarm.move_to(y=amount, mode='G0')
        elif c == 's':
            dexarm.move_to(y=-amount, mode='G0')
        elif c == 'a':
            dexarm.move_to(x=-amount, mode='G0')
        elif c == 'd':
            dexarm.move_to(x=amount, mode='G0')
        elif c == 'KEY_UP':
            dexarm.move_to(z=5)
        elif c == 'KEY_DOWN':
            dexarm.move_to(z=-5)
        elif c == 'z':
            dexarm.air_picker_neutral()
        elif c == 'x':
            dexarm.air_picker_pick()
        elif c == 'c':
            dexarm.air_picker_place()
        elif c == 'KEY_LEFT':
            dexarm._send_cmd(f'M2101 R-5\r')
        elif c == 'KEY_RIGHT':
            dexarm._send_cmd(f'M2101 R5\r')

    print(dexarm.get_current_position())

    dexarm.close()

curses.wrapper(main)