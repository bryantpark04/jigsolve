from jigsolve.robot.pydexarm import Dexarm
import curses

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
        if c == 'q': 
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