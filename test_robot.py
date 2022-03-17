from jigsolve.robot.pydexarm import Dexarm
from jigsolve.robot.puzzle_robot import pick_piece

dexarm = Dexarm(port="COM4")

dexarm.go_home()
dexarm.set_module_type(2)
# pick_piece(dexarm)
# print(dexarm.get_current_position())
dexarm.move_to(z=-56)
dexarm.air_picker_pick()
dexarm.go_home()
dexarm.move_to(x=100, z=-56)
dexarm.air_picker_neutral()
dexarm.move_to(x=100, z=-54)
dexarm.air_picker_place()
dexarm.go_home()
dexarm.air_picker_stop()

dexarm.close()
