from jigsolve.robot.pydexarm import Dexarm

dexarm = Dexarm(port="COM4")

dexarm.go_home()

dexarm.close()
