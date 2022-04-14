from jigsolve.robot.arm import Arm

arm = Arm('COM4')
arm.use_absolute(True)
arm.air_picker_neutral()
arm.go_home()