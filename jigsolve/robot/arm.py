from jigsolve.robot.pydexarm import Dexarm

class Arm(Dexarm):
  def __init__(self, port):
    super().__init__(port)
    self.set_module_type(2)

  def use_absolute(self, absolute):
    if absolute:
      self._send_cmd('G90\r')
    else:
      self._send_cmd('G91\r')
