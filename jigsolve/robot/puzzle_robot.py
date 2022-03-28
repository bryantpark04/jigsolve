

def pick_piece(dexarm):
    _, _, z_initial, _ = dexarm.get_current_position()
    dexarm.move_to(z=-56)
    dexarm.air_picker_pick()
    dexarm.move_to(z=z_initial)

def place_piece(dexarm):
    _, _, z_initial, _ = dexarm.get_current_position()
    dexarm.move_to(z=-56)
    dexarm.air_picker_neutral()
    dexarm.move_to(z=-54)
    dexarm.air_picker_place()
    dexarm.move_to(z=z_initial)
    dexarm.air_picker_stop()

def rotate_piece(deg):
    pass