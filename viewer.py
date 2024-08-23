import fast_cd_pyb as fcd 
import numpy as np
T0 = None
init_guizmo = True
transform = "translate"
vis_cd = True

viewer = fcd.fast_cd_viewer()



def guizmo_callback(A):
    T0 = A

def callback_key_pressed(key, modifier):
    global vis_cd, transform, viewer
    if (key == ord('g') or key == ord('G')):
        if (init_guizmo):
            if (transform == "translate"):
                transform = "rotate"
            elif (transform == "rotate"):
                transform = "scale"
            elif (transform == "scale"):
                transform = "translate"

            viewer.change_guizmo_op(transform)
        else:
            print("Guizmo not initialized, pass init_guizmo=True to the viewer constructor")
    if (key == ord('c') or key==ord('C') ):
        vis_cd = not vis_cd
    return False


def pre_draw_callback():
    pass

if init_guizmo:
    if T0 is None:
        T0 = np.identity(4).astype( dtype=np.float32, order="F")
    viewer.init_guizmo(True, T0, guizmo_callback, transform)
viewer.set_pre_draw_callback(pre_draw_callback)
viewer.set_key_callback(callback_key_pressed)
viewer.launch()

