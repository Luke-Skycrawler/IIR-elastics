# import fast_cd_pyb as fcd 
import fc_viewer as fcd
import numpy as np
T0 = None
init_guizmo = True
transform = "translate"
vis_cd = True

viewer = fcd.fast_cd_viewer()

from geometry_rod import V0, mid, F

# V0 = np.eye(3)
# mid = np.zeros(3)
# F = np.array([0, 1, 2]).reshape((1, 3)).astype(np.int32)
# V = V0.copy()



def guizmo_callback(A):
    global V
    T0 = A
    print(T0)
    print(V.shape)
    V = V0 @ A[:3, :3].T + mid + A[: 3, 3].reshape(1, 3)
    viewer.set_mesh(V, F, 0)

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


viewer.set_mesh(V, F, 0)

viewer.launch()

