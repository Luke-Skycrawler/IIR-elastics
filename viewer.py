import fast_cd_pyb as fcd 
import numpy as np
import taichi  as ti
T0 = None
init_guizmo = True
transform = "translate"
vis_cd = True

viewer = fcd.fast_cd_viewer()
ti.init()
n_x = 20
n_yz = 4
n_elements, n_nodes = n_yz ** 2 * n_x, (n_yz + 1) ** 2 * (n_x + 1)
L, W = 1, 0.2
dx = L / n_x

xcs = ti.Vector.field(3, float, (n_nodes))
faces = ti.Vector.field(4, ti.i32, (6))
indices = ti.field(ti.i32, (n_elements * 12 * 3))
nodes = ti.field(ti.i32, (n_elements, 8))
centers = ti.Vector.field(3, float, (n_elements))

@ti.func
def xc(I):
    '''
    coord of nodes
    '''
    i = I // ((n_yz + 1) ** 2)
    i_yz = I % ((n_yz + 1) ** 2)
    j = i_yz // (n_yz + 1)
    k = i_yz % (n_yz + 1)

    x = ti.Vector([i, j, k], float)
    return dx * x


@ti.func
def trans(I):
    i, j, k = I // 4, (I % 4) // 2, I % 2
    return (n_yz + 1) ** 2 * i + (n_yz + 1) * j + k 

@ti.func
def _trans(I):
    i = I // ((n_yz) ** 2)
    i_yz = I % ((n_yz) ** 2)
    j = i_yz // (n_yz)
    k = i_yz % (n_yz)
    return (n_yz + 1) ** 2 * i + (n_yz + 1) * j + k 

@ti.kernel
def geometry():
    faces[0] = ti.Vector([0, 1, 3, 2])
    faces[1] = ti.Vector([4, 5, 1, 0])
    faces[2] = ti.Vector([2, 3, 7, 6])
    faces[3] = ti.Vector([4, 0, 2, 6])
    faces[4] = ti.Vector([1, 5, 7, 3])
    faces[5] = ti.Vector([5, 4, 6, 7])
    for _e in range(n_elements):
        e = _trans(_e)
        centers[_e] = xc(e) + 0.5 * dx
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            I = e + i * (n_yz + 1) ** 2 + j * (n_yz + 1) + k
            J = i * 4 + j * 2 + k
            nodes[_e, J] = I

        for i in range(6):
            indices[_e * 36 + i * 6 + 0] = trans(faces[i][0]) + e
            indices[_e * 36 + i * 6 + 1] = trans(faces[i][1]) + e
            indices[_e * 36 + i * 6 + 2] = trans(faces[i][2]) + e
            indices[_e * 36 + i * 6 + 3] = trans(faces[i][2]) + e
            indices[_e * 36 + i * 6 + 4] = trans(faces[i][3]) + e
            indices[_e * 36 + i * 6 + 5] = trans(faces[i][0]) + e
    for i in xcs:
        xcs[i] = xc(i)

geometry()
V = xcs.to_numpy().astype(np.float64)
mid = (np.array([L, W, W]) * 0.5).reshape(1, 3)
V0 = V.copy() - mid
F = indices.to_numpy().reshape(-1, 3).astype(np.int32)



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

