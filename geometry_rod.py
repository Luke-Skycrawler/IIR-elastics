import taichi as ti 
import numpy as np
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
