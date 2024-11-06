import taichi as ti
import numpy as np
from tobj import export_tobj, import_tobj
import igl
L, W = 1, 0.2
mu, rho, lam = 1e6, 1., 125
g = 10.
n_x = 20
n_yz = 4
dx = L / n_x
nq = 8
wq = 1 / nq
wq_2d = 4
n_elements, n_nodes = n_yz ** 2 * n_x, (n_yz + 1) ** 2 * (n_x + 1)
n_tets = n_elements * 6
n_boundary_elements = n_yz * n_x * 4
delta = 0.08
volume = dx ** 3
area = dx ** 3
n_unknowns = n_nodes * 3


@ti.func
def xc(I):
    '''
    coord of nodes
    '''
    i = I // ((n_yz + 1) ** 2)
    i_yz = I % ((n_yz + 1) ** 2)
    j = i_yz // (n_yz + 1)
    k = i_yz % (n_yz + 1)

    x = ti.Vector([i, j, k], ti.f32)
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


@ti.data_oriented
class RodGeometryGenerator:
    def __init__(self):


        self.indices = ti.field(ti.i32, (n_elements * 12 * 3))


        self.nodes = ti.field(ti.i32, (n_elements, 8))
        self.centers = ti.Vector.field(3, ti.f32, (n_elements))
        self.xcs = ti.Vector.field(3, ti.f32, (n_nodes))
        self.faces = ti.Vector.field(4, ti.i32, (6))
        self.boundary_centers = ti.Vector.field(3, ti.f32, (n_boundary_elements))
        self.T = ti.field(int, (n_tets, 4))
        self.tet = ti.Vector.field(4, ti.i32, (6))
        self.Dm = ti.Matrix.field(3, 3, ti.f32, (n_tets))
        self.n_tets = self.T.shape[0]
        self.n_nodes = self.xcs.shape[0]
        self.geometry()
    
    @ti.kernel
    def geometry(self):
        self.faces[0] = ti.Vector([0, 1, 3, 2])
        self.faces[1] = ti.Vector([4, 5, 1, 0])
        self.faces[2] = ti.Vector([2, 3, 7, 6])
        self.faces[3] = ti.Vector([4, 0, 2, 6])
        self.faces[4] = ti.Vector([1, 5, 7, 3])
        self.faces[5] = ti.Vector([5, 4, 6, 7])

        # self.tet[0] = ti.Vector([1, 4, 5, 3])
        # self.tet[1] = ti.Vector([4, 5, 3, 6])
        # self.tet[2] = ti.Vector([5, 3, 6, 7])
        self.tet[0] = ti.Vector([4, 1, 5, 3])
        self.tet[1] = ti.Vector([5, 4, 3, 6])
        self.tet[2] = ti.Vector([3, 5, 6, 7])

        self.tet[3] = ti.Vector([0, 1, 4, 2])
        self.tet[4] = ti.Vector([1, 4, 2, 3])
        self.tet[5] = ti.Vector([4, 2, 3, 6])

        for _e in range(n_elements):
            e = _trans(_e)
            self.centers[_e] = xc(e) + 0.5 * dx
            for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                I = e + i * (n_yz + 1) ** 2 + j * (n_yz + 1) + k
                J = i * 4 + j * 2 + k
                self.nodes[_e, J] = I

            for i in range(6):
                self.indices[_e * 36 + i * 6 + 0] = trans(self.faces[i][0]) + e
                self.indices[_e * 36 + i * 6 + 1] = trans(self.faces[i][1]) + e
                self.indices[_e * 36 + i * 6 + 2] = trans(self.faces[i][2]) + e
                self.indices[_e * 36 + i * 6 + 3] = trans(self.faces[i][2]) + e
                self.indices[_e * 36 + i * 6 + 4] = trans(self.faces[i][3]) + e
                self.indices[_e * 36 + i * 6 + 5] = trans(self.faces[i][0]) + e

            for i in range(6):
                for k in ti.static(range(4)):
                    self.T[_e * 6 + i, k] = trans(self.tet[i][k]) + e

        for i in self.xcs:
            self.xcs[i] = xc(i)



@ti.data_oriented
class TOBJLoader:
    def __init__(self):
        '''
        Before calling super().__init__(), make sure to define self.filename
        '''
        V, T = import_tobj(self.filename)
        self.n_nodes = V.shape[0]
        self.n_tets = T.shape[0]
        self.xcs = ti.Vector.field(3, ti.f32, (self.n_nodes))
        self.T = ti.field(int, (self.n_tets, 4))

        self.T.from_numpy(T)
        self.xcs.from_numpy(V)

        F = igl.boundary_facets(T)
        self.indices = ti.field(ti.i32, F.shape)
        self.indices.from_numpy(F)
        print(f"{self.filename} loaded, {self.n_nodes} nodes, {self.n_tets} tets")
