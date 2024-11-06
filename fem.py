
import taichi as ti
import numpy as np
from geometry import RodGeometryGenerator, TOBJLoader
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import eigh, null_space

from params import *

@ti.data_oriented
class TetFEM:
    '''
    virtual interface for tetrahedral computing eigenvectors of tet FEM mesh

    assert self.n_nodes is defined
    '''
    def __init__(self):
        super().__init__()
        n_unknowns = 3 * self.n_nodes
        self.a = ti.field(ti.f32, (n_unknowns, n_unknowns))

        assert(hasattr(self, 'xcs'))
        assert(hasattr(self, 'T'))
        # assert(hasattr(self, 'a'))

        self.define_K()
        print(f"init tet FEM")


    def eigs(self):        
        lam, Q = eigh(self.K)
        return lam, Q


    def define_K(self):
        '''
        adds np.ndarray K to self
        '''
        # only depends on self.a, self.T, self.xcs

        self.a.fill(0.0)
        # self.bulk_kernel(b)
        self.tet_kernel()
        self.K = self.a.to_numpy()
        n_unknowns = self.n_nodes * 3   
        self.M = np.diag(np.ones(n_unknowns))

    @ti.kernel
    def tet_kernel(self): 
        for e in range(self.n_tets):
            
            x = self.xq_tet(e)
            for _i in range(4):
                i = self.T[e, _i]
                
                bi, dbidx = self.bf_tet(e, _i, x)

                for _j in range(4):
                    j = self.T[e, _j]
                    bj, dbjdx = self.bf_tet(e, _j, x)
                    for k in ti.static(range(3)):
                        grad_v = ti.Vector.unit(3, k, ti.f32).outer_product(dbidx)
                        eps = (grad_v + grad_v.transpose()) / 2
                        c = eps.trace() * lam * dbjdx + 2 * mu * eps.transpose() @ dbjdx
                        for l in ti.static(range(3)):
                            self.a[i * 3 + k, j * 3 + l] += c[l] * volume

    @ti.func
    def bf_tet(self, e, _i, x):
        bi = 0.25
        n = self.normal(e, _i)
        x0 = self.xcs[self.T[e, _i]]
        k = 0.75 / ((x0 - x).dot(n))
        dbidx = n * k
        
        return bi, dbidx

    @ti.func
    def normal(self, e, _i):
        v0 = self.T[e, 0]
        v1 = self.T[e, 1]
        v2 = self.T[e, 2]

        v = self.T[e, _i]
        x = self.xcs[v]
        if _i == 0:
            v0 = self.T[e, 3]
        if _i == 1:
            v1 = self.T[e, 3]
        if _i == 2:
            v2 = self.T[e, 3]

        x0 = self.xcs[v0]
        x1 = self.xcs[v1]
        x2 = self.xcs[v2]

        n = (x1 - x0).cross(x2 - x0).normalized()
        if n.dot(x - x0) < 0:
            n = -n
        return n
    

    @ti.func
    def xq_tet(self, e):
        x0 = self.xcs[self.T[e, 0]]
        x1 = self.xcs[self.T[e, 1]]
        x2 = self.xcs[self.T[e, 2]]
        x3 = self.xcs[self.T[e, 3]]

        return 0.25 * (x0 + x1 + x2 + x3)

@ti.data_oriented
class QuadFEM: 
    def __init__(self):
        super().__init__()
        n_unknowns = 3 * self.n_nodes
        self.a = ti.field(ti.f32, (n_unknowns, n_unknowns)) 
        assert(hasattr(self, 'xcs'))
        assert(hasattr(self, 'T'))

        self.define_K()
        print(f"init quad FEM")
    
    def eigs(self):
        lam, Q = eigh(self.K)
        return lam, Q
    
    def define_K(self):
        '''
        adds np.ndarray K to self
        '''
        # only depends on self.a, self.T, self.xcs

        n_unknowns = self.n_nodes * 3   
        self.a.fill(0.0)
        b = np.zeros((n_unknowns), np.float32)
        self.bulk_kernel(b)
        self.K = self.a.to_numpy()
        self.M = np.diag(np.ones(n_unknowns))


    @ti.kernel
    def bulk_kernel(self, b: ti.types.ndarray()):
        for e in range(n_elements):
            for iq in range(nq):
                x = xq(e, iq)
                for _i in range(8):
                    i = self.nodes[e, _i]

                    # if i < (n_yz + 1) ** 2:
                    #     # filter out boundary nodes
                    #     continue

                    bi, dbidx = bf_trilinear(i, x)
                    # grad_v = ti.Matrix.cols([dbidx, dbidx, dbidx])
                    # grad_v = (grad_v + grad_v.transpose()) / 2

                    f = fx(x)
                    for k in ti.static(range(3)):
                        b[i * 3 + k] += f[k] * bi * wq * volume
                        # \int fv dx
                        
                    for _j in range(8):
                        j = self.nodes[e, _j]
                        bj, dbjdx = bf_trilinear(j, x)
                        for k in ti.static(range(3)):
                            # fill in by row

                            grad_v = ti.Vector.unit(3, k, ti.f32).outer_product(dbidx)
                            eps = (grad_v + grad_v.transpose()) / 2
                            
                            c = eps.trace() * lam * dbjdx + 2 * mu * eps.transpose() @ dbjdx
                            for l in ti.static(range(3)):
                                self.a[i * 3 + k, j * 3 + l] += c[l] * wq * volume



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

@ti.func
def bf_trilinear(i, x):
    # FIXME: add divergence
    u = x - xc(i)
    bi = 0.
    dbidx = ti.Vector.zero(ti.f32, 3)
    abs_u = ti.abs(u)
    distance = ti.max(abs_u[0], abs_u[1], abs_u[2])
    if distance < dx:
        # so that excludes divided by zero 
        t = ti.abs(dx - abs_u)
        bi = t[0] * t[1] * t[2] / (dx ** 3)

        for k in ti.static(range(3)):
            if u[k] > 0:
                dbidx[k] = - bi / (dx - u[k])
            elif u[k] <= 0:
                dbidx[k] = bi / (u[k] + dx)
    return bi, dbidx

# @ti.kernel 
# def quadrature():
#     '''
#     8-pt quadrature
#     '''
#     a = 0.5773502692

# @ti.kernel
# def test():
#     x = xq(0, 0)
#     print(bf_trilinear(0, x))

@ti.func
def xq(e, I):
    a = 0.5773502692
    J = ti.Vector([I // 4, (I % 4) // 2, I % 2], ti.i32)
    x0 = xc(_trans(e))
    x = x0
    for k in ti.static(range(3)):
        x[k] += ((1 - a) / 2 + a * J[k]) * dx 
    return x

@ti.func
def xq_2d(e, I):
    a = 0.5773502692
    x0 = xc(boundary_nodes(e, 0))
    xi = xc(boundary_nodes(e, I))
    x = x0 + dx * (1 - a) / 2 + (xi - x0) * a
    return x

@ti.func
def boundary_nodes(e, I):
    '''
    e: boundary element (square) index
    I: node index within element
    '''
    side = e // (n_yz * n_x)
    t = e % (n_yz * n_x)
    i = t // n_yz
    j = 0
    k = 0
    if side == 0 or side == 2:
        k = t % n_yz
        j = 0 if side == 0 else n_yz
    else:
        j = t % n_yz
        k = 0 if side == 1 else n_yz

    # i, j, k: index of the bottom left 
    bl = (n_yz + 1) ** 2 * i + (n_yz + 1) * j + k 
    di = (I // 2) * (n_yz + 1) ** 2
    djj = 1 if side == 0 or side == 2 else (n_yz + 1)
    dj = (I % 2) * djj
    return bl + di + dj

@ti.func
def normal(e):
    side = e // (n_yz * n_x)
    ny = ti.Vector([0.0, 1.0, 0.0])
    nz = ti.Vector([0.0, 0.0, 1.0])
    n = ti.Vector.zero(ti.f32, 3)
    if side == 0:
        # bottom
        n = -ny
    elif side == 1:
        # left
        n = -nz
    elif side == 2:
        # top
        n = ny
    elif side == 3:
        # right 
        n = nz        
    return n

@ti.func
def fx(x):
    return -rho * g * ti.Vector.unit(3, 1, ti.f32)

        


@ti.kernel
def boundary_kernel(b: ti.types.ndarray()):
    for e in range(n_boundary_elements):
        for iq in range(4):
            x = xq_2d(e, iq)
            for _i in range(4):
                i = boundary_nodes(e, _i)
                if i < (n_yz + 1) ** 2:
                    continue
                bi, dbidx = bf_trilinear(i, x)
                v = ti.Vector([bi, bi, bi])
                
                for _j in range(4):
                    j = boundary_nodes(e, _j)
                    n = normal(e)
                    bj, dbjdx = bf_trilinear(j, x)
                    sigma = lam * dbidx.dot(ti.Vector([1., 1., 1.], ti.f32)) * ti.Matrix.identity(ti.f32, 3) + mu * (ti.Matrix.cols([dbjdx, dbjdx, dbjdx]) + ti.Matrix.rows([dbjdx, dbjdx, dbjdx]))
                    rhs = (n.transpose() @ sigma @ v * area * wq_2d)[0, 0]
                    b[i] += rhs