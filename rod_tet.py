
import taichi as ti
import numpy as np
from geometry import RodGeometryGenerator, TOBJLoader
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import eigh, null_space
L, W = 1, 0.2
mu, rho, lam = 1e6, 1., 125
g = 10.
n_x = 20
n_yz = 4
dx = L / n_x

volume = dx ** 3
# damping
alpha = 6
beta = 1e-7
dt = 1e-4


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
class InterfaceIIRSolver: 
    def __init__(self):
        super().__init__()

        self.define_vis_interface()

    def define_vis_interface(self):
        '''
        define attributes for polyscope visualization
        '''
        self.V = self.xcs.to_numpy()
        self.mid = (np.array([L, W, W]) * 0.5).reshape(1, 3)
        self.V0 = self.V.copy() - self.mid
        self.F = self.indices.to_numpy().reshape(-1, 3).astype(np.int32)
        
    '''
    interface for IIR solver
    '''
    def compute_H(self, Q):
        
        skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        H_row = lambda r: np.hstack([-skew(r), np.eye(3)])
        Gamma = np.vstack([H_row(r) for r in self.V0])
        H = Q.T @ Gamma / dt
        return H

    def update_pos(self, u, T):
        R = T[:, :3]
        b = T[:, 3]
        x = R @ (self.V0 + u.reshape(-1, 3)).T + b
        self.xcs.from_numpy(x.T)

    # @ti.kernel
    # def update_pos(self, u: ti.types.ndarray(), T: ti.types.ndarray()):
    #     r0 = ti.Vector([T[0, 0], T[0, 1], T[0, 2]])
    #     r1 = ti.Vector([T[1, 0], T[1, 1], T[1, 2]])
    #     r2 = ti.Vector([T[2, 0], T[2, 1], T[2, 2]])

    #     R = ti.Matrix.rows([r0, r1, r2])
    #     b = ti.Vector([T[0, 3], T[1, 3], T[2, 3]])

    #     mid = ti.Vector([L, W, W]) * 0.5
    #     for i in self.xcs:
    #         I = 3 * i
    #         v = ti.Vector([u[I], u[I+1], u[I+2]])
    #         x = xc(i) + v - mid
            
            
            
    #         self.xcs[i] = R @ x + b
        



@ti.data_oriented
class TobjFEM(InterfaceIIRSolver, TetFEM, TOBJLoader):
    def __init__(self, filename = "bunny_5.tobj"):
        self.filename = filename
        super().__init__()
    
Rod = TobjFEM
    
@ti.data_oriented
class Rod__(InterfaceIIRSolver, TetFEM, RodGeometryGenerator):
    def __init__(self): 
        super().__init__()
