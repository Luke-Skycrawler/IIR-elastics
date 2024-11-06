
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
