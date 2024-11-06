import numpy as np
import taichi as ti
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import eigh, null_space
import igl
from tobj import export_tobj
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

# damping
alpha = 6
beta = 1e-7
dt = 1e-4

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
class TetFEM:
    '''
    virtual interface for tetrahedral computing eigenvectors of tet FEM mesh
    '''
    def __init__(self):
        super().__init__()
        self.a = ti.field(ti.f32, (n_unknowns, n_unknowns))

        assert(hasattr(self, 'xcs'))
        assert(hasattr(self, 'T'))
        assert(hasattr(self, 'a'))

        print(f"init tet FEM")
        self.define_K()
        self.define_vis_interface()
        
    def define_vis_interface(self):
        '''
        define attributes for polyscope visualization
        '''
        self.V = self.xcs.to_numpy()
        self.mid = (np.array([L, W, W]) * 0.5).reshape(1, 3)
        self.V0 = self.V.copy() - self.mid
        self.F = self.indices.to_numpy().reshape(-1, 3).astype(np.int32)


    def eigs(self):        
        lam, Q = eigh(self.K)
        return lam, Q


    def define_K(self):
        '''
        adds np.ndarray K to self
        '''
        # only depends on self.a, self.T, self.xcs

        b = np.zeros((n_unknowns), np.float32)
        self.a.fill(0.0)
        # self.bulk_kernel(b)
        self.tet_kernel()
        # self.fill_hessian()
        self.K = self.a.to_numpy()

    @ti.kernel
    def tet_kernel(self): 
        for e in range(n_tets):
            
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
class Rod(TetFEM, RodGeometryGenerator):
    def __init__(self): 
        super().__init__()

@ti.data_oriented
class Rod_:
    def __init__(self):
        self.indices = ti.field(ti.i32, (n_elements * 12 * 3))


        self.nodes = ti.field(ti.i32, (n_elements, 8))
        self.centers = ti.Vector.field(3, ti.f32, (n_elements))
        self.xcs = ti.Vector.field(3, ti.f32, (n_nodes))
        self.faces = ti.Vector.field(4, ti.i32, (6))
        self.boundary_centers = ti.Vector.field(3, ti.f32, (n_boundary_elements))
        self.a = ti.field(ti.f32, (n_unknowns, n_unknowns))
        self.T = ti.field(int, (n_tets, 4))
        self.tet = ti.Vector.field(4, ti.i32, (6))
        self.Dm = ti.Matrix.field(3, 3, ti.f32, (n_tets))
        self.geometry()

        self.V = self.xcs.to_numpy()
        self.mid = (np.array([L, W, W]) * 0.5).reshape(1, 3)
        self.V0 = self.V.copy() - self.mid
        self.F = self.indices.to_numpy().reshape(-1, 3).astype(np.int32)

        self.get_K()    

        self.K = self.a.to_numpy()
        self.M = np.diag(np.ones(n_unknowns))

    def get_K(self):
        b = np.zeros((n_unknowns), np.float32)
        self.a.fill(0.0)
        # self.bulk_kernel(b)
        self.tet_kernel()
        # self.fill_hessian()

    def get_A1(self, N):
        V0 = self.V0
        J = np.zeros((N, 18))
        J[:, 0] = V0[:, 0]
        J[:, 4] = V0[:, 1]
        J[:, 8] = V0[:, 2]

        J[:, 9] = V0[:, 1]
        J[:, 10] = V0[:, 0]

        J[:, 12] = V0[:, 2]
        J[:, 14] = V0[:, 0]
        
        J[:, 16] = V0[:, 2]
        J[:, 17] = V0[:, 1]
        
        A1 = J.reshape(N * 3, 6)
        return A1


    def nullspace_eigs(self):
        # working imple to get eigen vector Phi of K, s.t. J^T Phi = 0 
        K, M = self.K, self.M

        N = K.shape[0] // 3
        A1 = self.get_A1(N)
        na1 = null_space(A1.T)
        tilde_K = na1.T @ K @ na1
        tilde_M = na1.T @ M @ na1
        lam, _Phi = eigh(tilde_K, tilde_M)
        Phi = na1 @ _Phi
        
        err = Phi.T @ K @ Phi - np.diag(lam)
        print(f"err = {np.max(np.abs(err))}")

    def KKT_eigs(self):
        # not working
        K = self.a.to_numpy()
        N = K.shape[0] // 3
        A1 = self.get_A1(N)
        row1 = np.hstack([K, A1])
        row2 = np.hstack([A1.T, np.zeros((6, 6))])
        sys = np.vstack([row1, row2])

        # M_diag = np.zeros(N * 3 + 6)
        # M_diag[:N * 3] = 1
        M_diag = np.ones(N * 3 + 6)
        b = np.diag(M_diag)
        # b[:N * 3, :N * 3] = np.identity(N * 3)
        lam, Phi_mu = eigh(sys, b)
        Lam = np.diag(lam[: -6])
        Phi = Phi_mu[: -6, : -6]
        n_inf = lambda x: np.max(np.abs(x))
        print(f"KKT lam = {lam[: 40]}")
        print(f"Phi = {Phi}, PhiTPhi = {np.diag(Phi.T @ b[: -6, :-6] @ Phi)[:10]}")
        # print(f"Phi = {Phi}, PhiTPhi = {np.diag(Phi.T @ b @ Phi)}")
        print(f"Phi^T J = {n_inf(Phi.T @ A1) }")
        print(f"Phi^T K Phi - lam = {n_inf((Phi.T @ K @ Phi) - Lam)}")
        # quit()
        return lam, Phi, A1
        
    @ti.kernel
    def boundary_condition(self, b: ti.types.ndarray()):
        for j, k in ti.ndrange(n_yz + 1, n_yz + 1):
            J = j * (n_yz + 1) + k
            for i in range(n_unknowns):
                for l in ti.static(range(3)):
                    self.a[J * 3 + l, i] = 0.0

                # FIXME: not exactly friendly to sparse matrix, fixed, shouldn't matter now
            for l in ti.static(range(3)):
                self.a[J * 3 + l, J * 3 + l] = 1.0
                b[J * 3 + l] = 0.0

    @ti.kernel
    def update_pos(self, u: ti.types.ndarray(), T: ti.types.ndarray()):
        r0 = ti.Vector([T[0, 0], T[0, 1], T[0, 2]])
        r1 = ti.Vector([T[1, 0], T[1, 1], T[1, 2]])
        r2 = ti.Vector([T[2, 0], T[2, 1], T[2, 2]])

        R = ti.Matrix.rows([r0, r1, r2])
        b = ti.Vector([T[0, 3], T[1, 3], T[2, 3]])

        mid = ti.Vector([L, W, W]) * 0.5
        for i in self.xcs:
            I = 3 * i
            v = ti.Vector([u[I], u[I+1], u[I+2]])
            x = xc(i) + v - mid
            
            
            
            self.xcs[i] = R @ x + b

    def export_tobj(self):
        export_tobj("bar2.tobj", self.V0, self.T.to_numpy())

    def compute_dFdx(self, Dm):

        ret = np.zeros((9, 12))

        s = np.sum(Dm, axis = 0)

        ret[0, 0] = -s[0]
        ret[3, 0] = -s[1]
        ret[6, 0] = -s[2]
        
        ret[1, 1] = -s[0]
        ret[4, 1] = -s[1]
        ret[7, 1] = -s[2]

        ret[2, 2] = -s[0]
        ret[5, 2] = -s[1]
        ret[8, 2] = -s[2]

        ret[0, 3] = Dm[0, 0]
        ret[3, 3] = Dm[0, 1]
        ret[6, 3] = Dm[0, 2]
        
        ret[1, 4] = Dm[0, 0]
        ret[4, 4] = Dm[0, 1]
        ret[7, 4] = Dm[0, 2]
        

        ret[2, 5] = Dm[0, 0]
        ret[5, 5] = Dm[0, 1]
        ret[8, 5] = Dm[0, 2]
        
        ret[0, 6] = Dm[1, 0]
        ret[3, 6] = Dm[1, 1]
        ret[6, 6] = Dm[1, 2]

        ret[1, 7] = Dm[1, 0]
        ret[4, 7] = Dm[1, 1]
        ret[7, 7] = Dm[1, 2]

        ret[2, 8] = Dm[1, 0]
        ret[5, 8] = Dm[1, 1]
        ret[8, 8] = Dm[1, 2]

        ret[0, 9] = Dm[2, 0]
        ret[3, 9] = Dm[2, 1]
        ret[6, 9] = Dm[2, 2]

        ret[1, 10] = Dm[2, 0]
        ret[4, 10] = Dm[2, 1]
        ret[7, 10] = Dm[2, 2]

        ret[2, 11] = Dm[2, 0]
        ret[5, 11] = Dm[2, 1]
        ret[8, 11] = Dm[2, 2]
        return ret
        
    def compute_hessian(self, Dm):
        dFdx = self.compute_dFdx(Dm)
        H = dFdx.T @ dFdx

        return H
    
    def fill_hessian(self):
        self.compute_Dm()
        Dmnp = self.Dm.to_numpy()

        for i in range(n_tets):
            Dm = Dmnp[i]
            H = self.compute_hessian(Dm)

            self.fill(i, H)

    @ti.kernel
    def fill(self, e: ti.i32, H: ti.types.ndarray()):
        for _i, _j in ti.ndrange(4, 4):
            i, j = self.T[e, _i], self.T[e, _j]
            for k, l in ti.static(ti.ndrange(3, 3)):
                self.a[i * 3 + k, j * 3 + l] += H[_i * 3 + k, _j * 3 + l] * self.compute_volume_tet(e)

    @ti.func
    def compute_volume_tet(self, e):
        x0 = self.xcs[self.T[e, 0]]
        x1 = self.xcs[self.T[e, 1]]
        x2 = self.xcs[self.T[e, 2]]
        x3 = self.xcs[self.T[e, 3]]

        ret = (x1 - x0).cross(x2 - x0).dot(x3 - x0) / 6.0
        return ret
        # return dx ** 3 / 6

    @ti.func
    def compute_Hij(self, F, i, j):
        dDsdx0 = ti.Vector([-1.0, -1.0, -1.0], ti.f32)
        dDsdx1 = ti.Vector([1.0, 0.0, 0.0], ti.f32)
        dDsdx2 = ti.Vector([0.0, 1.0, 0.0], ti.f32)
        dDsdx3 = ti.Vector([0.0, 0.0, 1.0], ti.f32)
        di = dDsdx0 if i == 0 else (dDsdx1 if i == 1 else (dDsdx2 if i == 2 else dDsdx3))
        dj = dDsdx0 if j == 0 else (dDsdx1 if j == 1 else (dDsdx2 if j == 2 else dDsdx3))
        return di.outer_product(dj) * 3.0

    @ti.kernel
    def compute_Dm(self):
        for e in range(n_tets):
            x0 = self.xcs[self.T[e, 0]]
            x1 = self.xcs[self.T[e, 1]]
            x2 = self.xcs[self.T[e, 2]]
            x3 = self.xcs[self.T[e, 3]]

            Dm = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])    
            inv_Dm = Dm.inverse()
            self.Dm[e] = inv_Dm

            
    @ti.kernel
    def tet_kernel(self): 
        for e in range(n_tets):
            
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

    def compute_H(self, Q):
        
        skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        H_row = lambda r: np.hstack([-skew(r), np.eye(3)])
        Gamma = np.vstack([H_row(r) for r in self.V0])
        H = Q.T @ Gamma / dt
        return H

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

    def eigs(self):
        
        # a_sparse = csr_matrix(_a)
        # u = spsolve(a_sparse, b)

        # lam, Q = eigsh(a_sparse, k = 10)
        # update_pos(Q[:, 0])
        # print(Q.T @ Q)
        lam, Q = eigh(self.K)
        return lam, Q

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


if __name__ == "__main__":
    ti.init()
    rod = Rod()
    rod.export_tobj()