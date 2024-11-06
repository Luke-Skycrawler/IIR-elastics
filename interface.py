import taichi as ti
import numpy as np
from params import *
from geometry import RodGeometryGenerator, TOBJLoader
from fem import TetFEM, QuadFEM
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
        R = T[: 3, :3]
        b = T[: 3, 3]
        x = (R @ (self.V0 + u.reshape(-1, 3)).T) + b.reshape(3, 1)
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
    
    
@ti.data_oriented
class RodTet(InterfaceIIRSolver, TetFEM, RodGeometryGenerator):
    def __init__(self): 
        super().__init__()

@ti.data_oriented
class RodQuad(InterfaceIIRSolver, QuadFEM, RodGeometryGenerator):
    def __init__(self): 
        super().__init__()
        
# Rod = TobjFEM
Rod = RodQuad