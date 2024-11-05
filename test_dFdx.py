import taichi as ti
import numpy as np
n_elements = 1
h = 1e-4
ti.init()
x = ti.Vector.field(3, dtype = ti.f64, shape = (n_elements, 4))
Dm = ti.Matrix.field(3, 3, dtype = ti.f64, shape = (n_elements)) 
F = ti.Matrix.field(3, 3, dtype = ti.f64, shape = (n_elements)) 

def computePFpx(DmInv):
    m = DmInv[0, 0]
    n = DmInv[0, 1]
    o = DmInv[0, 2]
    p = DmInv[1, 0]
    q = DmInv[1, 1]
    r = DmInv[1, 2]
    s = DmInv[2, 0]
    t = DmInv[2, 1]
    u = DmInv[2, 2]

    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u

    PFPu = np.zeros((9, 12))
    PFPu[0, 0]  = t1
    PFPu[0, 3]  = m
    PFPu[0, 6]  = p
    PFPu[0, 9]  = s
    PFPu[1, 1]  = t1
    PFPu[1, 4]  = m
    PFPu[1, 7]  = p
    PFPu[1, 10] = s
    PFPu[2, 2]  = t1
    PFPu[2, 5]  = m
    PFPu[2, 8]  = p
    PFPu[2, 11] = s
    PFPu[3, 0]  = t2
    PFPu[3, 3]  = n
    PFPu[3, 6]  = q
    PFPu[3, 9]  = t
    PFPu[4, 1]  = t2
    PFPu[4, 4]  = n
    PFPu[4, 7]  = q
    PFPu[4, 10] = t
    PFPu[5, 2]  = t2
    PFPu[5, 5]  = n
    PFPu[5, 8]  = q
    PFPu[5, 11] = t
    PFPu[6, 0]  = t3
    PFPu[6, 3]  = o
    PFPu[6, 6]  = r
    PFPu[6, 9]  = u
    PFPu[7, 1]  = t3
    PFPu[7, 4]  = o
    PFPu[7, 7]  = r
    PFPu[7, 10] = u
    PFPu[8, 2]  = t3
    PFPu[8, 5]  = o
    PFPu[8, 8]  = r
    PFPu[8, 11] = u

    return PFPu

def compute_dFdx(Dm):

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

@ti.kernel
def compute_Dm():
    for i in range(n_elements):
        x0 = x[i, 0]
        x1 = x[i, 1]
        x2 = x[i, 2]
        x3 = x[i, 3]

        _Dm = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        inv_Dm = _Dm.inverse()
        Dm[i] = inv_Dm

@ti.kernel
def compute_F():
    for i in range(n_elements): 
        x0 = x[i, 0]
        x1 = x[i, 1]
        x2 = x[i, 2]
        x3 = x[i, 3]

        Ds = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
        F[i] = Ds @ Dm[i]

x.from_numpy(np.random.rand(1, 4, 3))
compute_Dm()
compute_F()
Dm_np = Dm.to_numpy()
F_np = F.to_numpy()
dFdx1 = compute_dFdx(Dm_np[0])
dFdx = computePFpx(Dm_np[0])

print(f"F = {F_np[0]}")
dx = np.random.rand(12)
xnp = x.to_numpy()
xdx = xnp[0] + dx.reshape((4, 3)) * h
# xdx = xnp[0]
# xdx[0] += dx[:3] * h
# xdx[1] += dx[3:6] * h
# xdx[2] += dx[6:9] * h

xdx = xdx.reshape((1, 4, 3))

x.from_numpy(xdx)   
# compute_Dm()
compute_F()
Fdx = F.to_numpy()
dFdx_fd = (Fdx[0] - F_np[0]) / h

dFdx_analytical = dFdx @ dx

print(f"Analytical: {dFdx_analytical}, FD: {dFdx_fd}")

# print(f"analytical1 - {dFdx1 @ dx}")