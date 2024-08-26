
import taichi as ti
# ti.init(arch = ti.x64, default_fp=ti.f32)
alpha = 6
beta = 1e-7
dt = 1e4

@ti.func
def IIR(omega):
    xii = xi(omega)
    od = odi(omega)
    epsi = ti.exp(-xii * omega * dt)
    theta = od * dt
    gamma = ti.asin(xii)
    a1 = 2 * epsi * ti.cos(theta)
    a2 = - epsi ** 2
    b = 2.0 / 3.0 * (epsi * ti.cos(theta + gamma) - epsi ** 2 * ti.cos(2 * theta + gamma)) / (omega * od)
    return a1, a2, b

@ti.func
def xi(omega):
    return 0.5 * (alpha / omega + beta * omega)

@ti.func
def odi(omega):
    '''
    omega_di, damped natural frequency
    '''
    return omega * ti.sqrt(1 - xi(omega) ** 2.0)

@ti.kernel
def step(q0: ti.types.ndarray(), q1: ti.types.ndarray(), excitement: ti.types.ndarray(), lam: ti.types.ndarray()):
    for i in range(lam.shape[0]):
        omega = ti.sqrt(lam[i])
        a1, a2, b = IIR(omega)
        Qi = excitement[i]
        q_new = a1 * q1[i] + a2 * q0[i] + b * Qi
        q0[i] = q_new

        
        


    
