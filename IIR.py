
import taichi as ti
import numpy as np
# ti.init(arch = ti.x64, default_fp=ti.f32)
alpha = 6
beta = 1e-7
dt = 1e-3

def xi_np(omega):
    return 0.5 * (alpha / omega + beta * omega)
    
def odi_np(omega, xii):
    '''
    omega_di, damped natural frequency
    '''
    return omega * np.sqrt(1 - xii ** 2.0)

def IIR_np(omega):
    xii = xi_np(omega)
    a1, a2, b = 0, 0, 0
    # if 0 < xii and xii < 1:
    xii = np.clip(xii, 0, 1 - 1e-4)
    od = odi_np(omega, xii)
    epsi = np.exp(-xii * omega * dt)
    theta = od * dt
    gamma = np.arcsin(xii)
    a1 = 2 * epsi * np.cos(theta)
    a2 = - epsi ** 2
    b = 2.0 / 3.0 * (epsi * np.cos(theta + gamma) - epsi ** 2 * np.cos(2 * theta + gamma)) / (omega * od)
    return a1, a2, b

@ti.func
def IIR(omega):
    xii = xi(omega)
    a1, a2, b = 0, 0, 0
    if 0 < xii and xii < 1:
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
    for i in range(6, lam.shape[0]):
        omega = ti.sqrt(lam[i])
        a1, a2, b = IIR(omega)
        Qi = excitement[i]
        q_new = a1 * q1[i] + a2 * q0[i] + b * Qi
        q0[i] = q_new

def step_np(q0, q1, excitement, lam):
    omega = np.sqrt(lam)
    a1, a2, b = IIR_np(omega)
    print("a1, a2, b = ", a1, a2, b)
    Qi = excitement
    q_new = a1 * q1 + a2 * q0 + b * Qi
    q0[:] = q_new

        


    
