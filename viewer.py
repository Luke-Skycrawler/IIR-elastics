# import fast_cd_pyb as fcd 
import fc_viewer as fcd
import numpy as np
import taichi as ti 
from geometry_rod import Rod
from IIR import step, dt, alpha, step_np

ti.init(arch=ti.x64, default_fp=ti.f32)

rod = Rod()
mid, V0, F = rod.mid, rod.V0, rod.F
lam, Q = rod.eigs()
lam_vis = lam[6:]
print(f"lam = {lam[6:,]}, xi = {0.5 * alpha / np.sqrt(lam_vis)}")
H = rod.compute_H(Q)
q0 = np.zeros_like(lam)
q1 = np.zeros_like(q0)
n_substeps = 100


V = V0.copy()
n_unknowns =  3 * V0.shape[0]
T0 = np.identity(4).astype( dtype=np.float32, order="F")
spatial_vector = np.zeros((6,), np.float64)
excite = np.zeros((n_unknowns, ), np.float64)
init_guizmo = True
transform = "translate"
vis_cd = True
viewer_only = False

# V0 = np.eye(3)
# mid = np.zeros(3)
# F = np.array([0, 1, 2]).reshape((1, 3)).astype(np.int32)
# V = V0.copy()



def guizmo_callback(A):
    global V, excite, T0
    if viewer_only:
        return
    spatial_vector_new = sv(T0, A)
    excite = H @ (spatial_vector_new - spatial_vector) / n_substeps
    print(f"excite = {excite[6:]}, min = {np.min(excite[6:])}, max = {np.max(excite[6:])}")
    spatial_vector[:] = spatial_vector_new[:]

    T0 = A


def callback_key_pressed(key, modifier):
    global vis_cd, transform, viewer
    if (key == ord('g') or key == ord('G')):
        if (init_guizmo):
            if (transform == "translate"):
                transform = "rotate"
            elif (transform == "rotate"):
                transform = "scale"
            elif (transform == "scale"):
                transform = "translate"

            viewer.change_guizmo_op(transform)
        else:
            print("Guizmo not initialized, pass init_guizmo=True to the viewer constructor")
    if (key == ord('c') or key==ord('C') ):
        vis_cd = not vis_cd
    return False


def pre_draw_callback():
    global q0, q1, Q
    if viewer_only: 
        return 
    # step(q0, q1, excite, lam)
    for i in range(n_substeps):
        step_np(q0[6:], q1[6:], excite[6:], lam[6:])
        q0, q1 = q1, q0

    # V = V0 @ T0[:3, :3].T + mid + T0[: 3, 3].reshape(1, 3)

    # u = Q @ q1
    # u = Q[:, 0] * q1[0]
    Qi = Q[:, 6]
    # u = Qi * q1[6]
    u = Q @ q1
    # u = Qi
    print("q1 max = ", np.max(q1[6:]), "min = ", np.min(q1[6:]))

    print(f"q1 = {q1[6:]}, Q = {excite[6:]}, u = {u}")
    # u = np.zeros_like(q1)
    rod.update_pos(u, T0)
    V = rod.xcs.to_numpy()

    viewer.set_mesh(V, F, 0)

def compute_angular_velocity(R0, R1, dt):
    # Compute the relative rotation matrix
    if (R0 == R1).all():
        return np.zeros(3)

    R = R0 @ R1.T
    
    # Compute the angle of rotation
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    # Compute the rotation axis
    omega_skew = (R - R.T) / (2 * np.sin(theta))
    n = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
    
    # Compute the angular velocity
    angular_velocity = (theta / dt) * n
    return angular_velocity

def sv(T1, T0):
    v = (T1[:3, 3] - T0[:3, 3]) / dt
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]
    omega = compute_angular_velocity(R0, R1, dt)
    _sv = np.zeros(6)
    _sv[:3] = omega
    _sv[3:] = v
    return _sv

viewer = fcd.fast_cd_viewer()

if init_guizmo:
    viewer.init_guizmo(True, T0, guizmo_callback, transform)
viewer.set_pre_draw_callback(pre_draw_callback)
viewer.set_key_callback(callback_key_pressed)


viewer.set_mesh(V0, F, 0)
viewer.launch()




