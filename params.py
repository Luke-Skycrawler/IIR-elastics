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
