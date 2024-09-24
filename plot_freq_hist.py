import taichi as ti
import numpy as np


from geometry_rod import Rod
import matplotlib.pyplot as plt

from IIR import alpha, beta

def omega_underdamped(alpha, beta):
    delta = np.sqrt(1 - alpha * beta)
    o1, o2 = (1 - delta) / beta, (1 + delta) / beta
    return o1, o2
if __name__ == "__main__":
    ti.init()
    rod= Rod()
    n_bins = 100
    K = rod.get_K()
    lam, Q = rod.eigs()

    max_freq = np.max(np.sqrt(lam))

    o1, o2 = omega_underdamped(alpha, beta)
    lam_to_plot = lam[(o1 < lam) & ( lam < o2)]
    print(o1, o2)
    lams = np.linspace(0, max_freq, n_bins)
    lams_sqrt = np.sqrt(lam_to_plot)

    plt.hist(lams_sqrt, bins = n_bins)
    plt.show()