import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

# =============================================================================
# Reference solution
# =============================================================================

x_exact = np.array(
    [
        0.0,
        0.15,
        0.3,
        0.45,
        0.6,
        0.75,
        0.9,
        1.05,
        1.2,
        1.35,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
    ]
)
dx = x_exact[1:] - x_exact[:-1]
k_exact = 1.28657
phi_exact = np.array(
    [
        1,
        1.417721,
        1.698988,
        1.903163,
        2.03435,
        2.092069,
        2.075541,
        1.984535,
        1.818753,
        1.574144,
        1.199995,
        0.9532296,
        0.7980474,
        0.6788441,
        0.5823852,
        0.5020479,
        0.4337639,
        0.3747058,
        0.3226636,
        0.2755115,
        0.228371,
    ]
)
tmp = 0.5 * (phi_exact[1:] + phi_exact[:-1])
norm = np.sum(tmp * dx)
phi_exact /= norm

# =============================================================================
# Begin Plot
# =============================================================================
plt.figure(dpi=300, figsize=(8, 5))
plt.plot(x_exact, phi_exact, label='sol')

# =============================================================================
# MCDC results
# =============================================================================
# Results
with h5py.File("mc_output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1:] - x[:-1]
    phi_avg = f["tally/flux-x/mean"][:]
    phi_sd = f["tally/flux-x/sdev"][:]
    k = f["k_cycle"][:]
    k_avg = f["k_mean"][()]
    k_sd = f["k_sdev"][()]
    rg = f["gyration_radius"][:]
    f.close()
# Note the spatial (dx) and source strength (100+1) normalization
tmp = 0.5 * (phi_avg[1:] + phi_avg[:-1])
norm = np.sum(tmp * dx)
phi_avg /= norm


plt.plot(x, phi_avg, label='MC')


# =============================================================================
# Power Iteration Results
# =============================================================================

with h5py.File("PI_output.h5", "r") as f:
    # Note the spatial (dx) and source strength (100+1) normalization
    phi_avg = f["tally/iqmc_flux"][:]
    x = f["iqmc/grid/x"][:]
    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    f.close()

tmp = 0.5 * (phi_avg[1:] + phi_avg[:-1])
norm = np.sum(tmp * dx)
phi_avg /= norm

plt.plot(x_mid, phi_avg, label='PI')

# =============================================================================
# Davidson Results
# =============================================================================

with h5py.File("davidson_output.h5", "r") as f:
    # Note the spatial (dx) and source strength (100+1) normalization
    phi_avg = f["tally/iqmc_flux"][:]
    x = f["iqmc/grid/x"][:]
    dx = x[1] - x[0]
    x_mid = 0.5 * (x[:-1] + x[1:])
    f.close()

tmp = 0.5 * (phi_avg[1:] + phi_avg[:-1])
norm = np.sum(tmp * dx)
phi_avg /= norm
    
plt.plot(x_mid, phi_avg, label='davidson')


# =============================================================================
# Finish Plot
# =============================================================================
plt.title('Kornreich et al. Slab')
plt.ylabel(r"$\phi(x)$")
plt.xlabel(r"$x$")
plt.grid()
plt.legend()

