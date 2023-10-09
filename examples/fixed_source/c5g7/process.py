import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
from matplotlib import colors


# =============================================================================
# Plot results
# =============================================================================

# Results
with h5py.File("output.h5", "r") as f:
    phi_avg = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]
    x = f["tally/grid/x"][:]
    y = f["tally/grid/y"][:]
    z = f["tally/grid/z"][:]

dx = x[1] - x[0]
dz = z[1] - z[0]
dV = dx * dx * dz

phi_avg /= dV
phi_sd /= dV

phi_fast = phi_avg[0]
phi_thermal = phi_avg[1]
phi_fast_sd = phi_sd[0]
phi_thermal_sd = phi_sd[1]

# X-Y plane
x_mid = 0.5 * (x[1:] + x[:-1])
Y, X = np.meshgrid(x_mid, x_mid)
phi_fast_xy = np.sum(phi_fast, axis=2)
phi_thermal_xy = np.sum(phi_thermal, axis=2)
phi_fast_xy_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=2)) / phi_fast_xy
phi_thermal_xy_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=2)) / phi_thermal_xy

plt.pcolormesh(X, Y, phi_fast_xy, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast neutron flux")
plt.show()

plt.pcolormesh(X, Y, phi_fast_xy_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Fast neutron flux stdev")
plt.show()

plt.pcolormesh(X, Y, phi_thermal_xy, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal neutron flux")
plt.show()

plt.pcolormesh(X, Y, phi_thermal_xy_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Thermal neutron flux stdev")
plt.show()

# X-Z plane
z_mid = 0.5 * (z[1:] + z[:-1])
X, Z = np.meshgrid(z_mid, x_mid)
phi_fast_xz = np.sum(phi_fast, axis=1)
phi_thermal_xz = np.sum(phi_thermal, axis=1)
phi_fast_xz_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=1)) / phi_fast_xz
phi_thermal_xz_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=1)) / phi_thermal_xz

plt.pcolormesh(Z, X, phi_fast_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fast neutron flux")
plt.show()

plt.pcolormesh(Z, X, phi_fast_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fast neutron flux stdev")
plt.show()

plt.pcolormesh(Z, X, phi_thermal_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Thermal neutron flux")
plt.show()

plt.pcolormesh(Z, X, phi_thermal_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Thermal neutron flux stdev")
plt.show()

# Y-Z plane
X, Z = np.meshgrid(z_mid, x_mid)
phi_fast_xz = np.sum(phi_fast, axis=0)
phi_thermal_xz = np.sum(phi_thermal, axis=0)
phi_fast_xz_sd = np.sqrt(np.sum(np.square(phi_fast_sd), axis=0)) / phi_fast_xz
phi_thermal_xz_sd = np.sqrt(np.sum(np.square(phi_thermal_sd), axis=0)) / phi_thermal_xz

plt.pcolormesh(Z, X, phi_fast_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$y$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fast neutron flux")
plt.show()

plt.pcolormesh(Z, X, phi_fast_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$y$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Fast neutron flux stdev")
plt.show()

plt.pcolormesh(Z, X, phi_thermal_xz, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$y$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Thermal neutron flux")
plt.show()

plt.pcolormesh(Z, X, phi_thermal_xz_sd, shading="nearest")
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$y$ [cm]")
plt.ylabel(r"$z$ [cm]")
plt.title(r"Thermal neutron flux stdev")
plt.show()
