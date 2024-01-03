import matplotlib.pyplot as plt
import h5py
import numpy as np

# Load iqmc result
with h5py.File("output.h5", "r") as f:
    meshx = f["iqmc/grid/x"][:]
    meshy = f["iqmc/grid/y"][:]
    dx = [meshx[1:] - meshx[:-1]][-1]
    x_mid = 0.5 * (meshx[:-1] + meshx[1:])
    phi = f["iqmc/tally/flux"][:]

    f.close()

X, Y = np.meshgrid(x_mid, x_mid)

# =============================================================================
# DD Comparison
# =============================================================================

with h5py.File("dd_output.h5", "r") as f:
    dd_phi = f["iqmc/tally/flux"][:]
    f.close()

diff = abs(phi - dd_phi) / phi * 100

plt.figure(dpi=300, figsize=(8, 4))
plt.pcolormesh(X, Y, diff, shading="nearest")
plt.colorbar().set_label(r"DD Flux Comparison", rotation=270, labelpad=15)
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel(r"$x$ [cm]")
plt.ylabel(r"$y$ [cm]")
plt.title(r"Relative Difference")
plt.show()
plt.tight_layout()


# =============================================================================
# Flux Plot
# =============================================================================
Z = np.log10(np.abs(phi / phi.min()))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, Z, edgecolor="b", color="white", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"log($\phi$)", rotation=180)

ax.view_init(elev=15, azim=20)
plt.show()

Z = np.log10(np.abs(dd_phi / dd_phi.min()))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(12, 10))
ax.plot_surface(Y, X, Z, edgecolor="b", color="white", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"log($\phi$)", rotation=180)

ax.view_init(elev=15, azim=20)
plt.show()
