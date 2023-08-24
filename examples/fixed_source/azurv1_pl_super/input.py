import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Set materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Set surfaces
# s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
# s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")
s1 = mcdc.surface("plane-x", x=-20.5, bc="vacuum")
s2 = mcdc.surface("plane-x", x=20.5, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# iQMC Paramters
# =============================================================================
N = 10000
Nx = 50
Nt = 10
maxit = 100
tol = 1e-3
x = np.linspace(-20.5, 20.5, Nx+1)
t = np.linspace(0.0, 20.0, Nt+1)
generator = "halton"
solver = "gmres"
tilt = 0

fixed_source = np.zeros((Nt,Nx))
fixed_source[0, int(Nx/2)] = 1.0
phi0 = np.ones((Nt, Nx))

mcdc.iQMC(
    x=x,
    t=t,
    fixed_source=fixed_source,
    phi0=phi0,
    maxitt=maxit,
    tol=tol,
    generator=generator,
    fixed_source_solver=solver,
    source_tilt=tilt,
)


# =============================================================================
# MC Parameters 
# =============================================================================
# Isotropic pulse at x=t=0

# mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True)

# # Tally: cell-average, cell-edge, and time-edge scalar fluxes
# mcdc.tally(
#     scores=["flux", "flux-x", "flux-t"],
#     x=np.linspace(-20.5, 20.5, 202),
#     t=np.linspace(0.0, 20.0, 21),
# )

# =============================================================================
# MCDC Settings
# =============================================================================
# Setting
mcdc.setting(N_particle=N)

# Run
mcdc.run()
