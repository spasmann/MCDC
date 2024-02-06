import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Based on Sood, PNE, Volume 42, Issue 1, 2003, Pages 55-106 2003,
# "Analytical Benchmark Test Set For Criticality Code Verification"

# Set materials
# 2G-U Slab data
m1 = mcdc.material(
    capture=np.array([0.01344, 0.00384]),
    scatter=np.array([[0.26304, 0.0720], [0.00000, 0.078240]]),
    fission=np.array([0.06912, 0.06192]),
    nu_p=np.array([2.5, 2.7]),
    chi_p=np.array([[0.425, 0.425], [0.575, 0.575]]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=6.01275, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m1)

# =============================================================================
# iQMC Parameters
# =============================================================================
Nx = 10
N = 1000
tol = 1e-6
x = np.linspace(0.0, 6.01275, num=Nx + 1)
generator = "halton"
solver = "batch"
fixed_source = np.zeros((2, Nx))
phi0 = np.ones((2, Nx))

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.iQMC(
    x=x,
    g=np.ones(2),
    phi0=phi0,
    fixed_source=fixed_source,
    tol=tol,
    generator=generator,
    maxitt=20,
    eigenmode_solver=solver,
)

# Setting
mcdc.setting(N_particle=N)
mcdc.eigenmode(N_inactive=5, N_active=15)

# Run
mcdc.run()
