import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

mat_iron = mcdc.material(
    [
        ["Fe54", 4.82003e21 * 1e-24],
        ["Fe56", 7.6223e22 * 1e-24],
        ["Fe57", 1.82829e21 * 1e-24],
        ["Fe58", 2.32691e20 * 1e-24],
        ["C12", 1.02809e21 * 1e-24],
        ["Mn55", 8.56743e20 * 1e-24],
        ["P31", 5.9972e20 * 1e-24],
        ["S32", 8.56743e19 * 1e-24],
    ]
)
mat_air = mcdc.material(
    [
        ["N14", 4.36848e19 * 1e-24],
        ["O16", 1.02529e19 * 1e-24],
    ]
)

# =============================================================================
# Materials
# =============================================================================

# Surfaces
s0 = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=4.46)
sout = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=1000, bc="vacuum")

# Cells
cr1 = mcdc.cell(-s0, mat_iron)
void1 = mcdc.cell(+s0 & -sout, mat_air)


# =============================================================================
# Source
# =============================================================================

mcdc.source(
    point=[0.0, 0.0, 0.0],
    energy=np.array([[14e6 - 1, 14e6 + 1], [1.0, 1.0]]),
    isotropic=True,
)

# =============================================================================
# Tallies
# =============================================================================
PStally = mcdc.tally.mesh_tally(
    scores=["flux"],
    x=[740.0, 760.0],
    y=[-10.0, 10.0],
    z=[-10.0, 10.0],
    t=np.linspace(0.0, 500.0e-9, 250),
)

# =============================================================================
# Settings
# =============================================================================

mcdc.setting(N_particle=1e4)
mcdc.implicit_capture()

mcdc.run()