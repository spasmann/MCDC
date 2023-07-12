import numpy as np
import h5py
import mcdc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def pi_test():
    # =========================================================================
    # Set model
    # =========================================================================
    # Based on Kornreich, ANE 2004, 31, 1477-1494,
    # DOI: 10.1016/j.anucene.2004.03.012

    # Set materials
    m1 = mcdc.material(
        capture=np.array([0.0]),
        scatter=np.array([[0.9]]),
        fission=np.array([0.1]),
        nu_p=np.array([6.0]),
    )
    m2 = mcdc.material(
        capture=np.array([0.68]),
        scatter=np.array([[0.2]]),
        fission=np.array([0.12]),
        nu_p=np.array([2.5]),
    )

    # Set surfaces
    s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
    s2 = mcdc.surface("plane-x", x=1.5)
    s3 = mcdc.surface("plane-x", x=2.5, bc="vacuum")

    # Set cells
    mcdc.cell([+s1, -s2], m1)
    mcdc.cell([+s2, -s3], m2)

    # =========================================================================
    # iQMC Parameters
    # =========================================================================
    N = 10
    maxit = 5
    tol = 1e-3
    x = np.arange(0.0, 2.6, 0.1)
    Nx = len(x) - 1
    generator = "halton"
    solver = "power_iteration"
    fixed_source = np.zeros(Nx)
    phi0 = np.ones((Nx))

    # =========================================================================
    # Set tally, setting, and run mcdc
    # =========================================================================

    mcdc.iQMC(
        x=x,
        fixed_source=fixed_source,
        phi0=phi0,
        maxitt=maxit,
        tol=tol,
        generator=generator,
        eigenmode_solver=solver,
    )
    # Setting
    mcdc.setting(N_particle=N, output="pi_output")
    mcdc.eigenmode()

    # Run
    mcdc.run()

    # =========================================================================
    # Check output
    # =========================================================================
    output = h5py.File("pi_output.h5", "r")
    answer = h5py.File("pi_answer.h5", "r")

    a = answer["tally/iqmc_flux"][:]
    b = output["iqmc/flux"][:]
    assert np.allclose(a, b)

    a = output["k_eff"][()]
    b = answer["k_eff"][()]
    assert np.allclose(a, b)

    output.close()
    answer.close()


def davidson_test():
    # =========================================================================
    # Set model
    # =========================================================================
    # Based on Kornreich, ANE 2004, 31, 1477-1494,
    # DOI: 10.1016/j.anucene.2004.03.012

    # Set materials
    m1 = mcdc.material(
        capture=np.array([0.0]),
        scatter=np.array([[0.9]]),
        fission=np.array([0.1]),
        nu_p=np.array([6.0]),
    )
    m2 = mcdc.material(
        capture=np.array([0.68]),
        scatter=np.array([[0.2]]),
        fission=np.array([0.12]),
        nu_p=np.array([2.5]),
    )

    # Set surfaces
    s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
    s2 = mcdc.surface("plane-x", x=1.5)
    s3 = mcdc.surface("plane-x", x=2.5, bc="vacuum")

    # Set cells
    mcdc.cell([+s1, -s2], m1)
    mcdc.cell([+s2, -s3], m2)

    # =========================================================================
    # iQMC Parameters
    # =========================================================================
    N = 10
    maxit = 5
    tol = 1e-3
    x = np.arange(0.0, 2.6, 0.1)
    Nx = len(x) - 1
    generator = "halton"
    solver = "davidson"
    fixed_source = np.zeros(Nx)
    phi0 = np.ones((Nx))

    # =========================================================================
    # Set tally, setting, and run mcdc
    # =========================================================================

    mcdc.iQMC(
        x=x,
        fixed_source=fixed_source,
        phi0=phi0,
        maxitt=maxit,
        tol=tol,
        generator=generator,
        eigenmode_solver=solver,
    )
    # Setting
    mcdc.setting(N_particle=N, output="davidson_output")
    mcdc.eigenmode()

    # Run
    mcdc.run()

    # =========================================================================
    # Check output
    # =========================================================================

    output = h5py.File("davidson_output.h5", "r")
    answer = h5py.File("davidson_answer.h5", "r")

    a = answer["iqmc/flux"][:]
    b = output["iqmc/flux"][:]
    assert np.allclose(a, b)

    output.close()
    answer.close()


if __name__ == "__main__":
    pi_test()
    davidson_test()
