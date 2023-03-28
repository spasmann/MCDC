import numpy as np

from numba import njit, objmode, jit
from scipy.linalg import eig

import mcdc.kernel as kernel

from mcdc.constant import *
from mcdc.print_ import (
    print_progress,
    print_progress_eigenvalue,
    print_progress_iqmc,
    print_iqmc_eigenvalue_progress,
    print_iqmc_eigenvalue_exit_code,
)


# =========================================================================
# Main loop
# =========================================================================


@njit
def loop_main(mcdc):
    simulation_end = False
    while not simulation_end:
        # Loop over source particles
        loop_source(mcdc)

        # Eigenvalue cycle closeout
        if mcdc["setting"]["mode_eigenvalue"]:
            # Tally history closeout
            kernel.global_tally_closeout_history(mcdc)
            if mcdc["cycle_active"]:
                kernel.tally_closeout_history(mcdc)

            # Print progress
            with objmode():
                print_progress_eigenvalue(mcdc)

            # Manage particle banks
            kernel.manage_particle_banks(mcdc)

            # Cycle management
            mcdc["i_cycle"] += 1
            if mcdc["i_cycle"] == mcdc["setting"]["N_cycle"]:
                simulation_end = True
            elif mcdc["i_cycle"] >= mcdc["setting"]["N_inactive"]:
                mcdc["cycle_active"] = True

        # Time census closeout
        elif (
            mcdc["technique"]["time_census"]
            and mcdc["technique"]["census_idx"]
            < len(mcdc["technique"]["census_time"]) - 1
        ):
            # Manage particle banks
            kernel.manage_particle_banks(mcdc)

            # Increment census index
            mcdc["technique"]["census_idx"] += 1

        # Fixed-source closeout
        else:
            simulation_end = True

    # Tally closeout
    kernel.tally_closeout(mcdc)


# =============================================================================
# Source loop
# =============================================================================


@njit
def loop_source(mcdc):
    # Rebase rng skip_ahead seed
    kernel.rng_skip_ahead_strides(mcdc["mpi_work_start"], mcdc)
    kernel.rng_rebase(mcdc)

    # Progress bar indicator
    N_prog = 0

    # Loop over particle sources
    for work_idx in range(mcdc["mpi_work_size"]):
        # Particle tracker
        if mcdc["setting"]["track_particle"]:
            mcdc["particle_track_history_ID"] += 1

        # Initialize RNG wrt work index
        kernel.rng_skip_ahead_strides(work_idx, mcdc)

        # =====================================================================
        # Get a source particle and put into active bank
        # =====================================================================

        # Get from fixed-source?
        if mcdc["bank_source"]["size"] == 0:
            # Sample source
            xi = kernel.rng(mcdc)
            tot = 0.0
            for S in mcdc["sources"]:
                tot += S["prob"]
                if tot >= xi:
                    break
            P = kernel.source_particle(S, mcdc)

        # Get from source bank
        else:
            P = mcdc["bank_source"]["particles"][work_idx]

        # Check if it is beyond
        census_idx = mcdc["technique"]["census_idx"]
        if P["t"] > mcdc["technique"]["census_time"][census_idx]:
            P["t"] += SHIFT
            kernel.add_particle(P, mcdc["bank_census"])
        else:
            # Add the source particle into the active bank
            kernel.add_particle(P, mcdc["bank_active"])

        # =====================================================================
        # Run the source particle and its secondaries
        # =====================================================================

        # Loop until active bank is exhausted
        while mcdc["bank_active"]["size"] > 0:
            # Get particle from active bank
            P = kernel.get_particle(mcdc["bank_active"], mcdc)

            # Apply weight window
            if mcdc["technique"]["weight_window"]:
                kernel.weight_window(P, mcdc)

            # Particle tracker
            if mcdc["setting"]["track_particle"]:
                mcdc["particle_track_particle_ID"] += 1

            # Particle loop
            loop_particle(P, mcdc)

        # =====================================================================
        # Closeout
        # =====================================================================

        # Tally history closeout for fixed-source simulation
        if not mcdc["setting"]["mode_eigenvalue"]:
            kernel.tally_closeout_history(mcdc)

        # Progress printout
        percent = (work_idx + 1.0) / mcdc["mpi_work_size"]
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


# =========================================================================
# Particle loop
# =========================================================================


@njit
def loop_particle(P, mcdc):
    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)

    while P["alive"]:
        # Find cell from root universe if unknown
        if P["cell_ID"] == -1:
            trans = np.zeros(3)
            P["cell_ID"] = kernel.get_particle_cell(P, 0, trans, mcdc)

        # Determine and move to event
        kernel.move_to_event(P, mcdc)
        event = P["event"]

        # Collision
        if event == EVENT_COLLISION:
            # Generate IC?
            if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
                kernel.bank_IC(P, mcdc)

            # Branchless collision?
            if mcdc["technique"]["branchless_collision"]:
                kernel.branchless_collision(P, mcdc)

            # Analog collision
            else:
                # Get collision type
                kernel.collision(P, mcdc)
                event = P["event"]

                # Perform collision
                if event == EVENT_CAPTURE:
                    kernel.capture(P, mcdc)
                elif event == EVENT_SCATTERING:
                    kernel.scattering(P, mcdc)
                elif event == EVENT_FISSION:
                    kernel.fission(P, mcdc)

                # Sensitivity quantification for nuclide?
                material = mcdc["materials"][P["material_ID"]]
                if material["sensitivity"] and P["sensitivity_ID"] == 0:
                    kernel.sensitivity_material(P, mcdc)

        # Mesh crossing
        elif event == EVENT_MESH:
            kernel.mesh_crossing(P, mcdc)

        # Surface crossing
        elif event == EVENT_SURFACE:
            kernel.surface_crossing(P, mcdc)

        # Lattice crossing
        elif event == EVENT_LATTICE:
            kernel.shift_particle(P, SHIFT)

        # Time boundary
        elif event == EVENT_TIME_BOUNDARY:
            kernel.mesh_crossing(P, mcdc)
            kernel.time_boundary(P, mcdc)

        # Surface move
        elif event == EVENT_SURFACE_MOVE:
            P["t"] += SHIFT
            P["cell_ID"] = -1

        # Time census
        elif event == EVENT_CENSUS:
            P["t"] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc["bank_census"])
            P["alive"] = False

        # Surface and mesh crossing
        elif event == EVENT_SURFACE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.surface_crossing(P, mcdc)

        # Lattice and mesh crossing
        elif event == EVENT_LATTICE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            kernel.shift_particle(P, SHIFT)

        # Surface move and mesh crossing
        elif event == EVENT_SURFACE_MOVE_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            P["t"] += SHIFT
            P["cell_ID"] = -1

        # Time census and mesh crossing
        elif event == EVENT_CENSUS_N_MESH:
            kernel.mesh_crossing(P, mcdc)
            P["t"] += SHIFT
            kernel.add_particle(kernel.copy_particle(P), mcdc["bank_census"])
            P["alive"] = False

        # Apply weight window
        if mcdc["technique"]["weight_window"]:
            kernel.weight_window(P, mcdc)

        # Apply weight roulette
        if mcdc["technique"]["weight_roulette"]:
            # check if weight has fallen below threshold
            if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
                kernel.weight_roulette(P, mcdc)

    # Particle tracker
    if mcdc["setting"]["track_particle"]:
        kernel.track_particle(P, mcdc)


# =============================================================================
# iQMC Loop
# =============================================================================


@njit
def loop_iqmc(mcdc):
    # generate material index
    kernel.generate_iqmc_material_idx(mcdc)
    # function calls from specified solvers
    if mcdc["setting"]["mode_eigenvalue"]:
        if mcdc["technique"]["iqmc_eigenmode_solver"] == "davidson":
            davidson(mcdc)
        if mcdc["technique"]["iqmc_eigenmode_solver"] == "power_iteration":
            power_iteration(mcdc)
    else:
        source_iteration(mcdc)


@njit
def source_iteration(mcdc):
    simulation_end = False

    while not simulation_end:
        # reset particle bank size
        mcdc["bank_source"]["size"] = 0
        mcdc["technique"]["iqmc_source"] = np.zeros_like(
            mcdc["technique"]["iqmc_source"]
        )

        # set bank source
        kernel.prepare_qmc_source(mcdc)
        # initialize particles with LDS
        kernel.prepare_qmc_particles(mcdc)

        # prepare source for next iteration
        mcdc["technique"]["iqmc_flux"] = np.zeros_like(mcdc["technique"]["iqmc_flux"])

        # sweep particles
        loop_source(mcdc)
        # sum resultant flux on all processors
        kernel.iqmc_distribute_flux(mcdc)
        mcdc["technique"]["iqmc_itt"] += 1

        # calculate norm of flux iterations
        mcdc["technique"]["iqmc_res"] = kernel.qmc_res(
            mcdc["technique"]["iqmc_flux"], mcdc["technique"]["iqmc_flux_old"]
        )

        # iQMC convergence criteria
        if (mcdc["technique"]["iqmc_itt"] == mcdc["technique"]["iqmc_maxitt"]) or (
            mcdc["technique"]["iqmc_res"] <= mcdc["technique"]["iqmc_tol"]
        ):
            simulation_end = True

        # Print progres
        if not mcdc["setting"]["mode_eigenvalue"]:
            print_progress_iqmc(mcdc)

        # set flux_old = current flux
        mcdc["technique"]["iqmc_flux_old"] = mcdc["technique"]["iqmc_flux"].copy()


@njit
def power_iteration(mcdc):
    simulation_end = False

    # iteration tolerance
    tol = mcdc["technique"]["iqmc_tol"]
    # maximum number of iterations
    maxit = mcdc["technique"]["iqmc_maxitt"]
    mcdc["technique"]["iqmc_flux_outter"] = mcdc["technique"]["iqmc_flux"].copy()

    # assign function call from specified solvers
    # inner_iteration = globals()[mcdc["technique"]["fixed_source_solver"]]

    while not simulation_end:
        # iterate over scattering source
        source_iteration(mcdc)
        # reset counter for inner iteration
        mcdc["technique"]["iqmc_itt"] = 0

        # update k_eff
        kernel.UpdateK(
            mcdc["k_eff"],
            mcdc["technique"]["iqmc_flux_outter"],
            mcdc["technique"]["iqmc_flux"],
            mcdc,
        )

        # calculate diff in flux
        mcdc["technique"]["iqmc_res_outter"] = kernel.qmc_res(
            mcdc["technique"]["iqmc_flux"], mcdc["technique"]["iqmc_flux_outter"]
        )
        mcdc["technique"]["iqmc_flux_outter"] = mcdc["technique"]["iqmc_flux"].copy()
        mcdc["technique"]["iqmc_itt_outter"] += 1

        print_iqmc_eigenvalue_progress(mcdc)

        # iQMC convergence criteria
        if (mcdc["technique"]["iqmc_itt_outter"] == maxit) or (
            mcdc["technique"]["iqmc_res_outter"] <= tol
        ):
            simulation_end = True

    print_iqmc_eigenvalue_exit_code(mcdc)


@njit
def davidson(mcdc):
    # TODO: find a cleaner way to make all matrices contiguous arrays for
    # dot products

    # Davidson parameters
    simulation_end = False
    maxit = mcdc["technique"]["iqmc_maxitt"]
    tol = mcdc["technique"]["iqmc_tol"]
    # num_sweeps: number of preconditioner sweeps
    num_sweeps = mcdc["technique"]["iqmc_preconditioner_sweeps"]
    # m : restart parameter
    m = mcdc["technique"]["iqmc_krylov_restart"]

    # initial size of Krylov subspace
    Vsize = 1
    # l : number of eigenvalues to solve for
    l = 1

    phi0 = mcdc["technique"]["iqmc_flux"].copy()
    Nt = phi0.size
    phi0 = np.reshape(phi0, (Nt,))

    # orthonormalize initial guess
    V0 = phi0 / np.linalg.norm(phi0)
    # we allocate matrices then use slice indexing in loop
    V = np.zeros((Nt, maxit), dtype=np.float64)
    axv = np.zeros((Nt, maxit), dtype=np.float64)
    bxv = np.zeros((Nt, maxit), dtype=np.float64)

    V[:, 0] = V0

    if m is None:
        # unless specified there is no restart parameter
        m = maxit + 1
    V[:, :Vsize] = kernel.preconditioner(V[:, :Vsize][:, -1], mcdc, num_sweeps)
    # Davidson Routine
    while not simulation_end:
        # Calculate V*A*V (AxV is scattering linear operator function)
        axv[:, Vsize - 1] = kernel.AxV(V[:, :Vsize][:, -1], mcdc)[:, 0]
        AV = np.dot(
            np.ascontiguousarray(V[:, :Vsize].T), np.ascontiguousarray(axv[:, :Vsize])
        )
        # Calculate V*B*V (BxV is fission linear operator function)
        bxv[:, Vsize - 1] = kernel.BxV(V[:, :Vsize][:, -1], mcdc)[:, 0]
        BV = np.dot(
            np.ascontiguousarray(V[:, :Vsize].T), np.ascontiguousarray(bxv[:, :Vsize])
        )
        # solve for eigenvalues and vectors
        with objmode(Lambda="complex128[:]", w="complex128[:,:]"):
            Lambda, w = eig(AV, b=BV)
            Lambda = np.array(Lambda, dtype=np.complex128)
            w = np.array(w, dtype=np.complex128)

        # there can't be any imaginary eigenvalues
        assert Lambda.imag.all() == 0.0
        Lambda = Lambda.real
        w = w.real
        # get indices of eigenvalues from smallest to largest
        idx = Lambda.argsort()
        # sort eigenvalues from smalles to largest
        Lambda = Lambda[idx]
        # take the l largest eigenvalues
        Lambda = Lambda[:l]
        # assign keff
        mcdc["k_eff"] = 1.0 / Lambda[0]
        # sort corresponding eigenvector (oriented by column)
        w = w[:, idx]
        # take the l largest eigenvectors
        w = w[:, :l]
        # Ritz vectors
        # u = np.dot(np.ascontiguousarray(V[:, :Vsize]), np.ascontiguousarray(w))
        u = V[:, Vsize - 1] * w[-1, 0]
        # residual
        res = kernel.AxV(u, mcdc) - Lambda * kernel.BxV(u, mcdc)
        mcdc["technique"]["iqmc_res_outter"] = np.linalg.norm(res, ord=2)
        mcdc["technique"]["iqmc_itt_outter"] += 1
        print_iqmc_eigenvalue_progress(mcdc)
        # check convergence criteria
        if (mcdc["technique"]["iqmc_itt_outter"] == maxit) or (
            mcdc["technique"]["iqmc_res_outter"] <= tol
        ):
            simulation_end = True
            break
        else:
            # Precondition for next iteration
            t = kernel.preconditioner(res, mcdc, num_sweeps)
            # check restart condition
            if Vsize < m - l:
                Vsize += 1
                # appends new orthogonalization to V
                V[:, :Vsize] = kernel.modified_gram_schmidt(V[:, : Vsize - 1], t)
                # V[:,Vsize-1] = t[:,0]
                # V[:,:Vsize],R = np.linalg.qr(V[:,:Vsize])
            # else:
            #     # "restarts" by appending to a new array
            #     Vsize = 2
            #     V[:, :Vsize] = kernel.modified_gram_schmidt(u, t)

    print_iqmc_eigenvalue_exit_code(mcdc)

    # normalize and save final scalar flux
    flux = np.reshape(
        V[:, 0] / np.linalg.norm(V[:, 0]), mcdc["technique"]["iqmc_flux"].shape
    )
    mcdc["technique"]["iqmc_flux"] = flux
