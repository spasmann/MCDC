import sys
import argparse
import numba as nb
from mcdc.constant import SHIFT
from mcdc.type_ import iqmc_score_list

# Parse command-line arguments
#   TODO: Will be inside run() once Python/Numba adapter is integrated
parser = argparse.ArgumentParser(description="MC/DC: Monte Carlo Dynamic Code")
parser.add_argument(
    "--mode", type=str, help="Run mode", choices=["python", "numba"], default="python"
)
parser.add_argument("--N_particle", type=int, help="Number of particles")
parser.add_argument("--output", type=str, help="Output file name")
parser.add_argument("--progress_bar", default=False, action="store_true")
parser.add_argument("--no-progress_bar", dest="progress_bar", action="store_false")
args, unargs = parser.parse_known_args()

# Set mode
#   TODO: Will be inside run() once Python/Numba adapter is integrated
mode = args.mode
if mode == "python":
    nb.config.DISABLE_JIT = True
elif mode == "numba":
    nb.config.DISABLE_JIT = False

import h5py
import numpy as np

from mpi4py import MPI
from scipy.stats import qmc

import mcdc.kernel as kernel
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.loop import loop_fixed_source, loop_eigenvalue, loop_iqmc
from mcdc.print_ import (
    print_banner,
    print_msg,
    print_runtime,
    print_header_eigenvalue,
    print_error,
)

# Get input_deck
import mcdc.global_ as mcdc_

input_deck = mcdc_.input_deck


def run():
    # Override input deck with command-line argument, if given
    if args.N_particle is not None:
        input_deck.setting["N_particle"] = args.N_particle
    if args.output is not None:
        input_deck.setting["output_name"] = args.output
    if args.progress_bar is not None:
        input_deck.setting["progress_bar"] = args.progress_bar

    # Start timer
    total_start = MPI.Wtime()

    # Preparation
    #   Set up and get the global variable container `mcdc` based on
    #   input deck
    preparation_start = MPI.Wtime()
    mcdc = prepare()
    mcdc["runtime_preparation"] = MPI.Wtime() - preparation_start

    # Print banner, hardware configuration, and header
    print_banner(mcdc)
    print_msg(" Now running TNT...")
    if mcdc["setting"]["mode_eigenvalue"]:
        print_header_eigenvalue(mcdc)

    # Run simulation
    simulation_start = MPI.Wtime()
    if mcdc["technique"]["iQMC"]:
        loop_iqmc(mcdc)
    elif mcdc["setting"]["mode_eigenvalue"]:
        loop_eigenvalue(mcdc)
    else:
        loop_fixed_source(mcdc)
    mcdc["runtime_simulation"] = MPI.Wtime() - simulation_start

    # Output: generate hdf5 output files
    output_start = MPI.Wtime()
    generate_hdf5(mcdc)
    mcdc["runtime_output"] = MPI.Wtime() - output_start

    # Stop timer
    MPI.COMM_WORLD.Barrier()
    mcdc["runtime_total"] = MPI.Wtime() - total_start

    # Closout
    closeout(mcdc)


# =============================================================================
# prepare domain decomposition
# =============================================================================
def get_d_idx(i, j, k, ni, nj):
    N = i + j * ni + k * ni * nj
    return N


def get_indexes(N, nx, ny):
    k = int(N / (nx * ny))
    j = int((N - nx * ny * k) / nx)
    i = int(N - nx * ny * k - nx * j)
    return i, j, k


def get_neighbors(N, nx, ny, nz):
    i, j, k = get_indexes(N, nx, ny)
    if i > 0:
        xn = get_d_idx(i - 1, j, k, nx, ny)
    else:
        xn = None
    if i < (nx - 1):
        xp = get_d_idx(i + 1, j, k, nx, ny)
    else:
        xp = None
    if j > 0:
        yn = get_d_idx(i, j - 1, k, nx, ny)
    else:
        yn = None
    if j < (ny - 1):
        yp = get_d_idx(i, j + 1, k, nx, ny)
    else:
        yp = None
    if k > 0:
        zn = get_d_idx(i, j, k - 1, nx, ny)
    else:
        zn = None
    if k < (nz - 1):
        zp = get_d_idx(i, j, k + 1, nx, ny)
    else:
        zp = None
    return xn, xp, yn, yp, zn, zp


def get_domain_array(array, mesh, domain_mesh, d_id):
    """

    This function finds the overlap between a "global" mesh passed in,
    like a tally mesh or iqmc mesh, and the domain mesh then maps that overlap
    to the designated numpy array.

    -SLP
    """
    # number of domains in each dimension
    d_Nx = len(domain_mesh["x"]) - 1
    d_Ny = len(domain_mesh["y"]) - 1
    d_Nz = len(domain_mesh["z"]) - 1
    i, j, k = get_indexes(d_id, d_Nx, d_Ny)

    if d_Nx > 1:
        xo = domain_mesh["x"][i] - SHIFT
        xf = domain_mesh["x"][i + 1] + SHIFT
    else:
        xo = domain_mesh["x"][0]
        xf = domain_mesh["x"][-1]

    if d_Ny > 1:
        yo = domain_mesh["y"][j] - SHIFT
        yf = domain_mesh["y"][j + 1] + SHIFT
    else:
        yo = domain_mesh["y"][0]
        yf = domain_mesh["y"][-1]

    if d_Nz > 1:
        zo = domain_mesh["z"][k] - SHIFT
        zf = domain_mesh["z"][k + 1] + SHIFT
    else:
        zo = domain_mesh["z"][0]
        zf = domain_mesh["z"][-1]

    # where does the card mesh overlap the domain mesh
    idx = np.where((mesh["x"] >= xo) & (mesh["x"] <= xf))[0]
    idy = np.where((mesh["y"] >= yo) & (mesh["y"] <= yf))[0]
    idz = np.where((mesh["z"] >= zo) & (mesh["z"] <= zf))[0]
    array = array[:, :, idx[0] : idx[-1], idy[0] : idy[-1], idz[0] : idz[-1]]

    return array


def get_domain_mesh(mesh, domain_mesh, d_id):
    """

    This function finds the overlap between a "global" mesh passed in,
    like a tally mesh or iqmc mesh, and the domain mesh then
    reduces the global mesh size to just the overlap.

    -SLP
    """
    # number of domains in each dimension
    d_Nx = len(domain_mesh["x"]) - 1
    d_Ny = len(domain_mesh["y"]) - 1
    d_Nz = len(domain_mesh["z"]) - 1
    i, j, k = get_indexes(d_id, d_Nx, d_Ny)

    for dim, num_domain, l in zip(["x", "y", "z"], [d_Nx, d_Ny, d_Nz], [i, j, k]):
        # take the boundaries of the domain
        if num_domain > 1:
            do = domain_mesh[dim][l] - SHIFT
            df = domain_mesh[dim][l + 1] + SHIFT
        else:
            do = mesh[dim][0]
            df = mesh[dim][-1]
        # where does the mesh overlap the domain mesh
        idx = np.where((mesh[dim] >= do) & (mesh[dim] <= df))[0]
        # set the mesh = to the overlap section
        mesh[dim] = mesh[dim][idx[0] : idx[-1] + 1]

    return mesh


def dd_prepare():
    work_ratio = input_deck.technique["work_ratio"]

    d_Nx = input_deck.technique["domain_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["domain_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["domain_mesh"]["z"].size - 1

    if input_deck.technique["exchange_rate"] == None:
        input_deck.technique["exchange_rate"] = 100

    if work_ratio is None:
        work_ratio = np.ones(d_Nx * d_Ny * d_Nz)
        input_deck.technique["work_ratio"] = work_ratio

    if (
        input_deck.technique["domain_decomp"]
        and np.sum(work_ratio) != MPI.COMM_WORLD.Get_size()
    ):
        print_error(
            "Domain work ratio not equal to number of processors, %i != %i "
            % (np.sum(work_ratio), MPI.COMM_WORLD.Get_size())
        )

    if input_deck.technique["domain_decomp"]:
        # Assigning domain index
        i = 0
        rank_info = []
        for n in range(d_Nx * d_Ny * d_Nz):
            ranks = []
            for r in range(int(work_ratio[n])):
                ranks.append(i)
                if MPI.COMM_WORLD.Get_rank() == i:
                    d_idx = n
                i += 1
            rank_info.append(ranks)
        input_deck.technique["d_idx"] = d_idx
        xn, xp, yn, yp, zn, zp = get_neighbors(d_idx, d_Nx, d_Ny, d_Nz)
    else:
        input_deck.technique["d_idx"] = 0
        input_deck.technique["xp_neigh"] = []
        input_deck.technique["xn_neigh"] = []
        input_deck.technique["yp_neigh"] = []
        input_deck.technique["yn_neigh"] = []
        input_deck.technique["zp_neigh"] = []
        input_deck.technique["zn_neigh"] = []
        return

    if xp is not None:
        input_deck.technique["xp_neigh"] = rank_info[xp]
    else:
        input_deck.technique["xp_neigh"] = []
    if xn is not None:
        input_deck.technique["xn_neigh"] = rank_info[xn]
    else:
        input_deck.technique["xn_neigh"] = []

    if yp is not None:
        input_deck.technique["yp_neigh"] = rank_info[yp]
    else:
        input_deck.technique["yp_neigh"] = []
    if yn is not None:
        input_deck.technique["yn_neigh"] = rank_info[yn]
    else:
        input_deck.technique["yn_neigh"] = []

    if zp is not None:
        input_deck.technique["zp_neigh"] = rank_info[zp]
    else:
        input_deck.technique["zp_neigh"] = []
    if zn is not None:
        input_deck.technique["zn_neigh"] = rank_info[zn]
    else:
        input_deck.technique["zn_neigh"] = []


def prepare():
    """
    Preparing the MC transport simulation:
      (1) Process input deck
      (2) Make types
      (3) Set up and return the global variable container `mcdc`
    """

    dd_prepare()

    # =========================================================================
    # Sizes
    #   We need this to determine the maximum size of model objects
    # =========================================================================

    # Neutron and delayed neutron precursor group sizes
    #   We assume that all materials have the same group structures
    G = input_deck.materials[0]["G"]
    J = input_deck.materials[0]["J"]

    # Number of model objects
    N_nuclide = len(input_deck.nuclides)
    N_material = len(input_deck.materials)
    N_surface = len(input_deck.surfaces)
    N_cell = len(input_deck.cells)
    N_universe = len(input_deck.universes)
    N_lattice = len(input_deck.lattices)
    N_source = len(input_deck.sources)

    # Simulation parameters
    N_particle = input_deck.setting["N_particle"]
    N_cycle = input_deck.setting["N_cycle"]

    # Maximum object references
    # Maximum nuclides per material
    Nmax_nuclide = max([material["N_nuclide"] for material in input_deck.materials])
    # Maximum surfaces per cell
    Nmax_surface = max([cell["N_surface"] for cell in input_deck.cells])
    # Maximum cells per universe
    Nmax_cell = max([universe["N_cell"] for universe in input_deck.universes])

    # Maximum time-dependent surface slices
    Nmax_slice = 0
    for surface in input_deck.surfaces:
        Nmax_slice = max(Nmax_slice, surface["N_slice"])

    # =========================================================================
    # Other parameters needed to set up MC/DC object types
    # =========================================================================

    # Flags
    iQMC = input_deck.technique["iQMC"]

    # Numbers
    N_sensitivity = input_deck.setting["N_sensitivity"]
    N_tally_scores = 1 + N_sensitivity
    if input_deck.technique["dsm_order"] == 2:
        N_tally_scores = (
            1 + 2 * N_sensitivity + int(0.5 * N_sensitivity * (N_sensitivity - 1))
        )

    # =========================================================================
    # Default cards, if not given
    # =========================================================================

    # Default root universe
    if N_universe == 1:
        Nmax_cell = N_cell
        card = input_deck.universes[0]
        card["N_cell"] = N_cell
        card["cell_IDs"] = np.arange(N_cell)

    # =========================================================================
    # Make types
    # =========================================================================

    type_.make_type_particle(iQMC, G)
    type_.make_type_particle_record(iQMC, G)
    type_.make_type_nuclide(G, J)
    type_.make_type_material(G, J, Nmax_nuclide)
    type_.make_type_surface(Nmax_slice)
    type_.make_type_cell(Nmax_surface)
    type_.make_type_universe(Nmax_cell)
    type_.make_type_lattice(input_deck.lattices)
    type_.make_type_source(G)
    type_.make_type_tally(N_tally_scores, input_deck.tally)
    type_.make_type_uq_tally(N_tally_scores, input_deck.tally)
    type_.make_type_uq(input_deck.uq_deltas, G, J)
    type_.make_type_setting(input_deck)
    type_.make_type_technique(N_particle, G, input_deck)
    type_.make_type_global(input_deck)
    kernel.adapt_rng(nb.config.DISABLE_JIT)

    # =========================================================================
    # Make the global variable container
    #   TODO: Better alternative?
    # =========================================================================

    mcdc = np.zeros(1, dtype=type_.global_)[0]

    # =========================================================================
    # Nuclides
    # =========================================================================

    for i in range(N_nuclide):
        for name in type_.nuclide.names:
            mcdc["nuclides"][i][name] = input_deck.nuclides[i][name]

    # =========================================================================
    # Materials
    # =========================================================================

    for i in range(N_material):
        for name in type_.material.names:
            mcdc["materials"][i][name] = input_deck.materials[i][name]

    # =========================================================================
    # Surfaces
    # =========================================================================

    for i in range(N_surface):
        for name in type_.surface.names:
            if name not in ["J", "t"]:
                mcdc["surfaces"][i][name] = input_deck.surfaces[i][name]

        # Variables with possible different sizes
        for name in ["J", "t"]:
            N = len(input_deck.surfaces[i][name])
            mcdc["surfaces"][i][name][:N] = input_deck.surfaces[i][name]

    # =========================================================================
    # Cells
    # =========================================================================

    for i in range(N_cell):
        for name in type_.cell.names:
            if name not in ["surface_IDs", "positive_flags"]:
                mcdc["cells"][i][name] = input_deck.cells[i][name]

        # Variables with possible different sizes
        for name in ["surface_IDs", "positive_flags"]:
            N = len(input_deck.cells[i][name])
            mcdc["cells"][i][name][:N] = input_deck.cells[i][name]

    # =========================================================================
    # Universes
    # =========================================================================

    for i in range(N_universe):
        for name in type_.universe.names:
            if name not in ["cell_IDs"]:
                mcdc["universes"][i][name] = input_deck.universes[i][name]

        # Variables with possible different sizes
        for name in ["cell_IDs"]:
            N = mcdc["universes"][i]["N_cell"]
            mcdc["universes"][i][name][:N] = input_deck.universes[i][name]

    # =========================================================================
    # Lattices
    # =========================================================================

    for i in range(N_lattice):
        # Mesh
        for name in type_.mesh_uniform.names:
            mcdc["lattices"][i]["mesh"][name] = input_deck.lattices[i]["mesh"][name]

        # Universe IDs
        Nx = mcdc["lattices"][i]["mesh"]["Nx"]
        Ny = mcdc["lattices"][i]["mesh"]["Ny"]
        Nz = mcdc["lattices"][i]["mesh"]["Nz"]
        mcdc["lattices"][i]["universe_IDs"][:Nx, :Ny, :Nz] = input_deck.lattices[i][
            "universe_IDs"
        ]

    # =========================================================================
    # Source
    # =========================================================================

    for i in range(N_source):
        for name in type_.source.names:
            mcdc["sources"][i][name] = input_deck.sources[i][name]

    # Normalize source probabilities
    tot = 0.0
    for S in mcdc["sources"]:
        tot += S["prob"]
    for S in mcdc["sources"]:
        S["prob"] /= tot

    # =========================================================================
    # Tally
    # =========================================================================

    for name in type_.tally.names:
        if name not in ["score", "mesh"]:
            mcdc["tally"][name] = input_deck.tally[name]
    # Set mesh
    for name in type_.mesh_names:
        mcdc["tally"]["mesh"][name] = input_deck.tally["mesh"][name]

    # =========================================================================
    # Setting
    # =========================================================================

    for name in type_.setting.names:
        mcdc["setting"][name] = input_deck.setting[name]

    # Check if time boundary is above the final tally mesh time grid
    if mcdc["setting"]["time_boundary"] > mcdc["tally"]["mesh"]["t"][-1]:
        mcdc["setting"]["time_boundary"] = mcdc["tally"]["mesh"]["t"][-1]

    if input_deck.technique["iQMC"]:
        if len(mcdc["technique"]["iqmc"]["mesh"]["t"]) - 1 > 1:
            if (
                mcdc["setting"]["time_boundary"]
                > input_deck.technique["iqmc"]["mesh"]["t"][-1]
            ):
                mcdc["setting"]["time_boundary"] = input_deck.technique["iqmc"]["mesh"][
                    "t"
                ][-1]
    # =========================================================================
    # Technique
    # =========================================================================

    # Flags
    for name in [
        "weighted_emission",
        "implicit_capture",
        "population_control",
        "weight_window",
        "domain_decomp",
        "weight_roulette",
        "iQMC",
        "IC_generator",
        "branchless_collision",
        "uq",
    ]:
        mcdc["technique"][name] = input_deck.technique[name]

    # =========================================================================
    # Population control
    # =========================================================================

    # Population control technique (PCT)
    mcdc["technique"]["pct"] = input_deck.technique["pct"]
    mcdc["technique"]["pc_factor"] = input_deck.technique["pc_factor"]

    # =========================================================================
    # Weight window (WW)
    # =========================================================================

    # WW mesh
    for name in type_.mesh_names[:-1]:
        mcdc["technique"]["ww_mesh"][name] = input_deck.technique["ww_mesh"][name]

    # WW windows
    mcdc["technique"]["ww"] = input_deck.technique["ww"]
    mcdc["technique"]["ww_width"] = input_deck.technique["ww_width"]

    # =========================================================================
    # Weight roulette
    # =========================================================================

    # Threshold
    mcdc["technique"]["wr_threshold"] = input_deck.technique["wr_threshold"]

    # Survival probability
    mcdc["technique"]["wr_chance"] = input_deck.technique["wr_chance"]
    # =========================================================================
    # Domain Decomposition
    # =========================================================================

    # Set domain mesh
    if input_deck.technique["domain_decomp"]:
        name = "domain_mesh"
        mcdc["technique"][name]["x"] = input_deck.technique[name]["x"]
        mcdc["technique"][name]["y"] = input_deck.technique[name]["y"]
        mcdc["technique"][name]["z"] = input_deck.technique[name]["z"]
        mcdc["technique"][name]["t"] = input_deck.technique[name]["t"]
        mcdc["technique"][name]["mu"] = input_deck.technique[name]["mu"]
        mcdc["technique"][name]["azi"] = input_deck.technique[name]["azi"]
        # Set exchange rate
        mcdc["technique"]["exchange_rate"] = input_deck.technique["exchange_rate"]
        mcdc["technique"]["repro"] = input_deck.technique["repro"]
        # Set domain index
        mcdc["d_idx"] = input_deck.technique["d_idx"]
        mcdc["technique"]["xp_neigh"] = input_deck.technique["xp_neigh"]
        mcdc["technique"]["xn_neigh"] = input_deck.technique["xn_neigh"]
        mcdc["technique"]["yp_neigh"] = input_deck.technique["yp_neigh"]
        mcdc["technique"]["yn_neigh"] = input_deck.technique["yn_neigh"]
        mcdc["technique"]["zp_neigh"] = input_deck.technique["zp_neigh"]
        mcdc["technique"]["zn_neigh"] = input_deck.technique["zn_neigh"]
        mcdc["technique"]["work_ratio"] = input_deck.technique["work_ratio"]
        mcdc["technique"]["domain_bank_size"] = input_deck.technique["domain_bank_size"]

    # =========================================================================
    # Quasi Monte Carlo
    # =========================================================================

    if input_deck.technique["iQMC"]:
        iqmc = mcdc["technique"]["iqmc"]

        domain_decomp = input_deck.technique["domain_decomp"]
        if domain_decomp:
            mesh_shape = input_deck.technique["iqmc"]["source"].shape

        for name in type_.technique["iqmc"].names:
            if name not in [
                "mesh",
                "res",
                "lds",
                "generator",
                "sweep_counter",
                "total_source",
                "w_min",
                "score_list",
                "score",
                "material_idx",
                "Nc_total",
                "source-avg",
            ]:
                # reduce array size if domain decomp
                if domain_decomp:
                    if isinstance(input_deck.technique["iqmc"][name], np.ndarray):
                        if input_deck.technique["iqmc"][name].shape == mesh_shape:
                            input_deck.technique["iqmc"][name] = get_domain_array(
                                input_deck.technique["iqmc"][name],
                                input_deck.technique["iqmc"]["mesh"],
                                mcdc["technique"]["domain_mesh"],
                                mcdc["d_idx"],
                            )
                iqmc[name] = input_deck.technique["iqmc"][name]

        # pass in score list
        for name, bool_ in input_deck.technique["iqmc"]["score_list"].items():
            iqmc["score_list"][name] = bool_

        # pass in initial tallies, reduce array size if domain decomp
        for name, array in input_deck.technique["iqmc"]["score"].items():
            if domain_decomp:
                if iqmc["score_list"][name]:
                    if array.shape == mesh_shape:
                        array = get_domain_array(
                            array,
                            input_deck.technique["iqmc"]["mesh"],
                            mcdc["technique"]["domain_mesh"],
                            mcdc["d_idx"],
                        )
            iqmc["score"][name] = array

        # total number of cells in the global mesh
        iqmc["Nc_total"] = (
            (input_deck.technique["iqmc"]["mesh"]["x"].size - 1)
            * (input_deck.technique["iqmc"]["mesh"]["y"].size - 1)
            * (input_deck.technique["iqmc"]["mesh"]["z"].size - 1)
        )

        # Reduce mesh size if domain decomp
        # WARNING: mesh must be reduced after all other tally arrays
        if domain_decomp:
            input_deck.technique["iqmc"]["mesh"] = get_domain_mesh(
                input_deck.technique["iqmc"]["mesh"],
                mcdc["technique"]["domain_mesh"],
                mcdc["d_idx"],
            )
        iqmc["mesh"]["x"] = input_deck.technique["iqmc"]["mesh"]["x"]
        iqmc["mesh"]["y"] = input_deck.technique["iqmc"]["mesh"]["y"]
        iqmc["mesh"]["z"] = input_deck.technique["iqmc"]["mesh"]["z"]
        iqmc["mesh"]["t"] = input_deck.technique["iqmc"]["mesh"]["t"]

        # LDS generator
        iqmc["generator"] = input_deck.technique["iqmc"]["generator"]
        # minimum particle weight
        iqmc["w_min"] = 1e-13
        # variables to generate samples
        seed = mcdc["technique"]["iqmc"]["seed"]
        N_dim = mcdc["technique"]["iqmc"]["N_dim"]
        mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
        mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
        if input_deck.technique["domain_decomp"]:
            kernel.distribute_work_dd(N_particle, mcdc)
        else:
            kernel.distribute_work(N_particle, mcdc)
        N_work = int(mcdc["mpi_work_size"])
        N_start = int(mcdc["mpi_work_start"])

        # generate LDS
        if input_deck.technique["iqmc"]["generator"] == "sobol":
            if input_deck.technique["domain_decomp"]:
                sampler = qmc.Sobol(d=N_dim, scramble=True)
            else:
                sampler = qmc.Sobol(d=N_dim, scramble=False)
                # skip the first entry in Sobol sequence because its 0.0
                # skip the second because it maps to ux = 0.0
                sampler.fast_forward(2)
                sampler.fast_forward(N_start)
            mcdc["technique"]["iqmc"]["lds"] = sampler.random(N_work)

        if input_deck.technique["iqmc"]["generator"] == "halton":
            if input_deck.technique["domain_decomp"]:
                mcdc["technique"]["iqmc"]["lds"] = kernel.rhalton(
                    N_work, N_dim, seed=1234567, skip=N_start
                )
            else:
                mcdc["technique"]["iqmc"]["lds"] = kernel.halton(
                    N_work, N_dim, skip=N_start
                )

        if input_deck.technique["iqmc"]["generator"] == "random":
            np.random.seed(seed)
            seeds = np.random.randint(1e6, size=mcdc["mpi_size"])
            np.random.seed(seeds[mcdc["mpi_rank"]])
            iqmc["lds"] = np.random.random((N_particle, N_dim))

    # =========================================================================
    # Derivative Source Method
    # =========================================================================

    # Threshold
    mcdc["technique"]["dsm_order"] = input_deck.technique["dsm_order"]

    # =========================================================================
    # Variance Deconvolution - UQ
    # =========================================================================
    if mcdc["technique"]["uq"]:
        # Assumes that all tallies will also be uq tallies
        for name in type_.uq_tally.names:
            if name != "score":
                mcdc["technique"]["uq_tally"][name] = input_deck.tally[name]

        M = len(input_deck.uq_deltas["materials"])
        for i in range(M):
            idm = input_deck.uq_deltas["materials"][i]["ID"]
            mcdc["technique"]["uq_"]["materials"][i]["info"]["ID"] = idm
            mcdc["technique"]["uq_"]["materials"][i]["info"][
                "distribution"
            ] = input_deck.uq_deltas["materials"][i]["distribution"]
            for name in input_deck.uq_deltas["materials"][i]["flags"]:
                mcdc["technique"]["uq_"]["materials"][i]["flags"][name] = True
                mcdc["technique"]["uq_"]["materials"][i]["delta"][
                    name
                ] = input_deck.uq_deltas["materials"][i][name]
            flags = mcdc["technique"]["uq_"]["materials"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
                flags["speed"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True
            if mcdc["materials"][idm]["N_nuclide"] > 1:
                for name in type_.uq_mat.names:
                    mcdc["technique"]["uq_"]["materials"][i]["mean"][
                        name
                    ] = input_deck.materials[idm][name]

        N = len(input_deck.uq_deltas["nuclides"])
        for i in range(N):
            mcdc["technique"]["uq_"]["nuclides"][i]["info"][
                "distribution"
            ] = input_deck.uq_deltas["nuclides"][i]["distribution"]
            idn = input_deck.uq_deltas["nuclides"][i]["ID"]
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["ID"] = idn
            for name in type_.uq_nuc.names:
                mcdc["technique"]["uq_"]["nuclides"][i]["mean"][
                    name
                ] = input_deck.nuclides[idn][name]
            for name in input_deck.uq_deltas["nuclides"][i]["flags"]:
                mcdc["technique"]["uq_"]["nuclides"][i]["flags"][name] = True
                mcdc["technique"]["uq_"]["nuclides"][i]["delta"][
                    name
                ] = input_deck.uq_deltas["nuclides"][i][name]
            flags = mcdc["technique"]["uq_"]["nuclides"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True

    # =========================================================================
    # MPI
    # =========================================================================

    # MPI parameters
    mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
    mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
    mcdc["mpi_master"] = mcdc["mpi_rank"] == 0

    # Distribute work to MPI ranks
    if mcdc["technique"]["domain_decomp"]:
        kernel.distribute_work_dd(mcdc["setting"]["N_particle"], mcdc)
    else:
        kernel.distribute_work(mcdc["setting"]["N_particle"], mcdc)

    # =========================================================================
    # Particle banks
    # =========================================================================

    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"

    # IC generator banks
    if mcdc["technique"]["IC_generator"]:
        mcdc["technique"]["IC_bank_neutron_local"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor_local"]["tag"] = "precursor"
        mcdc["technique"]["IC_bank_neutron"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor"]["tag"] = "precursor"

    # =========================================================================
    # Eigenvalue (or fixed-source)
    # =========================================================================

    # Initial guess
    mcdc["k_eff"] = mcdc["setting"]["k_init"]

    # Activate tally scoring for fixed-source
    if not mcdc["setting"]["mode_eigenvalue"]:
        mcdc["cycle_active"] = True

    # All active eigenvalue cycle?
    elif mcdc["setting"]["N_inactive"] == 0:
        mcdc["cycle_active"] = True

    # =========================================================================
    # Source file
    # =========================================================================

    if mcdc["setting"]["source_file"]:
        with h5py.File(mcdc["setting"]["source_file_name"], "r") as f:
            # Get source particle size
            N_particle = f["particles_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["particles"][start:end]
            mcdc["bank_source"]["size"] = N_local

    # =========================================================================
    # IC file
    # =========================================================================

    if mcdc["setting"]["IC_file"]:
        with h5py.File(mcdc["setting"]["IC_file_name"], "r") as f:
            # =================================================================
            # Set neutron source
            # =================================================================

            # Get source particle size
            N_particle = f["IC/neutrons_size"][()]

            # Redistribute work
            kernel.distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["IC/neutrons"][start:end]
            mcdc["bank_source"]["size"] = N_local

            # =================================================================
            # Set precursor source
            # =================================================================

            # Get source particle size
            N_precursor = f["IC/precursors_size"][()]

            # Redistribute work
            kernel.distribute_work(N_precursor, mcdc, True)  # precursor = True
            N_local = mcdc["mpi_work_size_precursor"]
            start = mcdc["mpi_work_start_precursor"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_precursor"]["precursors"][:N_local] = f["IC/precursors"][
                start:end
            ]
            mcdc["bank_precursor"]["size"] = N_local

            # Set precursor strength
            if N_precursor > 0 and N_particle > 0:
                mcdc["precursor_strength"] = mcdc["bank_precursor"]["precursors"][0][
                    "w"
                ]

    return mcdc


def dictlist_to_h5group(dictlist, input_group, name):
    main_group = input_group.create_group(name + "s")
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % item["ID"])
        dict_to_h5group(item, group)


def dict_to_h5group(dict_, group):
    for k, v in dict_.items():
        if type(v) == dict:
            dict_to_h5group(dict_[k], group.create_group(k))
        elif v is None:
            next
        else:
            group[k] = v


# def create_domain_comm(mcdc):
#     """
#     This function creates a new MPI communicator between the first processor
#     listed in each domain.

#     """
#     work_ratio = mcdc["technique"]["work_ratio"]
#     targe_ranks = []
#     for i in range(len(work_ratio)):
#         targe_ranks.append(work_ratio[i][0])

#     group = MPI.COMM_WORLD.Get_group()
#     target_group = group.Incl(target_ranks)
#     new_comm = MPI.COMM_WORLD.Create(included_group)
#     return new_comm


# def gather_local_meshes(mcdc):
#     domain_comm = create_domain_comm(mcdc)
#     global_meshes = domain_comm.gather(mcdc["technique"]["iqmc"]["mesh"], root=0)

#     return global_meshes

# def gather_local_results(local_result, global_meshes):
#     domain_comm = create_domain_comm(mcdc)
#     # gather all local results
#     global_results = domain_comm.gather(local_result, root=0)
#     # Determine the total size of the global matrix
#     Nx = 0
#     Ny = 0
#     Nz = 0
#     Nt = 0
#     # total_size = sum(len(m) for m in global_meshes)
#     for local_mesh in global_meshes:
#         Nx += (len(local_mesh["x"]) - 1)
#         Ny += (len(local_mesh["y"]) - 1)
#         Nz += (len(local_mesh["z"]) - 1)
#         # time has not been discretized in domain decomp yet
#         Nt = (len(local_mesh["t"]) - 1)
#     total_size = int(Nx * Ny * Nz * Nt)

#     # Initialize the global matrix with zeros
#     global_matrix = np.zeros((total_size, (Nt, Ng, Nx, Ny, Nz)))

#     # Populate the global matrix with the local results
#     for meshes, result in zip(global_meshes, global_results):
#         for mesh in meshes:
#             start_points = [mesh["x"][0], mes["y"][0], mesh["z"][0], mesh["t"][0]]
#             # Convert mesh to global indices
#             global_indices = tuple(slice(int(start * total_size), int(end * total_size)) for start, end in zip(mesh["x"][0],
#                                                                                                                ))
#             global_matrix[global_indices] = result
#             current_idx += 1

#     return global_matrix


def generate_hdf5(mcdc):
    if mcdc["mpi_master"]:
        if mcdc["setting"]["progress_bar"]:
            print_msg("")
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "w") as f:
            # Input deck
            if mcdc["setting"]["save_input_deck"]:
                input_group = f.create_group("input_deck")
                dictlist_to_h5group(input_deck.nuclides, input_group, "nuclide")
                dictlist_to_h5group(input_deck.materials, input_group, "material")
                dictlist_to_h5group(input_deck.surfaces, input_group, "surface")
                dictlist_to_h5group(input_deck.cells, input_group, "cell")
                dictlist_to_h5group(input_deck.universes, input_group, "universe")
                dictlist_to_h5group(input_deck.lattices, input_group, "lattice")
                dictlist_to_h5group(input_deck.sources, input_group, "source")
                dict_to_h5group(input_deck.tally, input_group.create_group("tally"))
                dict_to_h5group(input_deck.setting, input_group.create_group("setting"))
                dict_to_h5group(
                    input_deck.technique, input_group.create_group("technique")
                )

            # Tally
            T = mcdc["tally"]
            f.create_dataset("tally/grid/t", data=T["mesh"]["t"])
            f.create_dataset("tally/grid/x", data=T["mesh"]["x"])
            f.create_dataset("tally/grid/y", data=T["mesh"]["y"])
            f.create_dataset("tally/grid/z", data=T["mesh"]["z"])
            f.create_dataset("tally/grid/mu", data=T["mesh"]["mu"])
            f.create_dataset("tally/grid/azi", data=T["mesh"]["azi"])
            f.create_dataset("tally/grid/g", data=T["mesh"]["g"])

            # Scores
            for name in T["score"].dtype.names:
                if mcdc["tally"][name]:
                    name_h5 = name.replace("_", "-")
                    f.create_dataset(
                        "tally/" + name_h5 + "/mean",
                        data=np.squeeze(T["score"][name]["mean"]),
                    )
                    f.create_dataset(
                        "tally/" + name_h5 + "/sdev",
                        data=np.squeeze(T["score"][name]["sdev"]),
                    )
                    if mcdc["technique"]["uq_tally"][name]:
                        mc_var = mcdc["technique"]["uq_tally"]["score"][name][
                            "batch_var"
                        ]
                        tot_var = mcdc["technique"]["uq_tally"]["score"][name][
                            "batch_bin"
                        ]
                        f.create_dataset(
                            "tally/" + name_h5 + "/uq_var",
                            data=np.squeeze(tot_var - mc_var),
                        )

            # Eigenvalues
            if mcdc["setting"]["mode_eigenvalue"]:
                if mcdc["technique"]["iQMC"]:
                    f.create_dataset("k_eff", data=mcdc["k_eff"])
                    N_cycle = mcdc["setting"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("N_inactive", data= mcdc["setting"]["N_inactive"])
                    f.create_dataset("N_active", data= mcdc["setting"]["N_active"])
                else:
                    N_cycle = mcdc["setting"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
                    f.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
                    f.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
                    f.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
                    f.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
                    f.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
                    if mcdc["setting"]["gyration_radius"]:
                        f.create_dataset(
                            "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                        )

            # iQMC
            if mcdc["technique"]["iQMC"]:
                # TODO: consolidate tally entry

                iqmc = mcdc["technique"]["iqmc"]

                # if mcdc["technique"]["domain_decomp"]:
                #     # gather all local meshes to root processor
                #     global_meshes = gather_local_meshes(mcdc)
                #     global_flux =

                for dim in ["x", "y", "z", "t"]:
                    f.create_dataset("iqmc/grid/" + dim, data=iqmc["mesh"][dim])

                f.create_dataset("mcdc_size", data=int(sys.getsizeof(mcdc)))

                # dump x,y,z scalar flux across all groups
                f.create_dataset(
                    "iqmc/tally/flux", data=np.squeeze(iqmc["score"]["flux"])
                )
                f.create_dataset(
                    "iqmc/tally/flux-sdev", data=np.squeeze(iqmc["score"]["flux-sdev"])
                )
                f.create_dataset(
                    "iqmc/tally/fission_source",
                    data=iqmc["score"]["fission-source"],
                )
                f.create_dataset(
                    "iqmc/tally/fission_power", data=iqmc["score"]["fission-power"]
                )
                f.create_dataset("iqmc/tally/source_constant", data=iqmc["source"])
                f.create_dataset("iqmc/tally/source_x", data=iqmc["score"]["tilt-x"])
                f.create_dataset("iqmc/tally/source_y", data=iqmc["score"]["tilt-y"])
                f.create_dataset("iqmc/tally/source_z", data=iqmc["score"]["tilt-z"])
                f.create_dataset("iqmc/tally/source_xy", data=iqmc["score"]["tilt-xy"])
                f.create_dataset("iqmc/tally/source_xz", data=iqmc["score"]["tilt-xz"])
                f.create_dataset("iqmc/tally/source_yz", data=iqmc["score"]["tilt-yz"])

                # iteration data
                f.create_dataset("iqmc/itteration_count", data=iqmc["itt"])
                f.create_dataset("iqmc/final_residual", data=iqmc["res"])
                f.create_dataset("iqmc/sweep_count", data=iqmc["sweep_counter"])
                if mcdc["setting"]["mode_eigenvalue"]:
                    f.create_dataset(
                        "iqmc/outter_itteration_count", data=iqmc["itt_outter"]
                    )
                    f.create_dataset(
                        "iqmc/outter_final_residual", data=iqmc["res_outter"]
                    )

            # Particle tracker
            if mcdc["setting"]["track_particle"]:
                with h5py.File(mcdc["setting"]["output"] + "_ptrack.h5", "w") as f:
                    N_track = mcdc["particle_track_N"]
                    f.create_dataset("tracks", data=mcdc["particle_track"][:N_track])

            # IC generator
            if mcdc["technique"]["IC_generator"]:
                Nn = mcdc["technique"]["IC_bank_neutron"]["size"]
                Np = mcdc["technique"]["IC_bank_precursor"]["size"]
                f.create_dataset(
                    "IC/neutrons",
                    data=mcdc["technique"]["IC_bank_neutron"]["particles"][:Nn],
                )
                f.create_dataset(
                    "IC/precursors",
                    data=mcdc["technique"]["IC_bank_precursor"]["precursors"][:Np],
                )
                f.create_dataset("IC/neutrons_size", data=Nn)
                f.create_dataset("IC/precursors_size", data=Np)
                f.create_dataset(
                    "IC/fission", data=mcdc["technique"]["IC_fission"] / Nn
                )

    # Save particle?
    if mcdc["setting"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Master saves the particle
        if mcdc["mpi_master"]:
            # Remove unwanted particle fields
            neutrons = np.concatenate(neutrons[:])

            # Create dataset
            with h5py.File(mcdc["setting"]["output"] + ".h5", "a") as f:
                f.create_dataset("particles", data=neutrons[:])
                f.create_dataset("particles_size", data=len(neutrons[:]))


def closeout(mcdc):
    # Runtime
    if mcdc["mpi_master"]:
        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
            for name in [
                "total",
                "preparation",
                "simulation",
                "output",
                "bank_management",
            ]:
                f.create_dataset(
                    "runtime/" + name, data=np.array([mcdc["runtime_" + name]])
                )

    print_runtime(mcdc)
    input_deck.reset()
