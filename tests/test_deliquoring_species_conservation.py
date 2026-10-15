"""Discrete solute-conservation regression for ``DeliquoringStep`` (issue #29).

Real liquid/solid phases and a ``Cake`` enter through the public ``Phases``
setter, which supplies the grid, species order, and inventory. The shared
``drying_cake_factory`` uses documented synthetic property data, not substitute
objects. Prescribed pressure and residual saturation isolate the dimensionless
finite-volume contract without requiring an Assimulo transient. Expected solute
efflux follows independently from the returned saturation derivative. The native
setup path is covered by ``test_deliquoring_particle_size_units.py``.
"""

import numpy as np
import pytest

from PharmaPy.SolidLiquidSep import DeliquoringStep


pytestmark = pytest.mark.unit


NUM_NODES = 5  # axial finite volumes along the cake [-]
NUM_SPECIES = 3  # [-], water/ethanol/carrier order in the shared cake fixture

# Irreducible saturation. Representative of the value ``get_sat_inf`` returns
# from its 0.155*(1 + 0.031*Ca**-0.49) correlation for a moderately fine cake.
SAT_INF = 0.2  # [-]

# Capillary threshold pressure, from the same expression ``solve_unit`` uses,
# p_thresh = 4.6*(1 - eps)*sigma/(eps*d), with eps = 0.5 [-],
# sigma = 0.03 N/m and d = 4.6e-6 m.
P_THRESH = 3.0e4  # [Pa]

P_ATM = 1.01325e5  # ambient pressure at the cake surface [Pa]
DELTA_P = 5.0e4  # applied pressure drop across the cake [Pa]
CONSERVATION_RTOL = 1e-10  # [-], roundoff allowance for flux telescoping
CONSERVATION_ATOL = 1e-14  # [-], roundoff allowance at zero-flux cells


def _build_deliquoring_step(drying_cake_factory):
    """Attach a real cake and prescribe deterministic drainage conditions.

    Parameters
    ----------
    drying_cake_factory : callable
        Fixture constructing a production cake with real liquid/solid phases.

    Returns
    -------
    DeliquoringStep
        Unit with public phase/grid setup and specified ``p_gas`` [Pa],
        ``p_thresh`` [Pa], and ``sat_inf`` [-].
    """
    unit = DeliquoringStep(num_nodes=NUM_NODES)
    unit.Phases = drying_cake_factory()

    # Gas pressure on the N + 1 face grid, decaying from the pressurized face
    # to ambient exactly as ``solve_unit`` builds it [Pa].
    unit.p_gas = np.linspace(P_ATM + DELTA_P, P_ATM, NUM_NODES + 1)

    unit.p_thresh = P_THRESH  # [Pa]
    unit.sat_inf = SAT_INF  # [-]

    return unit


def _cake_state():
    """Return a partially drained, non-degenerate cake state.

    The conservation identity under test must hold for *any* admissible state,
    so the profiles only need to be valid and non-degenerate: reduced saturation
    increases toward the drainage face (the gas-entry side dries first, which
    makes the liquid flux positive in +z). Opposite monotone trends for the
    first two species and a nonmonotone third profile expose donor-cell and
    species-order mistakes. Five cells and three species separate the axes.

    Returns
    -------
    sat_star : ndarray, shape (NUM_NODES,)
        Reduced saturation (S - s_inf)/(1 - s_inf) per cell [-].
    conc_star : ndarray, shape (NUM_NODES, NUM_SPECIES)
        Reduced liquid-phase mass concentration per cell and species [-].
    """
    sat_star = np.linspace(0.45, 0.85, NUM_NODES)  # [-]

    conc_star = np.column_stack(
        (
            np.linspace(0.80, 0.20, NUM_NODES),  # species 0 [-]
            np.linspace(0.15, 0.65, NUM_NODES),  # species 1 [-]
            np.array([0.05, 0.15, 0.35, 0.25, 0.45]),  # species 2 [-]
        )
    )  # [-]

    return sat_star, conc_star


def test_deliquoring_solute_inventory_matches_boundary_efflux(drying_cake_factory):
    """Discrete solute inventory changes only through the outlet face flux.

    The continuous species balance eps*d(S*C)/dt = -d(q*C)/dz makes the cake's
    total solute holdup change only through the boundary fluxes. With the
    zero-flux inlet condition used by ``material_balance``
    (``upwind_fvm(q_liq, boundary_cond=0)``), the only open boundary is the
    outlet face, so the discrete holdup rate must equal minus the outlet liquid
    flux times the upwind (last-cell) concentration, per species.

    Parameters
    ----------
    drying_cake_factory : callable
        Fixture constructing real liquid/solid cake collaborators.
    """
    unit = _build_deliquoring_step(drying_cake_factory)
    sat_star, conc_star = _cake_state()  # [-], [-]

    # Interleaved [saturation, mass_conc...] ordering per node, as
    # ``unit_model`` receives it from the integrator.
    states = np.column_stack((sat_star, conc_star)).ravel()  # [-]

    theta = 0.35  # non-dimensional deliquoring time [-] (autonomous RHS)
    derivatives = unit.unit_model(theta, states).reshape(
        NUM_NODES, NUM_SPECIES + 1)  # [-], per unit non-dimensional time

    dsat_star_dtheta = derivatives[:, 0]  # [-] per unit non-dimensional time
    dconc_dtheta = derivatives[:, 1:]  # [-] per unit non-dimensional time

    # Actual saturation and its rate, recovered from the reduced state.
    saturation = sat_star * (1 - SAT_INF) + SAT_INF  # [-]
    dsat_dtheta = dsat_star_dtheta * (1 - SAT_INF)  # [-] per non-dim. time

    cell_width = unit.delta_z  # [-]

    # Outlet liquid efflux, derived from the returned saturation derivative
    # alone: summing the saturation balance over all cells telescopes to the
    # single open boundary, so sum_i (dS_i/dtheta * dz_i) = -flux_out.
    liquid_efflux = -np.sum(dsat_dtheta * cell_width)  # [-] per non-dim. time
    assert liquid_efflux > 0, "fixture must drain liquid through the outlet face"

    # d/dtheta sum_i (S_i * C_i * dz_i), expanded with the product rule.
    inventory_rate = np.sum(
        (
            saturation[:, np.newaxis] * dconc_dtheta
            + conc_star * dsat_dtheta[:, np.newaxis]
        )
        * cell_width[:, np.newaxis],
        axis=0,
    )  # [-] per non-dimensional time, per species

    # Upwind outlet face carries the last cell's concentration.
    expected_rate = -liquid_efflux * conc_star[-1]  # [-] per non-dim. time

    np.testing.assert_allclose(inventory_rate, expected_rate,
                               rtol=CONSERVATION_RTOL, atol=CONSERVATION_ATOL)


def test_deliquoring_concentration_derivative_uses_upwind_face_flux(
        drying_cake_factory):
    """Per-cell concentration rates use the upstream liquid concentration.

    Eliminating ``dS/dtheta`` between the conservative solute and saturation
    balances leaves a local closed form involving the liquid flux at each
    cell's left face. Those face fluxes are recovered cumulatively from the
    returned saturation derivative, independently of the production flux
    assembly, so the assertion pins the donor cell at every interior face.

    Parameters
    ----------
    drying_cake_factory : callable
        Fixture constructing real liquid/solid cake collaborators.
    """
    unit = _build_deliquoring_step(drying_cake_factory)
    sat_star, conc_star = _cake_state()  # [-], [-]

    states = np.column_stack((sat_star, conc_star)).ravel()  # [-]
    theta = 0.35  # non-dimensional deliquoring time [-] (autonomous RHS)
    derivatives = unit.unit_model(theta, states).reshape(
        NUM_NODES, NUM_SPECIES + 1)  # [-], per unit non-dimensional time

    dsat_star_dtheta = derivatives[:, 0]  # [-] per non-dimensional time
    dconc_dtheta = derivatives[:, 1:]  # [-] per non-dimensional time

    saturation = sat_star * (1 - SAT_INF) + SAT_INF  # [-]
    cell_width = unit.delta_z  # [-]

    # Liquid flux on the left face of every cell, recovered from the returned
    # saturation balance: q_left[0] = 0 and q_right = q_left - dS*/dtheta*dz.
    face_flux = np.concatenate(
        ([0.0], np.cumsum(-dsat_star_dtheta * cell_width))
    )[:-1]  # [-]
    conc_upwind = np.vstack((conc_star[0], conc_star[:-1]))  # [-]

    expected_rate = (
        -(1 - SAT_INF)
        * face_flux[:, np.newaxis]
        * (conc_star - conc_upwind)
        / (saturation[:, np.newaxis] * cell_width[:, np.newaxis])
    )  # [-] per non-dimensional time

    np.testing.assert_allclose(dconc_dtheta, expected_rate,
                               rtol=CONSERVATION_RTOL, atol=CONSERVATION_ATOL)
