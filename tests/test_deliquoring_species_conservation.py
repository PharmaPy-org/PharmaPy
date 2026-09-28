"""Discrete solute-conservation regression for ``DeliquoringStep`` (issue #29).

The defect lives entirely in the finite-volume flux assembly of
``DeliquoringStep.material_balance``, so these tests drive the unit's public RHS
entry point ``unit_model`` with a compact synthetic cake state instead of a full
``solve_unit`` transient (which requires Assimulo). The cake fixture is built
from the same correlations ``solve_unit`` uses, but the expected outlet solute
efflux is derived from the *returned* saturation derivative rather than from the
production flux expression, so the assertion cannot go green by restating the
code under test.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.SolidLiquidSep import DeliquoringStep


pytestmark = pytest.mark.unit


NUM_NODES = 5  # axial finite volumes along the cake [-]
NUM_SPECIES = 2  # solutes carried by the mobile liquid [-]

# Irreducible saturation. Representative of the value ``get_sat_inf`` returns
# from its 0.155*(1 + 0.031*Ca**-0.49) correlation for a moderately fine cake.
SAT_INF = 0.2  # [-]

# Capillary threshold pressure, from the same expression ``solve_unit`` uses,
# p_thresh = 4.6*(1 - eps)*sigma/(eps*d), with eps = 0.5 [-],
# sigma = 0.03 N/m and d = 4.6e-6 m.
P_THRESH = 3.0e4  # [Pa]

P_ATM = 1.01325e5  # ambient pressure at the cake surface [Pa]
DELTA_P = 5.0e4  # applied pressure drop across the cake [Pa]


def _build_deliquoring_step():
    """Build a ``DeliquoringStep`` carrying only the RHS-relevant attributes.

    ``DeliquoringStep.Phases`` normally installs the grid and the cake
    properties, but that setter needs full ``Cake``/``Liquid``/``Solid`` phase
    objects and is unrelated to the flux assembly under test. The attributes set
    here are exactly those ``material_balance`` reads.

    Returns
    -------
    DeliquoringStep
        Unit whose ``p_gas`` [Pa], ``p_thresh`` [Pa], ``sat_inf`` [-],
        ``delta_z`` [-] and ``Liquid_1.num_species`` [-] are populated.
    """
    unit = DeliquoringStep(num_nodes=NUM_NODES)

    # Gas pressure on the N + 1 face grid, decaying from the pressurized face
    # to ambient exactly as ``solve_unit`` builds it [Pa].
    unit.p_gas = np.linspace(P_ATM + DELTA_P, P_ATM, NUM_NODES + 1)

    unit.p_thresh = P_THRESH  # [Pa]
    unit.sat_inf = SAT_INF  # [-]

    # Uniform non-dimensional cell width on z in [0, 1], as built by the
    # ``Phases`` setter from ``np.linspace(0, 1, num_nodes + 1)`` [-].
    unit.delta_z = np.diff(np.linspace(0, 1, NUM_NODES + 1))  # [-]

    unit.Liquid_1 = SimpleNamespace(num_species=NUM_SPECIES)

    return unit


def _cake_state():
    """Return a partially drained, non-degenerate cake state.

    The conservation identity under test must hold for *any* admissible state,
    so the profiles only need to be valid and non-degenerate: reduced saturation
    increases toward the drainage face (the gas-entry side dries first, which
    makes the liquid flux positive in +z), and the two solutes carry opposite
    monotone trends so that a species/axis transposition cannot cancel out.

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
        )
    )  # [-]

    return sat_star, conc_star


def test_deliquoring_solute_inventory_matches_boundary_efflux():
    """Discrete solute inventory changes only through the outlet face flux.

    The continuous species balance eps*d(S*C)/dt = -d(q*C)/dz makes the cake's
    total solute holdup change only through the boundary fluxes. With the
    zero-flux inlet condition used by ``material_balance``
    (``upwind_fvm(q_liq, boundary_cond=0)``), the only open boundary is the
    outlet face, so the discrete holdup rate must equal minus the outlet liquid
    flux times the upwind (last-cell) concentration, per species.
    """
    unit = _build_deliquoring_step()
    sat_star, conc_star = _cake_state()

    # Interleaved [saturation, mass_conc...] ordering per node, as
    # ``unit_model`` receives it from the integrator.
    states = np.column_stack((sat_star, conc_star)).ravel()

    theta = 0.35  # non-dimensional deliquoring time [-] (autonomous RHS)
    derivatives = unit.unit_model(theta, states).reshape(NUM_NODES, NUM_SPECIES + 1)

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

    np.testing.assert_allclose(inventory_rate, expected_rate, rtol=1e-10, atol=1e-14)


def test_deliquoring_concentration_derivative_uses_upwind_face_flux():
    """Per-cell concentration rates use the upstream liquid concentration.

    Eliminating ``dS/dtheta`` between the conservative solute and saturation
    balances leaves a local closed form involving the liquid flux at each
    cell's left face. Those face fluxes are recovered cumulatively from the
    returned saturation derivative, independently of the production flux
    assembly, so the assertion pins the donor cell at every interior face.
    """
    unit = _build_deliquoring_step()
    sat_star, conc_star = _cake_state()

    states = np.column_stack((sat_star, conc_star)).ravel()  # [-]
    theta = 0.35  # non-dimensional deliquoring time [-] (autonomous RHS)
    derivatives = unit.unit_model(theta, states).reshape(NUM_NODES, NUM_SPECIES + 1)

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

    np.testing.assert_allclose(dconc_dtheta, expected_rate, rtol=1e-10, atol=1e-14)
