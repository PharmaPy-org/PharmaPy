"""Verification contracts for the WP1.5 M6 bioconjugation case selection (#357).

Scope: exercises the representative cysteine-maleimide bioconjugation reaction
networks, unit conversions (both ``[L/(mmol*s)]`` and ``[L/(umol*s)]`` to
``[L/(mol*s)]``), stoichiometric mass conservation, macromixing and micromixing
Damkohler timescale scaling, dual-wavelength UV-Vis and multivariate spectral
PAT contracts, multi-experiment (batch and fed-batch) parameter estimation,
``CalibrationRecord`` provenance round-trip, WP4 four-step STN/RTN process
sequence balances, and native ``BatchReactor`` / ``SemibatchReactor`` execution
plus blocker reproducers documented in
``doc/review_notes/bioconjugation_case_selection.md``.

Data provenance:
- ``origin='literature'``: kinetic rate constants, standard deviations, 95%
  confidence intervals, initial concentrations, vessel mixing times, and
  248/280 nm molar extinction coefficients from peer-reviewed literature
  (Andris et al., 2019, doi:10.1016/j.jbiotec.2019.09.013; Weggen et al., 2023,
  doi:10.3389/fbioe.2023.1123842; Weggen et al., 2024,
  doi:10.3389/fbioe.2024.1403644; Weggen et al., 2025,
  doi:10.1002/bit.28899; Hamblett et al., 2004,
  doi:10.1158/1078-0432.CCR-04-0789; Schiemer et al., 2023,
  doi:10.1002/bit.28428).
- ``origin='synthetic'``: forward-simulated concentration time series, the
  4-wavelength MCR/PCR verification absorptivities at 310/342 nm, schema
  placeholder thermal properties (``cp_liq``, ``t_crit``), and WP4 scheduling
  task durations and hold-time design parameters. These verify the model, PAT,
  and scheduling interfaces and are not claimed as an empirical calibration
  dataset.

Core unit tests run in milliseconds without Assimulo; the final integration
test uses ``pytest.importorskip("assimulo")`` to exercise native
``BatchReactor`` and ``SemibatchReactor`` integration and reproduce the
demonstrated solver/sensitivity blockers.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import t as student_t

from PharmaPy.Calibration import PCR_calibration
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import (
    Experiment,
    Measurement,
    MultipleCurveResolution,
    ParameterEstimation,
)
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor, SemibatchReactor
from PharmaPy.Records import (
    CalibrationRecord,
    DataSource,
    ModelRevision,
    Transformation,
)
from PharmaPy.Streams import LiquidStream


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_NOTE = REPO_ROOT / "doc" / "review_notes" / "bioconjugation_case_selection.md"

# Exact unit conversions from literature bimolecular rate constants
# [L/(mmol*s)] and [L/(umol*s)] to PharmaPy's [L/(mol*s)] basis.
MMOL_PER_MOL = 1000.0  # [mmol/mol], exact metric prefix (1 mol = 1e3 mmol)
UMOL_PER_MOL = 1.0e6  # [umol/mol], exact metric prefix (1 mol = 1e6 umol)

# Model 1A rate constants: intact DAR 2 conjugation of ADC1 with surrogate
# maleimide payload NPM in 20 mM NaH2PO4, 1 mM EDTA, pH 7.0, 10% (v/v) DMSO
# at 295.15 K (Andris et al., 2019; Weggen et al., 2023, Eqs. 3-5).
K1_LIT_MMOL = 0.797  # [L/(mmol*s)], Weggen et al. (2023), Eq. (3)
K2_LIT_MMOL = 1.476  # [L/(mmol*s)], Weggen et al. (2023), Eq. (4)
K1_MODEL_1A = K1_LIT_MMOL * MMOL_PER_MOL  # [L/(mol*s)]
K2_MODEL_1A = K2_LIT_MMOL * MMOL_PER_MOL  # [L/(mol*s)]
K3_MODEL_1A = 1.55e-3  # [1/s], payload sink, Weggen et al. (2023), Eq. (5)

# Tier 2 rate constants: Stage 1 TCEP disulfide reduction (WithInt + ReOx
# 8k-model, Weggen et al., 2025, Biotechnol. Bioeng. 122(2):342-356, Sec. 2.4,
# Sec. 3.1, Fig. 3, and SI Table S4). Published in [L/(umol*s)].
K1_RED_LIT_UMOL = 6.98e-4  # [L/(umol*s)], fastest inter-HL reduction step at 293.15 K
K3_RED_LIT_UMOL = 6.81e-5  # [L/(umol*s)], slowest inter-HL reduction step at 293.15 K
K_RED_SEED_LIT_UMOL = 1.0e-4  # [L/(umol*s)], initial optimizer seed, Sec. 2.4
K1_RED_TIER2 = K1_RED_LIT_UMOL * UMOL_PER_MOL  # [L/(mol*s)], 698.0 L/(mol*s)
K3_RED_TIER2 = K3_RED_LIT_UMOL * UMOL_PER_MOL  # [L/(mol*s)], 68.1 L/(mol*s)
K_RED_SEED_TIER2 = K_RED_SEED_LIT_UMOL * UMOL_PER_MOL  # [L/(mol*s)], 100.0 L/(mol*s)

MODEL_1A_REACTIONS = (
    "mAb + Drug --> mAb_1Drug",
    "mAb_1Drug + Drug --> mAb_2Drug",
    "Drug --> Drug_depl",
)
MODEL_1A_PARTICIPATING = (
    "mAb",
    "mAb_1Drug",
    "mAb_2Drug",
    "Drug",
    "Drug_depl",
)

# Representative pure-component property table for dilute aqueous ADC buffer
# (10% v/v DMSO in 20 mM sodium phosphate buffer, rho_liq = 1010.5 kg/m**3,
# Weggen et al., 2023, Sec. 2.2.2) with surrogate maleimide payload
# N-(1-pyrenyl)maleimide (NPM, C20H11NO2, M = 297.31 g/mol, Weggen et al.,
# 2023, Sec. 2.1.2). Molecular weights satisfy exact stoichiometric mass
# conservation (stoich_matrix @ mw == 0) across R1, R2, and R3.
# Note: cp_liq [J/(mol*K)] and t_crit [K] are synthetic schema-required
# placeholders (origin='synthetic') that are not used by isothermal liquid
# reactor integration (isothermal=True).
BUFFER_DENSITY = 1010.5  # [kg/m**3], Weggen et al. (2023), Sec. 2.2.2
REACTION_TEMP = 295.15  # [K], 22 degC isothermal laboratory condition
MAB_MW = 150000.0  # [g/mol], intact IgG1 molar mass, Weggen et al. (2023), Sec. 2.1.1
NPM_MW = 297.31  # [g/mol], N-(1-pyrenyl)maleimide (C20H11NO2) molar mass
BUFFER_MW = 19.70  # [g/mol], mole-fraction-weighted 90% v/v H2O + 10% v/v DMSO
BIOCONJUGATION_DATABASE = {
    "mAb": {
        "mw": MAB_MW,  # [g/mol]
        "rho_liq": BUFFER_DENSITY,  # [kg/m**3]
        "cp_liq": [210000.0],  # [J/(mol*K)], synthetic placeholder
        "t_crit": 800.0,  # [K], synthetic placeholder
    },
    "mAb_1Drug": {
        "mw": MAB_MW + NPM_MW,  # [g/mol], 150297.31 g/mol (exact R1 mass balance)
        "rho_liq": BUFFER_DENSITY,  # [kg/m**3]
        "cp_liq": [210420.0],  # [J/(mol*K)], synthetic placeholder
        "t_crit": 800.0,  # [K], synthetic placeholder
    },
    "mAb_2Drug": {
        "mw": MAB_MW + 2.0 * NPM_MW,  # [g/mol], 150594.62 g/mol (exact R2 mass balance)
        "rho_liq": BUFFER_DENSITY,  # [kg/m**3]
        "cp_liq": [210840.0],  # [J/(mol*K)], synthetic placeholder
        "t_crit": 800.0,  # [K], synthetic placeholder
    },
    "Drug": {
        "mw": NPM_MW,  # [g/mol], 297.31 g/mol
        "rho_liq": BUFFER_DENSITY,  # [kg/m**3]
        "cp_liq": [420.0],  # [J/(mol*K)], synthetic placeholder
        "t_crit": 750.0,  # [K], synthetic placeholder
    },
    "Drug_depl": {
        "mw": NPM_MW,  # [g/mol], 297.31 g/mol (effective payload-moiety basis for R3)
        "rho_liq": BUFFER_DENSITY,  # [kg/m**3]
        "cp_liq": [420.0],  # [J/(mol*K)], synthetic placeholder
        "t_crit": 750.0,  # [K], synthetic placeholder
    },
    "buffer": {
        "mw": BUFFER_MW,  # [g/mol], 19.70 g/mol
        "rho_liq": BUFFER_DENSITY,  # [kg/m**3]
        "cp_liq": [78.0],  # [J/(mol*K)], synthetic placeholder
        "t_crit": 650.0,  # [K], synthetic placeholder
    },
}

# Model 1B rate constants: reduced heavy-chain DAR 2 Simple (1k + sink) model
# for ADC1 + Drug1 across batch (1 mL) and fed-batch (4 mL) runs (Weggen et
# al., 2024, Front. Bioeng. Biotechnol. 12:1403644, Table 2, Eqs. 13-14 &
# Sec. 3.1). Table 2 reports both parameter standard deviation sigma_1 (Eq. 13)
# and the 95% confidence interval [lb, ub] (Eq. 14).
K1_DAR2_DRUG1_MMOL = 0.251  # [L/(mmol*s)], Weggen et al. (2024), Table 2
K1_DAR2_DRUG1_SD_MMOL = 0.023  # [L/(mmol*s)], parameter SD sigma_1 (Eq. 13), Table 2
K1_DAR2_DRUG1_CI95_BOUNDS_MMOL = (0.204, 0.298)  # [L/(mmol*s)], 95% CI [lb, ub], Table 2
K1_DAR2_DRUG1_CI95_HALF_MMOL = 0.047  # [L/(mmol*s)], 95% CI half-width (Eq. 14)
K1_DAR2_DRUG1 = K1_DAR2_DRUG1_MMOL * MMOL_PER_MOL  # [L/(mol*s)]
K_DRUG_DRUG1 = 1.0e-3  # [1/s], Weggen et al. (2024), Sec. 3.1

# Mixing vessel 95% homogenization times, specific power inputs, and standard
# reactant concentrations from Weggen et al. (2023, Table 1, Table 2,
# Sec. 2.2.5, Sec. 2.2.8 & Sec. 3.2.1).
TAU_MIX_TUBE = 0.5  # [s], internally stirred 1.5 mL reaction tube
TAU_MIX_GST1 = 9.4  # [s], 300 mL lab glass stirred tank at 60 rpm
TAU_MIX_SUM = 17.6  # [s], 25 L single-use mixer at 400 rpm
TAU_MIX_GST2 = 32.2  # [s], 22 L glass stirred tank at 60 rpm
POWER_PER_VOL_GST1 = 3.81  # [W/m**3], Weggen et al. (2023), Table 1
POWER_PER_VOL_SUM = 23.71  # [W/m**3], Weggen et al. (2023), Table 1
POWER_PER_VOL_GST2 = 2.44  # [W/m**3], Weggen et al. (2023), Table 1
ENGULFMENT_COEFF = 17.24  # [-], Baldyga-Bourne / Vicum et al. (2004), Eq. (5)
KINEMATIC_VISC_WATER = 1.0e-6  # [m**2/s], dilute aqueous kinematic viscosity
WATER_DENSITY_APPROX = 1000.0  # [kg/m**3], density basis for epsilon = (P/V)/rho
MAB_MASS_CONC_STD = 5.0  # [g/L], Weggen et al. (2023), Table 2
C_MAB_STD = MAB_MASS_CONC_STD / MAB_MW  # [mol/L], 3.333e-5 mol/L
PAYLOAD_EXCESS_STD = 5.0  # [-], 5x molar payload excess, Table 2
C_DRUG_BULK_STD = PAYLOAD_EXCESS_STD * C_MAB_STD  # [mol/L], 1.667e-4 mol/L
C_DRUG_FEED_PLUME = 1.0e-2  # [mol/L], 10 mM stock feed plume, Sec. 2.2.5

# Dual-wavelength UV-Vis molar extinction coefficients at 248 nm and 280 nm
# for cAC10 IgG1 mAb and vc-MMAE (SGD-1006) drug-linker (origin='literature':
# Hamblett et al., 2004, Clin. Cancer Res. 10(20):7063-7070, p. 7064).
# SHORT_PATHLENGTH_CM = 0.10 cm represents a micro-volume / flow-cell optical
# pathlength (e.g., Unchained Labs Stunner, Weggen et al., 2024, Sec. 2.1.4)
# that keeps undiluted 5 g/L mAb absorbance inside the linear detector regime
# (A_280 <= 1.5 AU).
PATHLENGTH_CM = 1.0  # [cm], standard 1 cm reference cuvette pathlength
SHORT_PATHLENGTH_CM = 0.10  # [cm], 1 mm micro-volume / inline UV flow cell
MAX_LINEAR_ABSORBANCE_AU = 1.5  # [-], upper bound of linear photodiode regime
EPS_MAB_248 = 9.413e4  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_MAB_280 = 2.3415e5  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_DRUG_248 = 1.59e4  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_DRUG_280 = 1.50e3  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064

# Justified numerical tolerances:
# - CLOSED_FORM_RTOL (1e-12): direct algebraic/matrix identity evaluations in
#   IEEE 754 double precision.
# - ODE_FIT_RTOL (1e-4): Levenberg-Marquardt parameter recovery across two-piece
#   RK45 ODE integrations with finite-difference sensitivities.
# - REACTOR_REF_RTOL (1e-4): agreement between Assimulo CVode DAE integration of
#   native PharmaPy reactors and independent SciPy RK45 integration.
# - REPLAY_RTOL (4 * eps): bit-identical deterministic CalibrationRecord.rerun
#   re-execution on the same platform.
CLOSED_FORM_RTOL = 1e-12  # [-], IEEE floating-point algebraic identity tolerance
ODE_FIT_RTOL = 1e-4  # [-], ODE-integrated LM parameter recovery tolerance
REACTOR_REF_RTOL = 1e-4  # [-], CVode vs SciPy RK45 trajectory agreement tolerance
REPLAY_RTOL = 4 * np.finfo(float).eps  # [-], deterministic rerun tolerance


def _write_bioconjugation_db(tmp_path):
    """Write the pure-component JSON database for the bioconjugation case.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    str
        Filesystem path to the written JSON database.
    """
    db_path = tmp_path / "bioconjugation_components.json"
    db_path.write_text(json.dumps(BIOCONJUGATION_DATABASE, indent=2), encoding="utf-8")
    return str(db_path)


def _solve_dar2_heavy_chain_experiment(
    params,
    time_grid_s,
    reord_sens=False,
    c_h0_init_mol_l=6.0e-5,
    c_drug_init_mol_l=1.5e-4,
    v0_l=1.0e-3,
    feed_rate_l_s=0.0,
    feed_duration_s=0.0,
    c_drug_feed_mol_l=0.0,
):
    """Integrate the Weggen et al. (2024) Simple 1k heavy-chain DAR 2 model.

    Solves the isothermal batch or fed-batch material balance (Weggen et al.,
    2024, Eqs. 1-7) for mono-thiol heavy chain ``H0_1``, conjugated heavy
    chain ``H1_1``, free payload ``Drug``, and inactivated payload
    ``Drug_depl``:

    - ``H0_1 + Drug --k1--> H1_1``
    - ``Drug --k_drug--> Drug_depl``

    When a positive finite syringe-pump feed window ``0 < feed_duration_s <
    times[-1]`` is active, integration is performed in two smooth pieces
    ``[0, feed_duration_s]`` and ``[feed_duration_s, times[-1]]`` so the solver
    never steps across a discontinuous flow derivative.

    Parameters
    ----------
    params : sequence of float
        ``(k1, k_drug)`` with ``k1`` in [L/(mol*s)] and ``k_drug`` in [1/s].
    time_grid_s : array_like
        Observation times [s].
    reord_sens : bool, optional
        Sensitivity reordering flag accepted for ``MultipleCurveResolution``
        callback compatibility [-].
    c_h0_init_mol_l : float, optional
        Initial concentration of unconjugated reactive heavy chain ``H0_1``
        [mol/L].
    c_drug_init_mol_l : float, optional
        Initial concentration of free payload ``Drug`` in the vessel [mol/L].
    v0_l : float, optional
        Initial liquid volume [L].
    feed_rate_l_s : float, optional
        Volumetric payload feed rate during ``0 <= t <= feed_duration_s``
        [L/s].
    feed_duration_s : float, optional
        Duration of the constant syringe-pump feed [s].
    c_drug_feed_mol_l : float, optional
        Payload concentration in the syringe-pump feed solution [mol/L].

    Returns
    -------
    numpy.ndarray
        Concentration profile of shape ``(len(time_grid_s), 4)`` with columns
        ``(c_H0_1, c_H1_1, c_Drug, c_Drug_depl)`` [mol/L].
    """
    _ = reord_sens
    k1_val, k_drug_val = float(params[0]), float(params[1])  # [L/(mol*s)], [1/s]
    times = np.asarray(time_grid_s, dtype=float)  # [s]

    def _make_rhs(active_flow_l_s):
        """Build a smooth constant-flow RHS for a single integration piece.

        Parameters
        ----------
        active_flow_l_s : float
            Constant volumetric feed rate on the current time interval [L/s].

        Returns
        -------
        callable
            Right-hand-side function ``rhs(time_s, state)`` returning time
            derivatives ``[dc_h0, dc_h1, dc_drug, dc_depl, dvol]`` in
            ``[mol/(L*s) x 4, L/s]``.
        """

        def rhs(_time_s, state):
            """Evaluate time derivatives of heavy-chain species and volume.

            Parameters
            ----------
            _time_s : float
                Current integration time [s].
            state : sequence of float
                ``(c_h0, c_h1, c_drug, c_depl, vol_l)`` in
                ``[mol/L x 4, L]``.

            Returns
            -------
            list of float
                Time derivatives ``[dc_h0, dc_h1, dc_drug, dc_depl, dvol]`` in
                ``[mol/(L*s) x 4, L/s]``.
            """
            c_h0, c_h1, c_drug, c_depl, vol_l = state  # [mol/L x 4, L]
            dilution_s = active_flow_l_s / vol_l  # [1/s]
            r_conj = k1_val * c_h0 * c_drug  # [mol/(L*s)]
            r_depl = k_drug_val * c_drug  # [mol/(L*s)]
            dc_h0 = -r_conj - dilution_s * c_h0  # [mol/(L*s)]
            dc_h1 = r_conj - dilution_s * c_h1  # [mol/(L*s)]
            dc_drug = (
                -r_conj - r_depl + dilution_s * (c_drug_feed_mol_l - c_drug)
            )  # [mol/(L*s)]
            dc_depl = r_depl - dilution_s * c_depl  # [mol/(L*s)]
            return [dc_h0, dc_h1, dc_drug, dc_depl, active_flow_l_s]

        return rhs

    initial_state = [
        float(c_h0_init_mol_l),
        0.0,
        float(c_drug_init_mol_l),
        0.0,
        float(v0_l),
    ]  # [mol/L x 4, L]
    t_end_s = float(times[-1])  # [s]

    if feed_rate_l_s > 0.0 and 0.0 < feed_duration_s < t_end_s:
        times_piece1 = times[times <= feed_duration_s]  # [s]
        eval_piece1 = (
            times_piece1
            if (
                times_piece1.size > 0
                and np.isclose(times_piece1[-1], feed_duration_s)
            )
            else np.append(times_piece1, feed_duration_s)
        )  # [s]
        sol1 = solve_ivp(
            _make_rhs(float(feed_rate_l_s)),
            (0.0, float(feed_duration_s)),
            initial_state,
            t_eval=eval_piece1,
            method="RK45",
            rtol=1e-10,
            atol=1e-13,
        )
        if not sol1.success:
            raise RuntimeError(f"Piece 1 ODE integration failed: {sol1.message}")
        state_at_switch = sol1.y[:, -1].tolist()  # [mol/L x 4, L]
        times_piece2 = times[times > feed_duration_s]  # [s]
        eval_piece2 = np.insert(times_piece2, 0, feed_duration_s)  # [s]
        sol2 = solve_ivp(
            _make_rhs(0.0),
            (float(feed_duration_s), t_end_s),
            state_at_switch,
            t_eval=eval_piece2,
            method="RK45",
            rtol=1e-10,
            atol=1e-13,
        )
        if not sol2.success:
            raise RuntimeError(f"Piece 2 ODE integration failed: {sol2.message}")
        y_piece1 = sol1.y[:4, : times_piece1.size].T  # [mol/L]
        y_piece2 = sol2.y[:4, 1:].T  # [mol/L]
        return np.vstack((y_piece1, y_piece2))

    active_flow = (
        float(feed_rate_l_s)
        if (feed_rate_l_s > 0.0 and feed_duration_s >= t_end_s)
        else 0.0
    )  # [L/s]
    sol = solve_ivp(
        _make_rhs(active_flow),
        (0.0, t_end_s),
        initial_state,
        t_eval=times,
        method="RK45",
        rtol=1e-10,
        atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(f"Bioconjugation ODE integration failed: {sol.message}")
    return sol.y[:4, :].T


def _dual_wavelength_uv_invert(absorbance_248_280, pathlength_cm=PATHLENGTH_CM):
    """Invert 248 nm and 280 nm UV absorbances to [c_mAb, c_Drug] and DAR.

    Implements the dual-wavelength Beer-Lambert inversion of Hamblett et al.
    (2004, Clin. Cancer Res. 10(20):7063-7070, p. 7064).

    Parameters
    ----------
    absorbance_248_280 : array_like
        Absorbance array of shape ``(..., 2)`` with columns ``(A_248, A_280)``
        [-].
    pathlength_cm : float, optional
        Optical pathlength [cm].

    Returns
    -------
    c_mab : numpy.ndarray
        Recovered total antibody concentration [mol/L].
    c_drug : numpy.ndarray
        Recovered drug-chromophore concentration [mol/L].
    dar : numpy.ndarray
        Recovered drug-to-antibody molar ratio ``c_drug / c_mab`` [-].
    """
    abs_arr = np.asarray(absorbance_248_280, dtype=float)  # [-]
    a_248 = abs_arr[..., 0]  # [-]
    a_280 = abs_arr[..., 1]  # [-]
    det_eps = (
        EPS_MAB_248 * EPS_DRUG_280 - EPS_MAB_280 * EPS_DRUG_248
    )  # [L**2/(mol**2*cm**2)]
    c_mab = (EPS_DRUG_280 * a_248 - EPS_DRUG_248 * a_280) / (
        pathlength_cm * det_eps
    )  # [mol/L]
    c_drug = (EPS_MAB_248 * a_280 - EPS_MAB_280 * a_248) / (
        pathlength_cm * det_eps
    )  # [mol/L]
    ratio = a_248 / a_280  # [-]
    dar = (EPS_MAB_248 - ratio * EPS_MAB_280) / (
        ratio * EPS_DRUG_280 - EPS_DRUG_248
    )  # [-]
    return c_mab, c_drug, dar


@pytest.mark.unit
def test_dar2_intact_kinetics_rates_jacobians_and_invariants(tmp_path):
    """Verify Model 1A stoichiometry, mass balance, rates, Jacobians, and unit conversions."""
    # Verify both [L/(mmol*s)] and [L/(umol*s)] -> [L/(mol*s)] conversions (B1, B2).
    assert K1_MODEL_1A == pytest.approx(797.0, rel=CLOSED_FORM_RTOL)
    assert K2_MODEL_1A == pytest.approx(1476.0, rel=CLOSED_FORM_RTOL)
    assert K1_RED_TIER2 == pytest.approx(698.0, rel=CLOSED_FORM_RTOL)
    assert K3_RED_TIER2 == pytest.approx(68.1, rel=CLOSED_FORM_RTOL)
    assert K_RED_SEED_TIER2 == pytest.approx(100.0, rel=CLOSED_FORM_RTOL)

    # Verify Weggen et al. (2024, Table 2) parameter SD vs 95% CI distinction (B2).
    ci_low, ci_high = K1_DAR2_DRUG1_CI95_BOUNDS_MMOL  # [L/(mmol*s)]
    assert 0.5 * (ci_high - ci_low) == pytest.approx(
        K1_DAR2_DRUG1_CI95_HALF_MMOL, rel=CLOSED_FORM_RTOL
    )
    assert K1_DAR2_DRUG1_CI95_HALF_MMOL > K1_DAR2_DRUG1_SD_MMOL

    db_path = _write_bioconjugation_db(tmp_path)
    kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )

    assert tuple(kinetics.partic_species) == MODEL_1A_PARTICIPATING
    expected_stoich = np.array(
        [
            [-1.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 1.0],
        ]
    )  # [-], shape (3 reactions, 5 participating species)
    np.testing.assert_array_equal(kinetics.stoich_matrix, expected_stoich)

    # Exact stoichiometric mass conservation across R1, R2, and R3 (B5):
    # stoich_matrix @ mw_participating == 0 [g/mol] within the IEEE 754 ULP
    # bound of subtracting 150000 g/mol from 150297.31 g/mol (~3.3e-11 g/mol).
    mw_partic = np.array(
        [BIOCONJUGATION_DATABASE[name]["mw"] for name in MODEL_1A_PARTICIPATING],
        dtype=float,
    )  # [g/mol]
    mw_ulp_atol = 16.0 * MAB_MW * np.finfo(float).eps  # [g/mol], ~5.3e-10 g/mol
    np.testing.assert_allclose(
        kinetics.stoich_matrix @ mw_partic, np.zeros(3), atol=mw_ulp_atol
    )

    # Asymmetric non-unit concentrations [mol/L] for (mAb, mAb_1Drug,
    # mAb_2Drug, Drug, Drug_depl) to expose any axis or species-ordering swap.
    conc = np.array([3.2e-5, 1.4e-5, 0.6e-5, 1.1e-4, 0.25e-4])  # [mol/L]
    c_mab, c_mab1, _c_mab2, c_drug, _c_depl = conc  # [mol/L]

    r1_expected = K1_MODEL_1A * c_mab * c_drug  # [mol/(L*s)]
    r2_expected = K2_MODEL_1A * c_mab1 * c_drug  # [mol/(L*s)]
    r3_expected = K3_MODEL_1A * c_drug  # [mol/(L*s)]
    expected_species_rates = np.array(
        [
            -r1_expected,
            r1_expected - r2_expected,
            r2_expected,
            -r1_expected - r2_expected - r3_expected,
            r3_expected,
        ]
    )  # [mol/(L*s)]

    rates = kinetics.get_rxn_rates(
        conc, temp=REACTION_TEMP, overall_rates=True, jac=False
    )  # [mol/(L*s)]
    jac_states = kinetics.get_rxn_rates(
        conc, temp=REACTION_TEMP, jac=True
    )  # [1/s]
    jac_params = kinetics.derivatives(
        conc, temp=REACTION_TEMP, dstates=False
    )  # [mol**2/L**2 for k1, k2; mol/L for k3]

    np.testing.assert_allclose(rates, expected_species_rates, rtol=CLOSED_FORM_RTOL)

    # Exact linear invariants:
    # 1. Antibody core conservation: r_mAb + r_mAb_1Drug + r_mAb_2Drug == 0
    # 2. Drug moiety conservation: r_Drug + r_Drug_depl + r_mAb_1Drug + 2*r_mAb_2Drug == 0
    # 3. Total solution mass conservation: rates @ mw_partic == 0 [g/(L*s)]
    assert rates[0] + rates[1] + rates[2] == pytest.approx(0.0, abs=1e-20)
    assert (rates[3] + rates[4] + rates[1] + 2.0 * rates[2]) == pytest.approx(
        0.0, abs=1e-20
    )
    assert float(rates @ mw_partic) == pytest.approx(
        0.0, abs=mw_ulp_atol * float(np.max(np.abs(rates)))
    )

    expected_jac_states = np.array(
        [
            [-K1_MODEL_1A * c_drug, 0.0, 0.0, -K1_MODEL_1A * c_mab, 0.0],
            [
                K1_MODEL_1A * c_drug,
                -K2_MODEL_1A * c_drug,
                0.0,
                K1_MODEL_1A * c_mab - K2_MODEL_1A * c_mab1,
                0.0,
            ],
            [0.0, K2_MODEL_1A * c_drug, 0.0, K2_MODEL_1A * c_mab1, 0.0],
            [
                -K1_MODEL_1A * c_drug,
                -K2_MODEL_1A * c_drug,
                0.0,
                -K1_MODEL_1A * c_mab - K2_MODEL_1A * c_mab1 - K3_MODEL_1A,
                0.0,
            ],
            [0.0, 0.0, 0.0, K3_MODEL_1A, 0.0],
        ]
    )  # [1/s]
    np.testing.assert_allclose(jac_states, expected_jac_states, rtol=CLOSED_FORM_RTOL)

    # Parameter Jacobian columns follow concat_params: (k1, k2, k3, Ea1, Ea2, Ea3);
    # the rate-constant columns are indices (0, 1, 2).
    expected_jac_k = np.array(
        [
            [-c_mab * c_drug, 0.0, 0.0],
            [c_mab * c_drug, -c_mab1 * c_drug, 0.0],
            [0.0, c_mab1 * c_drug, 0.0],
            [-c_mab * c_drug, -c_mab1 * c_drug, -c_drug],
            [0.0, 0.0, c_drug],
        ]
    )  # [mol**2/L**2 for k1, k2; mol/L for k3]
    np.testing.assert_allclose(
        jac_params[:, :3], expected_jac_k, rtol=CLOSED_FORM_RTOL
    )


@pytest.mark.unit
def test_damkohler_scaling_distinguishes_bolus_plume_from_semibatch_bulk():
    """Verify Vicum et al. (2004) / Weggen et al. (2023) Eqs. (9)-(10) timescale ratios."""
    # Characteristic bimolecular reaction time tau_R = 1 / (k1 * (c_mAb + c_Drug)) [s].
    tau_rxn_bulk = 1.0 / (K1_MODEL_1A * (C_MAB_STD + C_DRUG_BULK_STD))  # [s]
    tau_rxn_plume = 1.0 / (K1_MODEL_1A * (C_MAB_STD + C_DRUG_FEED_PLUME))  # [s]

    assert tau_rxn_bulk == pytest.approx(30000.0 / (797.0 * 6.0), rel=CLOSED_FORM_RTOL)
    assert tau_rxn_plume == pytest.approx(
        30000.0 / (797.0 * 301.0), rel=CLOSED_FORM_RTOL
    )
    assert tau_rxn_bulk == pytest.approx(6.2735, rel=1e-4)
    assert tau_rxn_plume == pytest.approx(0.12506, rel=1e-4)

    vessel_mixing_times = {
        "tube_1_5mL": TAU_MIX_TUBE,  # [s]
        "GST_1_300mL": TAU_MIX_GST1,  # [s]
        "SUM_25L": TAU_MIX_SUM,  # [s]
        "GST_2_22L": TAU_MIX_GST2,  # [s]
    }
    da_blend_bulk = {
        name: tau_mix / tau_rxn_bulk for name, tau_mix in vessel_mixing_times.items()
    }  # [-]
    da_blend_plume = {
        name: tau_mix / tau_rxn_plume for name, tau_mix in vessel_mixing_times.items()
    }  # [-]

    # Verify all tabulated macromixing blend Damkohler values from Section 5.
    assert da_blend_bulk["tube_1_5mL"] == pytest.approx(0.0797, rel=1e-2)
    assert da_blend_bulk["GST_1_300mL"] == pytest.approx(1.498, rel=1e-2)
    assert da_blend_bulk["SUM_25L"] == pytest.approx(2.805, rel=1e-2)
    assert da_blend_bulk["GST_2_22L"] == pytest.approx(5.133, rel=1e-2)

    assert da_blend_plume["tube_1_5mL"] == pytest.approx(4.00, rel=1e-2)
    assert da_blend_plume["GST_1_300mL"] == pytest.approx(75.16, rel=1e-2)
    assert da_blend_plume["SUM_25L"] == pytest.approx(140.7, rel=1e-2)
    assert da_blend_plume["GST_2_22L"] == pytest.approx(257.5, rel=1e-2)

    # Verify Baldyga-Bourne / Vicum et al. (2004, Eq. 5) micromixing engulfment
    # time tau_E = 17.24 * sqrt(nu / epsilon) [s] and engulfment Damkohler
    # number Da_E = tau_E / tau_R_plume [-] (Weggen et al., 2023, Eqs. 9-10).
    vessel_power_per_vol = {
        "GST_1_300mL": POWER_PER_VOL_GST1,  # [W/m**3]
        "SUM_25L": POWER_PER_VOL_SUM,  # [W/m**3]
        "GST_2_22L": POWER_PER_VOL_GST2,  # [W/m**3]
    }
    tau_engulfment = {
        name: ENGULFMENT_COEFF
        * np.sqrt(KINEMATIC_VISC_WATER / (p_per_v / WATER_DENSITY_APPROX))
        for name, p_per_v in vessel_power_per_vol.items()
    }  # [s]
    da_engulfment_plume = {
        name: tau_e / tau_rxn_plume for name, tau_e in tau_engulfment.items()
    }  # [-]

    assert tau_engulfment["GST_1_300mL"] == pytest.approx(0.279, rel=1e-2)
    assert tau_engulfment["SUM_25L"] == pytest.approx(0.112, rel=1e-2)
    assert tau_engulfment["GST_2_22L"] == pytest.approx(0.349, rel=1e-2)

    assert da_engulfment_plume["GST_1_300mL"] == pytest.approx(2.23, rel=1e-2)
    assert da_engulfment_plume["SUM_25L"] == pytest.approx(0.896, rel=1e-2)
    assert da_engulfment_plume["GST_2_22L"] == pytest.approx(2.79, rel=1e-2)

    # Monotone ordering of vessel blend Damkohler matches the CFD Delta-DAR
    # ranking reported in Weggen et al. (2023, Sec. 3.2.3: tube < GST-1 < SUM < GST-2).
    assert (
        da_blend_bulk["tube_1_5mL"]
        < da_blend_bulk["GST_1_300mL"]
        < da_blend_bulk["SUM_25L"]
        < da_blend_bulk["GST_2_22L"]
    )


@pytest.mark.unit
def test_dual_wavelength_uv_vis_contract_and_in_situ_mcr_deconvolution():
    """Verify Hamblett et al. (2004) UV-Vis DAR inversion, optical rank, and MCR/PCR PAT contracts."""
    c_mab_true = 3.5e-5  # [mol/L], 5.25 g/L intact antibody
    dar_targets = np.array([0.0, 1.0, 1.9, 2.0, 4.0, 8.0])  # [-]
    c_drug_conj_true = c_mab_true * dar_targets  # [mol/L]

    # Forward Beer-Lambert absorbance prediction at 248 nm and 280 nm using a
    # short-pathlength (0.10 cm = 1 mm) micro-volume / flow cell so undiluted
    # 5.25 g/L mAb stays within linear photodiode bounds (A_280 <= 1.5 AU) (B4).
    a_248_short = SHORT_PATHLENGTH_CM * (
        EPS_MAB_248 * c_mab_true + EPS_DRUG_248 * c_drug_conj_true
    )  # [-]
    a_280_short = SHORT_PATHLENGTH_CM * (
        EPS_MAB_280 * c_mab_true + EPS_DRUG_280 * c_drug_conj_true
    )  # [-]
    assert np.all(a_280_short <= MAX_LINEAR_ABSORBANCE_AU)
    spectra_desalted = np.column_stack((a_248_short, a_280_short))  # [-]

    c_mab_rec, c_drug_rec, dar_rec = _dual_wavelength_uv_invert(
        spectra_desalted, pathlength_cm=SHORT_PATHLENGTH_CM
    )
    np.testing.assert_allclose(
        c_mab_rec, np.full_like(dar_targets, c_mab_true), rtol=CLOSED_FORM_RTOL
    )
    np.testing.assert_allclose(
        c_drug_rec, c_drug_conj_true, rtol=CLOSED_FORM_RTOL, atol=1e-18
    )
    np.testing.assert_allclose(dar_rec, dar_targets, rtol=CLOSED_FORM_RTOL, atol=1e-14)

    # 1. Closed-batch aliasing theorem (Schiemer et al., 2023, doi:10.1002/bit.28428):
    # if unconjugated and conjugated payload had identical extinction at 248 and
    # 280 nm, closed-batch drug-moiety conservation keeps (A_248, A_280) constant
    # across all reaction extents.
    c_drug_total = 4.0 * c_mab_true  # [mol/L], fixed 4x initial payload charge
    c_free_drug = c_drug_total - c_drug_conj_true[:4]  # [mol/L], for DAR 0..2
    a_248_aliased = SHORT_PATHLENGTH_CM * (
        EPS_MAB_248 * c_mab_true + EPS_DRUG_248 * (c_drug_conj_true[:4] + c_free_drug)
    )  # [-]
    np.testing.assert_allclose(
        a_248_aliased, np.full(4, a_248_aliased[0]), rtol=CLOSED_FORM_RTOL
    )

    # 2. Optical rank theorem for intact species (mAb_0, mAb_1, mAb_2, Drug, Drug_depl):
    # Under additive bound-chromophore spectra (eps_mAb1 = eps_mAb + eps_conj_site,
    # eps_mAb2 = eps_mAb + 2 * eps_conj_site), 2 * eps_mAb1 == eps_mAb0 + eps_mAb2,
    # so the 5-species extinction matrix has column rank 4 (cannot separate mAb_1
    # from 0.5*(mAb_0 + mAb_2) purely optically without the kinetic ODE observer),
    # whereas the 4-coordinate basis (mAb_total, conj_sites, Drug, Drug_depl) has
    # full rank 4 when Delta_eps_conj != 0 and Delta_eps_depl != 0 (B4).
    eps_mab = np.array([EPS_MAB_248, EPS_MAB_280, 1.20e3, 0.15e3])  # [L/(mol*cm)]
    eps_conj_site = np.array([EPS_DRUG_248, EPS_DRUG_280, 8.20e3, 4.10e3])  # [L/(mol*cm)]
    eps_drug_free = np.array([EPS_DRUG_248, EPS_DRUG_280, 1.45e4, 9.80e3])  # [L/(mol*cm)]
    eps_drug_depl = np.array([1.42e4, 1.20e3, 2.10e3, 0.60e3])  # [L/(mol*cm)]

    eps_5species = np.vstack(
        (
            eps_mab,
            eps_mab + eps_conj_site,
            eps_mab + 2.0 * eps_conj_site,
            eps_drug_free,
            eps_drug_depl,
        )
    )  # [L/(mol*cm)], shape (5 species, 4 wavelengths)
    eps_4chromophores = np.vstack(
        (eps_mab, eps_conj_site, eps_drug_free, eps_drug_depl)
    )  # [L/(mol*cm)], shape (4 chromophore coordinates, 4 wavelengths)
    assert np.linalg.matrix_rank(eps_5species) == 4
    assert np.linalg.matrix_rank(eps_4chromophores) == 4
    np.testing.assert_allclose(
        np.array([-1.0, 2.0, -1.0, 0.0, 0.0]) @ eps_5species,
        np.zeros(4),
        atol=1e-12,
    )

    # 3. Multi-wavelength spectral resolution including Drug_depl across 4
    # diagnostic wavelengths (248, 280, 310, 342 nm; origin='synthetic' at
    # 310/342 nm): exercise MultipleCurveResolution (S = C @ A) and
    # PCR_calibration on all 3 independent dynamic reaction species
    # (H1_1, Drug, Drug_depl) after subtracting the constant total heavy-chain
    # protein core background c_H_total * eps_H0 (B4).
    times_s = np.linspace(0.0, 900.0, 25)  # [s]
    traj = _solve_dar2_heavy_chain_experiment(
        (K1_DAR2_DRUG1, K_DRUG_DRUG1),
        times_s,
        c_h0_init_mol_l=6.0e-5,
        c_drug_init_mol_l=1.5e-4,
    )  # [mol/L], columns (H0_1, H1_1, Drug, Drug_depl)
    conc_three = traj[:, [1, 2, 3]]  # [mol/L], shape (25, 3): (H1_1, Drug, Drug_depl)
    pure_absorptivities = np.vstack(
        (
            SHORT_PATHLENGTH_CM * eps_conj_site,
            SHORT_PATHLENGTH_CM * eps_drug_free,
            SHORT_PATHLENGTH_CM * eps_drug_depl,
        )
    )  # [L/mol], effective pathlength-scaled absorptivities (l * eps), shape (3, 4)
    spectral_matrix = conc_three @ pure_absorptivities  # [-], shape (25, 4)

    mcr_true_params = np.array([K1_DAR2_DRUG1, K_DRUG_DRUG1])  # [L/(mol*s)], [1/s]
    mcr = MultipleCurveResolution(
        _solve_dar2_heavy_chain_experiment,
        param_seed=mcr_true_params,
        time_data=times_s,
        y_spectra=spectral_matrix,
        measured_ind=[1, 2, 3],
        kwargs_fun={"c_h0_init_mol_l": 6.0e-5, "c_drug_init_mol_l": 1.5e-4},
    )
    mcr_obj = mcr.get_objective(mcr_true_params, out_array=False)
    assert mcr_obj == pytest.approx(0.0, abs=1e-20)
    np.testing.assert_allclose(mcr.epsilon_mcr, pure_absorptivities, rtol=1e-9)
    y_runs, _resid_runs, _weighted_resid, absorptivity_pure = mcr.get_global_analysis(
        mcr_true_params
    )
    np.testing.assert_allclose(absorptivity_pure, pure_absorptivities, rtol=1e-9)
    np.testing.assert_allclose(y_runs[0], spectral_matrix, rtol=1e-9)

    pcr = PCR_calibration(spectral_matrix, num_comp=3, standardize=True, snv=False)
    pcr.get_regression(conc_three, num_comp=3)
    conc_pcr_pred = pcr.predict(spectral_matrix)  # [mol/L]
    np.testing.assert_allclose(conc_pcr_pred, conc_three, rtol=1e-7, atol=1e-10)


@pytest.mark.unit
def test_weggen_2024_dar2_batch_and_fedbatch_estimation_and_calibration_record(
    tmp_path,
):
    """Fit the Weggen et al. (2024) Simple 1k + sink model and round-trip its record."""
    # Experiment 1: 1.0 mL batch run (c_H0_1,0 = 6.0e-5 mol/L, c_Drug,0 = 1.5e-4 mol/L).
    # Experiment 2: 4.0 mL fed-batch run with 600 s syringe-pump dosing of 1.5 mM
    # payload stock at 6.667e-7 L/s (total 0.4 mL added = 6.0e-7 mol payload).
    batch_times = np.array([0.0, 30.0, 60.0, 120.0, 240.0, 480.0, 900.0])  # [s]
    fedbatch_times = np.array(
        [0.0, 60.0, 180.0, 300.0, 450.0, 600.0, 750.0, 900.0]
    )  # [s]
    true_params = np.array([K1_DAR2_DRUG1, K_DRUG_DRUG1])  # [L/(mol*s)], [1/s]

    batch_kwargs = {
        "c_h0_init_mol_l": 6.0e-5,  # [mol/L]
        "c_drug_init_mol_l": 1.5e-4,  # [mol/L]
        "v0_l": 1.0e-3,  # [L]
        "feed_rate_l_s": 0.0,  # [L/s]
        "feed_duration_s": 0.0,  # [s]
        "c_drug_feed_mol_l": 0.0,  # [mol/L]
    }
    fedbatch_kwargs = {
        "c_h0_init_mol_l": 6.0e-5,  # [mol/L]
        "c_drug_init_mol_l": 0.0,  # [mol/L]
        "v0_l": 4.0e-3,  # [L]
        "feed_rate_l_s": 4.0e-4 / 600.0,  # [L/s]
        "feed_duration_s": 600.0,  # [s]
        "c_drug_feed_mol_l": 1.5e-3,  # [mol/L]
    }

    batch_obs = _solve_dar2_heavy_chain_experiment(
        true_params, batch_times, **batch_kwargs
    )  # [mol/L]
    fedbatch_obs = _solve_dar2_heavy_chain_experiment(
        true_params, fedbatch_times, **fedbatch_kwargs
    )  # [mol/L]

    output_names = ("H0_1", "H1_1", "Drug", "Drug_depl")
    measured_species = ("H0_1", "H1_1", "Drug")
    uncertainty_mol_l = 1.863e-6  # [mol/L], RMSE_CV of Weggen et al. (2024), Table 2

    experiments = {
        "batch_1mL": Experiment(
            {
                name: Measurement(
                    name,
                    batch_times,
                    batch_obs[:, idx],
                    uncertainty=np.full(batch_times.size, uncertainty_mol_l),
                    units="mol/L",
                    basis="molar concentration",
                    x_units="s",
                )
                for idx, name in enumerate(measured_species)
            },
            kwargs=batch_kwargs,
        ),
        "fedbatch_4mL_600s": Experiment(
            {
                name: Measurement(
                    name,
                    fedbatch_times,
                    fedbatch_obs[:, idx],
                    uncertainty=np.full(fedbatch_times.size, uncertainty_mol_l),
                    units="mol/L",
                    basis="molar concentration",
                    x_units="s",
                )
                for idx, name in enumerate(measured_species)
            },
            kwargs=fedbatch_kwargs,
        ),
    }

    param_seed = np.array([180.0, 1.8e-3])  # [L/(mol*s)], [1/s]
    estimator = ParameterEstimation(
        _solve_dar2_heavy_chain_experiment,
        param_seed=param_seed,
        x_data=experiments,
        name_params=["k1", "k_drug"],
        output_names=output_names,
    )
    fitted, covariance, info = estimator.estimate_parameters(
        method="LM", verbose=False
    )

    assert "jac" in info and "fun" in info
    np.testing.assert_allclose(fitted, true_params, rtol=ODE_FIT_RTOL)
    assert covariance.shape == (2, 2)
    assert np.all(np.diag(covariance) >= 0.0)
    intervals = estimator.get_confidence_intervals(verbose=False)
    assert set(intervals) == {"k1", "k_drug"}

    # Verify confidence-interval half-widths against independent Student-t
    # calculation from covariance diagonal (NB2).
    n_obs_total = (batch_times.size + fedbatch_times.size) * len(measured_species)
    dof = n_obs_total - len(true_params)
    t_val = float(student_t.ppf(0.975, dof))  # [-]
    expected_half_widths = t_val * np.sqrt(np.diag(covariance))  # [L/(mol*s)], [1/s]
    for idx, pname in enumerate(("k1", "k_drug")):
        low, high = intervals[pname]
        assert 0.5 * (high + low) == pytest.approx(fitted[idx], rel=1e-10)
        assert 0.5 * (high - low) == pytest.approx(
            expected_half_widths[idx], rel=1e-6, abs=1e-18
        )

    # Compare Fisher Information Matrix F = J^T W J across:
    # (1) a single batch run,
    # (2) a duplicated two-batch baseline (which scales F by 2x and det(F) by
    #     2^2 = 4x while leaving the normalized off-diagonal parameter
    #     collinearity |F_01| / sqrt(F_00 * F_11) unchanged), and
    # (3) the combined batch + 600 s fed-batch design, which increases det(F)
    #     over batch-only by >2.2x AND flips the sign of the cross-sensitivity
    #     between k1 and k_drug (F_01 > 0 in batch vs F_01 < 0 in fed-batch),
    #     reducing normalized parameter collinearity by >6x compared to a
    #     duplicated two-batch design (Weggen et al., 2024, Sec. 3.3.2) (NB2).
    batch_only_est = ParameterEstimation(
        _solve_dar2_heavy_chain_experiment,
        param_seed=true_params,
        x_data={"batch_1mL": experiments["batch_1mL"]},
        name_params=["k1", "k_drug"],
        output_names=output_names,
    )
    two_batch_est = ParameterEstimation(
        _solve_dar2_heavy_chain_experiment,
        param_seed=true_params,
        x_data={
            "batch_1mL_a": experiments["batch_1mL"],
            "batch_1mL_b": experiments["batch_1mL"],
        },
        name_params=["k1", "k_drug"],
        output_names=output_names,
    )
    jac_batch = batch_only_est.get_gradient(true_params, out_array=True)
    jac_two_batch = two_batch_est.get_gradient(true_params, out_array=True)
    jac_combined = estimator.get_gradient(true_params, out_array=True)
    fim_batch = jac_batch @ jac_batch.T
    fim_two_batch = jac_two_batch @ jac_two_batch.T
    fim_combined = jac_combined @ jac_combined.T
    assert np.linalg.det(fim_two_batch) == pytest.approx(
        4.0 * np.linalg.det(fim_batch), rel=1e-10
    )
    assert np.linalg.det(fim_combined) > 2.2 * np.linalg.det(fim_batch)
    collinearity_two_batch = abs(fim_two_batch[0, 1]) / np.sqrt(
        fim_two_batch[0, 0] * fim_two_batch[1, 1]
    )  # [-]
    collinearity_combined = abs(fim_combined[0, 1]) / np.sqrt(
        fim_combined[0, 0] * fim_combined[1, 1]
    )  # [-]
    assert collinearity_combined < 0.2 * collinearity_two_batch

    # Capture CalibrationRecord with explicit literature + synthetic provenance.
    # Pin the SHA-256 digest of the canonical JSON serialization of the
    # generated synthetic observation payload on the synthetic DataSource (NB3).
    synthetic_payload_bytes = json.dumps(
        {
            "generator": "_solve_dar2_heavy_chain_experiment",
            "true_params": {
                "k1_l_mol_s": float(true_params[0]),
                "k_drug_1_s": float(true_params[1]),
            },
            "uncertainty_mol_l": uncertainty_mol_l,
            "batch_1mL": {
                "times_s": batch_times.tolist(),
                "obs_mol_l": np.round(batch_obs, 15).tolist(),
            },
            "fedbatch_4mL_600s": {
                "times_s": fedbatch_times.tolist(),
                "obs_mol_l": np.round(fedbatch_obs, 15).tolist(),
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    synthetic_sha256 = hashlib.sha256(synthetic_payload_bytes).hexdigest()
    assert len(synthetic_sha256) == 64

    sources = [
        DataSource(
            "weggen_2024_table2",
            "literature",
            "Published Open Access under CC BY 4.0 in Frontiers in Bioengineering "
            "and Biotechnology 12:1403644 (PMC11274341)",
            description=(
                "Published rate constants, parameter standard deviation, and "
                "95% confidence interval for the DAR 2 Simple (1k + k_drug) "
                "heavy-chain conjugation model of ADC1 + Drug1: "
                "k1 = 0.251 L/(mmol*s) (sigma_1 = 0.023, 95% CI [0.204, 0.298] "
                "L/(mmol*s); 251.0 L/(mol*s)) and k_drug = 0.001 1/s "
                "(Table 2 and Section 3.1)"
            ),
            citation="Weggen et al. (2024), Front. Bioeng. Biotechnol. 12:1403644",
            uri="https://doi.org/10.3389/fbioe.2024.1403644",
            role="literature parameters",
        ),
        DataSource(
            "wp15_m6_synthetic_forward_trajectories",
            "synthetic",
            "Generated deterministically in tests/test_bioconjugation_case_selection.py "
            "and documented in doc/review_notes/bioconjugation_case_selection.md; "
            "synthetic forward-simulated trajectories from published Weggen et al. "
            "(2024) parameters for M6 contract verification, not an empirical dataset",
            description=(
                "Two-piece RK45 forward simulation of the Weggen et al. (2024) "
                "Simple 1k + k_drug model for 1 mL batch and 4 mL 600 s "
                "syringe-pump fed-batch experiments"
            ),
            uri=REVIEW_NOTE.relative_to(REPO_ROOT).as_posix(),
            sha256=synthetic_sha256,
            role="calibration",
        ),
    ]
    transformations = [
        Transformation(
            "Converted published k1 from [L/(mmol*s)] to PharmaPy [L/(mol*s)] "
            "by multiplying by 1000 mmol/mol",
            source="weggen_2024_table2",
            parameters={"mmol_per_mol": {"value": MMOL_PER_MOL, "units": "mmol/mol"}},
        ),
        Transformation(
            "Evaluated H0_1, H1_1, and Drug molar concentrations [mol/L] on the "
            "batch and fed-batch sampling grids with per-sample uncertainty set "
            "to the published cross-validation RMSE (1.863e-6 mol/L)",
            fields=("H0_1", "H1_1", "Drug"),
            source="wp15_m6_synthetic_forward_trajectories",
        ),
    ]
    model = ModelRevision(
        "_solve_dar2_heavy_chain_experiment",
        description=(
            "Isothermal batch and fed-batch heavy-chain DAR 2 Simple (1k + "
            "k_drug) bioconjugation ODE model of Weggen et al. (2024), Eqs. 1-7"
        ),
    )

    record = CalibrationRecord.from_estimation(
        estimator,
        experiments,
        case="wp15-m6-dar2-bioconjugation-verification",
        parameter_units={"k1": "L/(mol*s)", "k_drug": "1/s"},
        model=model,
        sources=sources,
        transformations=transformations,
        description=(
            "WP1.5 M6 verification fit of the Weggen et al. (2024) DAR 2 "
            "Simple (1k + k_drug) conjugation model across batch and fed-batch "
            "operating modes; synthetic forward trajectory from published "
            "parameters, not an empirical calibration"
        ),
    )

    record_path = tmp_path / "wp15_m6_bioconjugation_record.json"
    record.save(record_path)
    reloaded = CalibrationRecord.load(record_path)

    assert reloaded == record
    assert [src.origin for src in reloaded.sources] == ["literature", "synthetic"]
    assert reloaded.sources[1].sha256 == synthetic_sha256
    assert "not an empirical" in reloaded.description

    replayed = reloaded.rerun(
        _solve_dar2_heavy_chain_experiment, rtol=REPLAY_RTOL, atol=0.0
    )
    np.testing.assert_allclose(
        replayed.params_convg, fitted, rtol=REPLAY_RTOL, atol=0.0
    )


@pytest.mark.unit
def test_wp4_four_step_sequence_mass_balance_and_storage_contract():
    """Verify WP4 STN/RTN four-step solution mass fractions, continuity, and hold policies (B3)."""
    # Canonical WP4 four-task bioconjugation sequence from Section 8.2 of
    # doc/review_notes/bioconjugation_case_selection.md (origin='synthetic'
    # scheduling design parameters anchored to bench/pilot operating windows).
    # Basis: total liquid solution mass fraction [kg/kg] at uniform dilute
    # buffer density BUFFER_DENSITY = 1010.5 kg/m**3 (0.001 m**3/L).
    wp4_tasks = (
        {
            "task_id": "T1_reduction",
            "inputs_kg_kg": {
                "S0_native_mAb": 19.60 / 20.00,  # [kg/kg], 0.9800
                "S_TCEP_reductant": 0.40 / 20.00,  # [kg/kg], 0.0200
            },
            "outputs_kg_kg": {
                "S1_reduced_mAb": 1.0000,  # [kg/kg]
            },
            "vol_in_l": 20.00,  # [L]
            "vol_out_main_l": 20.00,  # [L]
            "vol_out_waste_l": 0.00,  # [L]
            "protein_in_kg": 0.200,  # [kg]
            "protein_out_kg": 0.200,  # [kg]
            "duration_s": 5400.0,  # [s]
            "duration_range_s": (3600.0, 7200.0),  # [s]
            "storage_policy": "FW",
            "t_hold_max_s": 1800.0,  # [s]
        },
        {
            "task_id": "T2_conjugation",
            "inputs_kg_kg": {
                "S1_reduced_mAb": 20.00 / 22.475,  # [kg/kg], 0.8899
                "S_payload_DMSO": 2.25 / 22.475,  # [kg/kg], 0.1001 (10% v/v of 22.25 L rxn)
                "S_NAC_quench": 0.225 / 22.475,  # [kg/kg], 0.0100
            },
            "outputs_kg_kg": {
                "S2_crude_ADC": 1.0000,  # [kg/kg]
            },
            "vol_in_l": 22.475,  # [L]
            "vol_out_main_l": 22.475,  # [L]
            "vol_out_waste_l": 0.00,  # [L]
            "protein_in_kg": 0.200,  # [kg]
            "protein_out_kg": 0.200,  # [kg]
            "duration_s": 2700.0,  # [s]
            "duration_range_s": (1800.0, 3600.0),  # [s]
            "storage_policy": "FW",
            "t_hold_max_s": 3600.0,  # [s]
        },
        {
            "task_id": "T3_desalting_ufdf",
            "inputs_kg_kg": {
                "S2_crude_ADC": 22.475 / 162.475,  # [kg/kg], 0.1383
                "S_diafiltration_buffer": 140.00 / 162.475,  # [kg/kg], 0.8617 (7 diavolumes)
            },
            "outputs_kg_kg": {
                "S3_desalted_ADC": 20.00 / 162.475,  # [kg/kg], 0.1231
                "S_ufdf_permeate_waste": 142.475 / 162.475,  # [kg/kg], 0.8769
            },
            "vol_in_l": 162.475,  # [L]
            "vol_out_main_l": 20.00,  # [L]
            "vol_out_waste_l": 142.475,  # [L]
            "protein_in_kg": 0.200,  # [kg]
            "protein_out_kg": 0.196,  # [kg], 98.0% TFF retentate step recovery
            "duration_s": 4500.0,  # [s]
            "duration_range_s": (3600.0, 5400.0),  # [s]
            "storage_policy": "UIS",
            "t_hold_max_s": 86400.0,  # [s], configurable multi-day FW bound
        },
        {
            "task_id": "T4_purification",
            "inputs_kg_kg": {
                "S3_desalted_ADC": 20.00 / 80.00,  # [kg/kg], 0.2500
                "S_chrom_mobile_phase": 60.00 / 80.00,  # [kg/kg], 0.7500
            },
            "outputs_kg_kg": {
                "S4_purified_ADC_DS": 18.00 / 80.00,  # [kg/kg], 0.2250
                "S_chrom_strip_waste": 62.00 / 80.00,  # [kg/kg], 0.7750
            },
            "vol_in_l": 80.00,  # [L]
            "vol_out_main_l": 18.00,  # [L]
            "vol_out_waste_l": 62.00,  # [L]
            "protein_in_kg": 0.196,  # [kg]
            "protein_out_kg": 0.180,  # [kg], 90.0% overall four-step protein yield
            "duration_s": 3600.0,  # [s]
            "duration_range_s": (2700.0, 5400.0),  # [s]
            "storage_policy": "UIS",
            "t_hold_max_s": np.inf,  # [s]
        },
    )

    for task in wp4_tasks:
        assert sum(task["inputs_kg_kg"].values()) == pytest.approx(
            1.0, abs=CLOSED_FORM_RTOL
        )
        assert sum(task["outputs_kg_kg"].values()) == pytest.approx(
            1.0, abs=CLOSED_FORM_RTOL
        )
        assert (
            task["vol_out_main_l"] + task["vol_out_waste_l"]
        ) == pytest.approx(task["vol_in_l"], rel=CLOSED_FORM_RTOL)
        low_s, high_s = task["duration_range_s"]
        assert low_s <= task["duration_s"] <= high_s
        assert task["storage_policy"] in {"FW", "UIS"}

    # Verify inter-task state continuity of main intermediate liquid streams
    # and protein solute mass across T1 -> T2 -> T3 -> T4.
    assert wp4_tasks[0]["vol_out_main_l"] == pytest.approx(
        wp4_tasks[1]["vol_in_l"] * wp4_tasks[1]["inputs_kg_kg"]["S1_reduced_mAb"],
        rel=CLOSED_FORM_RTOL,
    )
    assert wp4_tasks[1]["vol_out_main_l"] == pytest.approx(
        wp4_tasks[2]["vol_in_l"] * wp4_tasks[2]["inputs_kg_kg"]["S2_crude_ADC"],
        rel=CLOSED_FORM_RTOL,
    )
    assert wp4_tasks[2]["vol_out_main_l"] == pytest.approx(
        wp4_tasks[3]["vol_in_l"] * wp4_tasks[3]["inputs_kg_kg"]["S3_desalted_ADC"],
        rel=CLOSED_FORM_RTOL,
    )
    for idx in range(len(wp4_tasks) - 1):
        assert wp4_tasks[idx]["protein_out_kg"] == pytest.approx(
            wp4_tasks[idx + 1]["protein_in_kg"], rel=CLOSED_FORM_RTOL
        )
    overall_protein_yield = (
        wp4_tasks[-1]["protein_out_kg"] / wp4_tasks[0]["protein_in_kg"]
    )  # [-]
    assert overall_protein_yield == pytest.approx(0.90, rel=CLOSED_FORM_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_native_batch_and_semibatch_reactors_execute_dar2_conjugation(tmp_path):
    """Run native BatchReactor and SemibatchReactor solves and reproduce Section 7.2 blockers (B6)."""
    pytest.importorskip("assimulo")
    db_path = _write_bioconjugation_db(tmp_path)

    # 1. BatchReactor native solve at 5 g/L mAb (3.333e-5 mol/L) and 5x payload
    # in GST-1 (0.300 L = 3.0e-4 m**3).
    gst1_vol_m3 = 3.0e-4  # [m**3], 0.300 L lab stirred tank GST-1
    gst1_vol_l = 0.300  # [L]
    batch_conc_init = [
        C_MAB_STD,
        0.0,
        0.0,
        C_DRUG_BULK_STD,
        0.0,
        0.0,
    ]  # [mol/L], mAb, mAb_1Drug, mAb_2Drug, Drug, Drug_depl, buffer
    time_grid = np.linspace(0.0, 300.0, 31)  # [s]

    # 1a. Reproduce Section 7.2 blocker: default CVode tolerance (atol = 1e-6
    # mol/L) on micromolar antibody states (~3.33e-5 mol/L) allows O(1e-6)
    # mol/L negative-concentration undershoot and unphysical DAR > 2 overshoot.
    batch_default_atol = BatchReactor(isothermal=True)
    batch_default_atol.Kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )
    batch_default_atol.Phases = LiquidPhase(
        db_path,
        temp=REACTION_TEMP,
        vol=gst1_vol_m3,
        mole_conc=batch_conc_init,
        name_solv="buffer",
    )
    batch_default_atol.solve_unit(
        time_grid=time_grid,
        eval_sens=False,
        verbose=False,
    )
    prof_default = batch_default_atol.result.mole_conc  # [mol/L], shape (31, 6)
    dar_default_final = (
        prof_default[-1, 1] + 2.0 * prof_default[-1, 2]
    ) / (
        prof_default[-1, 0] + prof_default[-1, 1] + prof_default[-1, 2]
    )  # [-]
    assert np.min(prof_default[:, :3]) < -1.0e-7
    assert dar_default_final > 2.01

    # 1b. Verified workaround with micromolar-scaled tolerances (atol = 1e-11
    # mol/L) and forward sensitivity integration (eval_sens=True).
    batch_kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )
    batch_reactor = BatchReactor(isothermal=True)
    batch_reactor.Kinetics = batch_kinetics
    batch_reactor.Phases = LiquidPhase(
        db_path,
        temp=REACTION_TEMP,
        vol=gst1_vol_m3,
        mole_conc=batch_conc_init,
        name_solv="buffer",
    )
    micromolar_sundials_opts = {"rtol": 1e-8, "atol": 1e-11}
    _time_out, _states_out, sensit_out = batch_reactor.solve_unit(
        time_grid=time_grid,
        eval_sens=True,
        verbose=False,
        sundials_opts=micromolar_sundials_opts,
    )

    prof = batch_reactor.result.mole_conc  # [mol/L], shape (31, 5)
    mab_total = prof[:, 0] + prof[:, 1] + prof[:, 2]  # [mol/L]
    drug_moiety_total = (
        prof[:, 3] + prof[:, 4] + prof[:, 1] + 2.0 * prof[:, 2]
    )  # [mol/L]
    dar_final = (prof[-1, 1] + 2.0 * prof[-1, 2]) / mab_total[-1]  # [-]

    np.testing.assert_allclose(mab_total, C_MAB_STD, rtol=1e-6)
    np.testing.assert_allclose(drug_moiety_total, C_DRUG_BULK_STD, rtol=1e-6)
    assert np.all(prof[:, :5] >= -1e-11)
    assert 1.98 < dar_final <= 2.0 + 1e-6

    # Verify BatchReactor forward sensitivity output (eval_sens=True) is
    # populated on batch_reactor.sensit (and returned as sensit_out), finite,
    # and non-zero with respect to k1, k2, k3 (B6).
    assert hasattr(batch_reactor, "sensit")
    sens = np.asarray(batch_reactor.sensit, dtype=float)
    np.testing.assert_array_equal(sens, np.asarray(sensit_out, dtype=float))
    assert sens.shape == (6, len(time_grid), 5)
    assert np.all(np.isfinite(sens))
    assert np.linalg.norm(sens[:3, :, :]) > 0.0

    # 2. SemibatchReactor native solve with 60 s payload dosing into 0.300 L
    # (3.0e-4 m**3) inside a 0.500 L (5.0e-4 m**3) tank.
    tank_capacity_m3 = 5.0e-4  # [m**3], 0.500 L vessel capacity
    feed_time_s = 60.0  # [s], standard GST-1/GST-2 addition time, Table 2
    feed_vol_m3 = 5.0e-6  # [m**3], 5 mL (0.005 L) payload stock addition
    feed_vol_l = 0.005  # [L]
    feed_flow_m3_s = feed_vol_m3 / feed_time_s  # [m**3/s]
    feed_flow_l_s = feed_vol_l / feed_time_s  # [L/s]
    total_drug_moles = gst1_vol_l * C_DRUG_BULK_STD  # [mol]
    feed_drug_conc = total_drug_moles / feed_vol_l  # [mol/L], 10 mM stock

    # 2a. Reproduce Section 7.2 blocker (#486): SemibatchReactor.solve_unit
    # raises NotImplementedError when eval_sens=True is requested.
    semibatch_sens_blocker = SemibatchReactor(
        vol_tank=tank_capacity_m3, isothermal=True
    )
    semibatch_sens_blocker.Kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )
    semibatch_sens_blocker.Phases = LiquidPhase(
        db_path,
        temp=REACTION_TEMP,
        vol=gst1_vol_m3,
        mole_conc=[C_MAB_STD, 0.0, 0.0, 0.0, 0.0, 0.0],
        name_solv="buffer",
    )
    semibatch_sens_blocker.Inlet = LiquidStream(
        db_path,
        temp=REACTION_TEMP,
        vol_flow=feed_flow_m3_s,
        mole_conc=[0.0, 0.0, 0.0, feed_drug_conc, 0.0, 0.0],
        name_solv="buffer",
    )
    with pytest.raises(NotImplementedError):
        semibatch_sens_blocker.solve_unit(
            runtime=feed_time_s,
            eval_sens=True,
            verbose=False,
            sundials_opts=micromolar_sundials_opts,
        )

    # 2b. Verified SemibatchReactor state integration (eval_sens=False) and
    # trajectory comparison against an independent SciPy RK45 reference solve.
    semibatch = SemibatchReactor(vol_tank=tank_capacity_m3, isothermal=True)
    semibatch.Kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )
    semibatch.Phases = LiquidPhase(
        db_path,
        temp=REACTION_TEMP,
        vol=gst1_vol_m3,
        mole_conc=[C_MAB_STD, 0.0, 0.0, 0.0, 0.0, 0.0],
        name_solv="buffer",
    )
    semibatch.Inlet = LiquidStream(
        db_path,
        temp=REACTION_TEMP,
        vol_flow=feed_flow_m3_s,
        mole_conc=[0.0, 0.0, 0.0, feed_drug_conc, 0.0, 0.0],
        name_solv="buffer",
    )
    semibatch.solve_unit(
        runtime=feed_time_s,
        eval_sens=False,
        verbose=False,
        sundials_opts=micromolar_sundials_opts,
    )

    vol_final_m3 = semibatch.result.vol[-1]  # [m**3]
    vol_final_l = vol_final_m3 * 1000.0  # [L]
    sb_conc_final = semibatch.result.mole_conc[-1]  # [mol/L]
    assert vol_final_m3 == pytest.approx(gst1_vol_m3 + feed_vol_m3, rel=1e-5)
    assert (
        vol_final_l * (sb_conc_final[0] + sb_conc_final[1] + sb_conc_final[2])
    ) == pytest.approx(gst1_vol_l * C_MAB_STD, rel=1e-5)
    assert (
        vol_final_l
        * (
            sb_conc_final[3]
            + sb_conc_final[4]
            + sb_conc_final[1]
            + 2.0 * sb_conc_final[2]
        )
    ) == pytest.approx(total_drug_moles, rel=1e-5)

    def _scipy_semibatch_model_1a_rhs(_time_s, state):
        """Evaluate Model 1A fed-batch ODEs for independent SciPy verification.

        Parameters
        ----------
        _time_s : float
            Current integration time [s].
        state : sequence of float
            ``(c_mab, c_mab1, c_mab2, c_drug, c_depl, vol_l)`` in
            ``[mol/L x 5, L]``.

        Returns
        -------
        list of float
            Time derivatives ``[dc_mab, dc_mab1, dc_mab2, dc_drug, dc_depl,
            dvol]`` in ``[mol/(L*s) x 5, L/s]``.
        """
        c_mab, c_mab1, c_mab2, c_drug, c_depl, vol_l = state  # [mol/L x 5, L]
        dilution_s = feed_flow_l_s / vol_l  # [1/s]
        r1 = K1_MODEL_1A * c_mab * c_drug  # [mol/(L*s)]
        r2 = K2_MODEL_1A * c_mab1 * c_drug  # [mol/(L*s)]
        r3 = K3_MODEL_1A * c_drug  # [mol/(L*s)]
        return [
            -r1 - dilution_s * c_mab,
            r1 - r2 - dilution_s * c_mab1,
            r2 - dilution_s * c_mab2,
            -r1 - r2 - r3 + dilution_s * (feed_drug_conc - c_drug),
            r3 - dilution_s * c_depl,
            feed_flow_l_s,
        ]

    ref_sol = solve_ivp(
        _scipy_semibatch_model_1a_rhs,
        (0.0, feed_time_s),
        [C_MAB_STD, 0.0, 0.0, 0.0, 0.0, gst1_vol_l],
        t_eval=[feed_time_s],
        method="RK45",
        rtol=1e-10,
        atol=1e-13,
    )
    assert ref_sol.success
    ref_conc_final = ref_sol.y[:5, -1]  # [mol/L]
    np.testing.assert_allclose(
        sb_conc_final[:5], ref_conc_final, rtol=REACTOR_REF_RTOL, atol=1e-9
    )
    sb_dar_final = (sb_conc_final[1] + 2.0 * sb_conc_final[2]) / (
        sb_conc_final[0] + sb_conc_final[1] + sb_conc_final[2]
    )  # [-]
    ref_dar_final = (ref_conc_final[1] + 2.0 * ref_conc_final[2]) / (
        ref_conc_final[0] + ref_conc_final[1] + ref_conc_final[2]
    )  # [-]
    assert sb_dar_final == pytest.approx(ref_dar_final, rel=REACTOR_REF_RTOL)
