"""Verification contracts for the WP1.5 M6 bioconjugation case selection (#357).

Scope: exercises the representative cysteine-maleimide bioconjugation reaction
networks, unit conversions (both ``[L/(mmol*s)]`` and ``[L/(umol*s)]`` to
``[L/(mol*s)]``), stoichiometric mass conservation, macromixing and micromixing
Damkohler timescale scaling, dual-wavelength UV-Vis and multivariate spectral
PAT contracts, multi-experiment (batch and fed-batch) parameter estimation,
``CalibrationRecord`` provenance round-trip, WP4 four-step STN/RTN process
sequence balances (both the 4-task Tier 2 baseline and the 3-sub-task Tier 1
UF/DF-exchange + DHAA-reoxidation expansion), and native ``BatchReactor`` /
``SemibatchReactor`` execution plus blocker reproducers (#486, #531) documented
in ``doc/review_notes/bioconjugation_case_selection.md``.

Data provenance:
- ``origin='literature'``: kinetic rate constants, standard deviations, 95%
  confidence intervals, initial concentrations, vessel mixing times, buffer
  density and viscosity, and 248/280 nm molar extinction coefficients from
  peer-reviewed literature (Andris et al., 2019,
  doi:10.1016/j.jbiotec.2019.09.013; Weggen et al., 2023,
  doi:10.3389/fbioe.2023.1123842; Weggen et al., 2024,
  doi:10.3389/fbioe.2024.1403644; Weggen et al., 2025,
  doi:10.1002/bit.28899; Hamblett et al., 2004,
  doi:10.1158/1078-0432.CCR-04-0789; Schiemer et al., 2023,
  doi:10.1002/bit.28428). Note that the kinetic benchmark uses ``ADC1`` with
  ``NPM`` / ``Drug1`` (Weggen et al., 2023, 2024) whereas the dual-wavelength
  UV-Vis benchmark uses ``cAC10`` with ``vc-MMAE`` (Hamblett et al., 2004).
- ``origin='synthetic'``: forward-simulated concentration time series, the use
  of published ``RMSE_CV`` (``1.863e-6 mol/L``) as a synthetic homoscedastic
  measurement-weighting standard deviation, all 4-wavelength MCR/PCR
  verification absorptivities beyond the two literature ``cAC10`` / ``vc-MMAE``
  reference wavelengths (including the entire ``Drug_depl`` row and the
  310/342 nm columns), schema placeholder thermal properties (``cp_liq``,
  ``t_crit``), and WP4 scheduling task durations and hold-time design
  parameters. These verify the model, PAT, and scheduling interfaces and are
  not claimed as an empirical calibration dataset.

Core unit tests run in under three seconds without Assimulo; the final
integration test uses ``pytest.importorskip("assimulo")`` to exercise native
``BatchReactor`` and ``SemibatchReactor`` integration and reproduce the
demonstrated solver/sensitivity blockers (#486, #531).
"""

import hashlib
import inspect
import json
from pathlib import Path
import re

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
# mu = 1.06e-3 Pa*s, Weggen et al., 2023, Sec. 2.2.2) with surrogate maleimide
# payload N-(1-pyrenyl)maleimide (NPM, C20H11NO2, M = 297.31 g/mol, Weggen et
# al., 2023, Sec. 2.1.2). Molecular weights satisfy exact stoichiometric mass
# conservation (stoich_matrix @ mw == 0) across R1, R2, and R3.
# Note: cp_liq [J/(mol*K)] and t_crit [K] are synthetic schema-required
# placeholders (origin='synthetic') that are not used by isothermal liquid
# reactor integration (isothermal=True).
BUFFER_DENSITY = 1010.5  # [kg/m**3], Weggen et al. (2023), Sec. 2.2.2
DYNAMIC_VISC_BUFFER = 1.06e-3  # [Pa*s], 1.06 mPa*s, Weggen et al. (2023), Sec. 2.2.2
KINEMATIC_VISC_BUFFER = (
    DYNAMIC_VISC_BUFFER / BUFFER_DENSITY
)  # [m**2/s], 1.049e-6 m**2/s
DMSO_DENSITY = (
    1100.4  # [kg/m**3], neat DMSO liquid density at 293.15 K, CRC Handbook
)
REACTION_TEMP = 295.15  # [K], 22 degC isothermal laboratory condition
MAB_MW = 150000.0  # [g/mol], intact IgG1 molar mass, Weggen et al. (2023), Sec. 2.1.1
NPM_MW = 297.31  # [g/mol], N-(1-pyrenyl)maleimide (C20H11NO2) molar mass
BUFFER_MW = 19.67  # [g/mol], mole-fraction-weighted 90% v/v H2O + 10% v/v DMSO
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
        "mw": BUFFER_MW,  # [g/mol], 19.67 g/mol
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
# Sec. 2.2.2, Sec. 2.2.5, Sec. 2.2.8 & Sec. 3.2.1).
TAU_MIX_TUBE = 0.5  # [s], internally stirred 1.5 mL reaction tube
TAU_MIX_GST1 = 9.4  # [s], 300 mL lab glass stirred tank at 60 rpm
TAU_MIX_SUM = 17.6  # [s], 25 L single-use mixer at 400 rpm
TAU_MIX_GST2 = 32.2  # [s], 22 L glass stirred tank at 60 rpm
POWER_PER_VOL_GST1 = 3.81  # [W/m**3], Weggen et al. (2023), Table 1
POWER_PER_VOL_SUM = 23.71  # [W/m**3], Weggen et al. (2023), Table 1
POWER_PER_VOL_GST2 = 2.44  # [W/m**3], Weggen et al. (2023), Table 1
ENGULFMENT_COEFF = 17.24  # [-], Baldyga-Bourne / Vicum et al. (2004), Eq. (5)
MAB_MASS_CONC_STD = 5.0  # [g/L], Weggen et al. (2023), Table 2
C_MAB_STD = MAB_MASS_CONC_STD / MAB_MW  # [mol/L], 3.333e-5 mol/L
PAYLOAD_EXCESS_STD = 5.0  # [-], 5x molar payload excess, Table 2
C_DRUG_BULK_STD = PAYLOAD_EXCESS_STD * C_MAB_STD  # [mol/L], 1.667e-4 mol/L
C_DRUG_FEED_PLUME = 1.0e-2  # [mol/L], 10 mM stock feed plume, Sec. 2.2.5

# Dual-wavelength UV-Vis molar extinction coefficients at 248 nm and 280 nm
# for cAC10 IgG1 mAb and vc-MMAE (SGD-1006) drug-linker (origin='literature':
# Hamblett et al., 2004, Clin. Cancer Res. 10(20):7063-7070, p. 7064-7065;
# Cruz & Kayser, 2019, Cancers 11(6):870, doi:10.3390/cancers11060870, Sec. 4.4).
# SHORT_PATHLENGTH_CM = 0.10 cm represents a micro-volume / flow-cell optical
# pathlength (e.g., Unchained Labs Stunner, Weggen et al., 2024, Sec. 2.1.4)
# that keeps undiluted 5 g/L mAb absorbance inside the linear detector regime
# (A_280 <= 1.5 AU).
PATHLENGTH_CM = 1.0  # [cm], standard 1 cm reference cuvette pathlength
SHORT_PATHLENGTH_CM = 0.10  # [cm], 1 mm micro-volume / inline UV flow cell
MAX_LINEAR_ABSORBANCE_AU = 1.5  # [-], upper bound of linear photodiode regime
EPS_MAB_248 = 9.413e4  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_MAB_280 = 2.3415e5  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_DRUG_248 = 1.59e4  # [L/(mol*cm)], Hamblett et al. (2004); Cruz & Kayser (2019)
EPS_DRUG_280 = 1.50e3  # [L/(mol*cm)], Hamblett et al. (2004); Cruz & Kayser (2019)

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
    """Integrate the mono-thiol sub-case of the Weggen et al. (2024) Simple 1k model.

    Solves the isothermal batch or fed-batch material balance for the
    homogeneous mono-thiol heavy-chain limiting sub-case
    (``c_H0_0 = c_H0_2 = 0``) of the Simple ``1k + k_drug`` model (Weggen et
    al., 2024, Eqs. 1-10) with mono-thiol heavy chain ``H0_1``, conjugated
    heavy chain ``H1_1``, free payload ``Drug``, and inactivated payload
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
    switch_time_s = float(feed_duration_s)  # [s]

    if feed_rate_l_s > 0.0 and 0.0 < switch_time_s < t_end_s:
        times_piece1 = times[times <= switch_time_s]  # [s]
        has_exact_switch = (
            times_piece1.size > 0 and float(times_piece1[-1]) == switch_time_s
        )
        eval_piece1 = (
            times_piece1
            if has_exact_switch
            else np.append(times_piece1, switch_time_s)
        )  # [s]
        sol1 = solve_ivp(
            _make_rhs(float(feed_rate_l_s)),
            (0.0, switch_time_s),
            initial_state,
            t_eval=eval_piece1,
            method="RK45",
            rtol=1e-10,
            atol=1e-13,
        )
        if not sol1.success:
            raise RuntimeError(f"Piece 1 ODE integration failed: {sol1.message}")
        state_at_switch = sol1.y[:, -1].tolist()  # [mol/L x 4, L]
        times_piece2 = times[times > switch_time_s]  # [s]
        eval_piece2 = np.insert(times_piece2, 0, switch_time_s)  # [s]
        sol2 = solve_ivp(
            _make_rhs(0.0),
            (switch_time_s, t_end_s),
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
        if (feed_rate_l_s > 0.0 and switch_time_s >= t_end_s)
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
    (2004, Clin. Cancer Res. 10(20):7063-7070, p. 7064-7065).

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
    # Verify both [L/(mmol*s)] and [L/(umol*s)] -> [L/(mol*s)] conversions.
    assert K1_MODEL_1A == pytest.approx(797.0, rel=CLOSED_FORM_RTOL)
    assert K2_MODEL_1A == pytest.approx(1476.0, rel=CLOSED_FORM_RTOL)
    assert K1_RED_TIER2 == pytest.approx(698.0, rel=CLOSED_FORM_RTOL)
    assert K3_RED_TIER2 == pytest.approx(68.1, rel=CLOSED_FORM_RTOL)
    assert K_RED_SEED_TIER2 == pytest.approx(100.0, rel=CLOSED_FORM_RTOL)

    # Verify Weggen et al. (2024, Table 2) parameter SD vs 95% CI distinction.
    ci_low, ci_high = K1_DAR2_DRUG1_CI95_BOUNDS_MMOL  # [L/(mmol*s)]
    assert 0.5 * (ci_high - ci_low) == pytest.approx(
        K1_DAR2_DRUG1_CI95_HALF_MMOL, rel=CLOSED_FORM_RTOL
    )
    assert K1_DAR2_DRUG1_CI95_HALF_MMOL > K1_DAR2_DRUG1_SD_MMOL

    # Verify the two distinct reduced thiol heterogeneity distributions in
    # Section 3.2:
    # 1. Weggen et al. (2024, Sec. 3.3.1) heavy-chain distribution (0.11 H0_0,
    #    0.84 H0_1, 0.05 H0_2) -> intact-mAb DAR_max = 2 * (0.84 + 2*0.05) = 1.88.
    # 2. Weggen et al. (2023, Sec. 2.2.5) intact-mAb distribution (88.59% 2-site,
    #    8.60% 1-site, 2.81% 0-site) -> intact-mAb DAR_max = 1.8578.
    hc_frac_2024 = np.array([0.11, 0.84, 0.05])  # [-], (f_H0_0, f_H0_1, f_H0_2)
    mab_frac_2023 = np.array([0.8859, 0.0860, 0.0281])  # [-], (2-site, 1-site, 0-site)
    assert float(np.sum(hc_frac_2024)) == pytest.approx(1.0, rel=CLOSED_FORM_RTOL)
    assert float(np.sum(mab_frac_2023)) == pytest.approx(1.0, rel=CLOSED_FORM_RTOL)
    dar_max_2024 = 2.0 * (hc_frac_2024[1] + 2.0 * hc_frac_2024[2])  # [-]
    dar_max_2023 = 2.0 * mab_frac_2023[0] + 1.0 * mab_frac_2023[1]  # [-]
    assert dar_max_2024 == pytest.approx(1.88, rel=CLOSED_FORM_RTOL)
    assert dar_max_2023 == pytest.approx(1.8578, rel=CLOSED_FORM_RTOL)

    # Verify the 6-species heavy-chain population conservation and intact-mAb
    # DAR projection formula (Section 6.1, Weggen et al., 2024, Eqs. 1-11) on an
    # asymmetric intermediate heavy-chain state (H0_0, H0_1, H0_2, H1_1, H1_2, H2_2).
    c_hc_total = 6.0e-5  # [mol/L], total heavy-chain molar concentration (2 * c_mAb)
    c_hc_state = c_hc_total * np.array(
        [0.11, 0.24, 0.01, 0.60, 0.015, 0.025]
    )  # [mol/L], (H0_0, H0_1, H0_2, H1_1, H1_2, H2_2)
    assert c_hc_state[0] == pytest.approx(
        hc_frac_2024[0] * c_hc_total, rel=CLOSED_FORM_RTOL
    )
    assert (c_hc_state[1] + c_hc_state[3]) == pytest.approx(
        hc_frac_2024[1] * c_hc_total, rel=CLOSED_FORM_RTOL
    )
    assert (c_hc_state[2] + c_hc_state[4] + c_hc_state[5]) == pytest.approx(
        hc_frac_2024[2] * c_hc_total, rel=CLOSED_FORM_RTOL
    )
    dar_hc_proj = (
        2.0 * (c_hc_state[3] + c_hc_state[4] + 2.0 * c_hc_state[5]) / c_hc_total
    )  # [-]
    assert dar_hc_proj == pytest.approx(
        2.0 * (0.60 + 0.015 + 2.0 * 0.025), rel=CLOSED_FORM_RTOL
    )
    assert 0.0 < dar_hc_proj < dar_max_2024

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

    # Exact stoichiometric mass conservation across R1, R2, and R3:
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
    # number Da_E = tau_E / tau_R_plume [-] using the conjugation buffer
    # density (1010.5 kg/m**3) and viscosity (1.06e-3 Pa*s -> nu = 1.049e-6
    # m**2/s, Weggen et al., 2023, Sec. 2.2.2 & Eqs. 9-10).
    vessel_power_per_vol = {
        "GST_1_300mL": POWER_PER_VOL_GST1,  # [W/m**3]
        "SUM_25L": POWER_PER_VOL_SUM,  # [W/m**3]
        "GST_2_22L": POWER_PER_VOL_GST2,  # [W/m**3]
    }
    tau_engulfment = {
        name: ENGULFMENT_COEFF
        * np.sqrt(KINEMATIC_VISC_BUFFER / (p_per_v / BUFFER_DENSITY))
        for name, p_per_v in vessel_power_per_vol.items()
    }  # [s]
    da_engulfment_plume = {
        name: tau_e / tau_rxn_plume for name, tau_e in tau_engulfment.items()
    }  # [-]

    assert tau_engulfment["GST_1_300mL"] == pytest.approx(0.288, rel=1e-2)
    assert tau_engulfment["SUM_25L"] == pytest.approx(0.115, rel=1e-2)
    assert tau_engulfment["GST_2_22L"] == pytest.approx(0.359, rel=1e-2)

    assert da_engulfment_plume["GST_1_300mL"] == pytest.approx(2.30, rel=1e-2)
    assert da_engulfment_plume["SUM_25L"] == pytest.approx(0.922, rel=1e-2)
    assert da_engulfment_plume["GST_2_22L"] == pytest.approx(2.87, rel=1e-2)

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
    # 5.25 g/L mAb stays within linear photodiode bounds (A_280 <= 1.5 AU).
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

    # Integrate the 25-point closed-batch ODE trajectory (H0_1, H1_1, Drug,
    # Drug_depl) across t in [0, 900] s.
    times_s = np.linspace(0.0, 900.0, 25)  # [s]
    c_h0_init = 6.0e-5  # [mol/L]
    c_drug_init = 1.5e-4  # [mol/L]
    traj_batch = _solve_dar2_heavy_chain_experiment(
        (K1_DAR2_DRUG1, K_DRUG_DRUG1),
        times_s,
        c_h0_init_mol_l=c_h0_init,
        c_drug_init_mol_l=c_drug_init,
    )  # [mol/L], columns (H0_1, H1_1, Drug, Drug_depl)
    assert traj_batch[-1, 1] > 0.85 * c_h0_init
    assert traj_batch[-1, 3] > 0.35 * c_drug_init

    # 1. Closed-batch spectral aliasing on the full 4-species ODE trajectory
    # (Schiemer et al., 2023, doi:10.1002/bit.28428): when conjugated sites and
    # inactivated payload have the same 248/280 nm extinction as free payload
    # (eps_H1_1 = eps_H0_1 + eps_Drug and eps_Drug_depl = eps_Drug), the 25x2
    # in-situ absorbance trajectory [A_248(t), A_280(t)] is strictly invariant
    # across all 25 time points despite >85% heavy-chain conversion, and naive
    # two-wavelength inversion returns the constant initial payload-to-chain
    # charge ratio (2.50) instead of the true conjugated ratio H1_1(t)/c_h0_init.
    eps_aliased_4x2 = np.array(
        [
            [EPS_MAB_248, EPS_MAB_280],
            [EPS_MAB_248 + EPS_DRUG_248, EPS_MAB_280 + EPS_DRUG_280],
            [EPS_DRUG_248, EPS_DRUG_280],
            [EPS_DRUG_248, EPS_DRUG_280],
        ]
    )  # [L/(mol*cm)], shape (4 species, 2 wavelengths)
    spectra_aliased_2wl = SHORT_PATHLENGTH_CM * (
        traj_batch @ eps_aliased_4x2
    )  # [-], shape (25, 2)
    np.testing.assert_allclose(
        spectra_aliased_2wl,
        np.broadcast_to(spectra_aliased_2wl[0], spectra_aliased_2wl.shape),
        rtol=CLOSED_FORM_RTOL,
    )
    _c_h_aliased, _c_d_aliased, ratio_aliased = _dual_wavelength_uv_invert(
        spectra_aliased_2wl, pathlength_cm=SHORT_PATHLENGTH_CM
    )
    np.testing.assert_allclose(
        ratio_aliased,
        np.full(times_s.size, c_drug_init / c_h0_init),
        rtol=CLOSED_FORM_RTOL,
    )

    # Also verify closed-batch 2-wavelength (248, 280 nm) spectral aliasing on
    # the 5-species intact-antibody Model 1A trajectory (mAb, mAb_1Drug,
    # mAb_2Drug, Drug, Drug_depl) across t in [0, 300] s.
    def _model_1a_batch_rhs(_time_s, state):
        """Evaluate Model 1A closed-batch ODEs for 5-species spectral aliasing.

        Parameters
        ----------
        _time_s : float
            Current integration time [s].
        state : sequence of float
            ``(c_mab, c_mab1, c_mab2, c_drug, c_depl)`` [mol/L].

        Returns
        -------
        list of float
            Time derivatives ``[dc_mab, dc_mab1, dc_mab2, dc_drug, dc_depl]``
            [mol/(L*s)].
        """
        c_m0, c_m1, _c_m2, c_d, _c_dep = state  # [mol/L]
        r1 = K1_MODEL_1A * c_m0 * c_d  # [mol/(L*s)]
        r2 = K2_MODEL_1A * c_m1 * c_d  # [mol/(L*s)]
        r3 = K3_MODEL_1A * c_d  # [mol/(L*s)]
        return [-r1, r1 - r2, r2, -r1 - r2 - r3, r3]

    times_1a_s = np.linspace(0.0, 300.0, 25)  # [s]
    sol_1a = solve_ivp(
        _model_1a_batch_rhs,
        (0.0, 300.0),
        [C_MAB_STD, 0.0, 0.0, C_DRUG_BULK_STD, 0.0],
        t_eval=times_1a_s,
        method="RK45",
        rtol=1e-10,
        atol=1e-13,
    )
    assert sol_1a.success
    traj_1a = sol_1a.y.T  # [mol/L], shape (25, 5)
    assert traj_1a[-1, 2] > 0.98 * C_MAB_STD
    eps_aliased_5x2 = np.array(
        [
            [EPS_MAB_248, EPS_MAB_280],
            [EPS_MAB_248 + EPS_DRUG_248, EPS_MAB_280 + EPS_DRUG_280],
            [EPS_MAB_248 + 2.0 * EPS_DRUG_248, EPS_MAB_280 + 2.0 * EPS_DRUG_280],
            [EPS_DRUG_248, EPS_DRUG_280],
            [EPS_DRUG_248, EPS_DRUG_280],
        ]
    )  # [L/(mol*cm)], shape (5 species, 2 wavelengths)
    spectra_1a_aliased = SHORT_PATHLENGTH_CM * (
        traj_1a @ eps_aliased_5x2
    )  # [-], shape (25, 2)
    np.testing.assert_allclose(
        spectra_1a_aliased,
        np.broadcast_to(spectra_1a_aliased[0], spectra_1a_aliased.shape),
        rtol=CLOSED_FORM_RTOL,
    )
    _c_m1a_aliased, _c_d1a_aliased, dar_1a_aliased = _dual_wavelength_uv_invert(
        spectra_1a_aliased, pathlength_cm=SHORT_PATHLENGTH_CM
    )
    np.testing.assert_allclose(
        dar_1a_aliased,
        np.full(times_1a_s.size, PAYLOAD_EXCESS_STD),
        rtol=CLOSED_FORM_RTOL,
    )

    # 2. Optical rank theorem for intact species (mAb_0, mAb_1, mAb_2, Drug, Drug_depl):
    # Under additive bound-chromophore spectra (eps_mAb1 = eps_mAb + eps_conj_site,
    # eps_mAb2 = eps_mAb + 2 * eps_conj_site), 2 * eps_mAb1 == eps_mAb0 + eps_mAb2,
    # so the 5-species extinction matrix has column rank 4 (cannot separate mAb_1
    # from 0.5*(mAb_0 + mAb_2) purely optically without the kinetic ODE observer).
    # The 4-coordinate basis (mAb_total, conj_sites, Drug, Drug_depl) has full
    # rank 4 if and only if [eps_mAb, eps_Drug, Delta_eps_conj, Delta_eps_depl]
    # are linearly independent; if Delta_eps_depl is collinear with Delta_eps_conj,
    # the 4-chromophore matrix drops to rank 3 even when both difference spectra
    # are non-zero.
    # Provenance: only the 248/280 nm entries of eps_mab, eps_conj_site, and
    # eps_drug_free are origin='literature' (Hamblett et al., 2004, p. 7064);
    # the entire eps_drug_depl row and all 310/342 nm entries are
    # origin='synthetic' verification fixtures.
    eps_mab = np.array([EPS_MAB_248, EPS_MAB_280, 1.20e3, 0.15e3])  # [L/(mol*cm)]
    eps_conj_site = np.array(
        [EPS_DRUG_248, EPS_DRUG_280, 8.20e3, 4.10e3]
    )  # [L/(mol*cm)]
    eps_drug_free = np.array(
        [EPS_DRUG_248, EPS_DRUG_280, 1.45e4, 9.80e3]
    )  # [L/(mol*cm)]
    eps_drug_depl = np.array(
        [1.42e4, 1.20e3, 2.10e3, 0.60e3]
    )  # [L/(mol*cm)], synthetic

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
    eps_drug_depl_collinear = eps_drug_free + 0.5 * (
        eps_conj_site - eps_drug_free
    )  # [L/(mol*cm)], collinear non-zero difference spectrum
    assert np.linalg.norm(eps_drug_depl_collinear - eps_drug_free) > 0.0
    assert (
        np.linalg.matrix_rank(
            np.vstack(
                (eps_mab, eps_conj_site, eps_drug_free, eps_drug_depl_collinear)
            )
        )
        == 3
    )

    # 3. Multi-wavelength spectral resolution including Drug_depl across 4
    # diagnostic wavelengths (248, 280, 310, 342 nm; origin='synthetic' for
    # Drug_depl and 310/342 nm): exercise MultipleCurveResolution (S = C @ A)
    # on the batch trajectory and evaluate PCR_calibration out-of-sample on a
    # held-out 600 s fed-batch trajectory after subtracting the known heavy-chain
    # protein core background c_H_total(t) * eps_H0.
    conc_three_batch = traj_batch[
        :, [1, 2, 3]
    ]  # [mol/L], shape (25, 3): (H1_1, Drug, Drug_depl)
    pure_absorptivities = np.vstack(
        (
            SHORT_PATHLENGTH_CM * eps_conj_site,
            SHORT_PATHLENGTH_CM * eps_drug_free,
            SHORT_PATHLENGTH_CM * eps_drug_depl,
        )
    )  # [L/mol], effective pathlength-scaled absorptivities (l * eps), shape (3, 4)
    spectral_matrix_batch = (
        conc_three_batch @ pure_absorptivities
    )  # [-], shape (25, 4)

    mcr_true_params = np.array([K1_DAR2_DRUG1, K_DRUG_DRUG1])  # [L/(mol*s)], [1/s]
    mcr = MultipleCurveResolution(
        _solve_dar2_heavy_chain_experiment,
        param_seed=mcr_true_params,
        time_data=times_s,
        y_spectra=spectral_matrix_batch,
        measured_ind=[1, 2, 3],
        kwargs_fun={"c_h0_init_mol_l": c_h0_init, "c_drug_init_mol_l": c_drug_init},
    )
    mcr_obj = mcr.get_objective(mcr_true_params, out_array=False)
    assert mcr_obj == pytest.approx(0.0, abs=1e-20)
    np.testing.assert_allclose(mcr.epsilon_mcr, pure_absorptivities, rtol=1e-9)
    y_runs, _resid_runs, _weighted_resid, absorptivity_pure = mcr.get_global_analysis(
        mcr_true_params
    )
    np.testing.assert_allclose(absorptivity_pure, pure_absorptivities, rtol=1e-9)
    np.testing.assert_allclose(y_runs[0], spectral_matrix_batch, rtol=1e-9)

    # Train PCR_calibration on a two-level batch calibration set (c_drug_init =
    # 1.0e-4 and 2.0e-4 mol/L, spanning all 3 degrees of freedom of (H1_1, Drug,
    # Drug_depl) without the single-batch constant-sum constraint) and validate
    # out-of-sample prediction on a held-out 25-point 600 s fed-batch trajectory.
    traj_batch_low = _solve_dar2_heavy_chain_experiment(
        (K1_DAR2_DRUG1, K_DRUG_DRUG1),
        times_s,
        c_h0_init_mol_l=c_h0_init,
        c_drug_init_mol_l=1.0e-4,
    )  # [mol/L], shape (25, 4)
    traj_batch_high = _solve_dar2_heavy_chain_experiment(
        (K1_DAR2_DRUG1, K_DRUG_DRUG1),
        times_s,
        c_h0_init_mol_l=c_h0_init,
        c_drug_init_mol_l=2.0e-4,
    )  # [mol/L], shape (25, 4)
    conc_three_train = np.vstack(
        (traj_batch_low[:, [1, 2, 3]], traj_batch_high[:, [1, 2, 3]])
    )  # [mol/L], shape (50, 3)
    spectral_matrix_train = (
        conc_three_train @ pure_absorptivities
    )  # [-], shape (50, 4)

    traj_fedbatch = _solve_dar2_heavy_chain_experiment(
        (K1_DAR2_DRUG1, K_DRUG_DRUG1),
        times_s,
        c_h0_init_mol_l=c_h0_init,
        c_drug_init_mol_l=0.0,
        v0_l=4.0e-3,
        feed_rate_l_s=4.0e-4 / 600.0,
        feed_duration_s=600.0,
        c_drug_feed_mol_l=1.5e-3,
    )  # [mol/L], held-out fed-batch trajectory, shape (25, 4)
    conc_three_fedbatch = traj_fedbatch[:, [1, 2, 3]]  # [mol/L], shape (25, 3)
    spectral_matrix_fedbatch = (
        conc_three_fedbatch @ pure_absorptivities
    )  # [-], held-out spectra, shape (25, 4)

    pcr = PCR_calibration(
        spectral_matrix_train, num_comp=3, standardize=True, snv=False
    )
    pcr.get_regression(conc_three_train, num_comp=3)
    conc_pcr_train = pcr.predict(spectral_matrix_train)  # [mol/L]
    conc_pcr_test = pcr.predict(spectral_matrix_fedbatch)  # [mol/L]
    np.testing.assert_allclose(
        conc_pcr_train, conc_three_train, rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        conc_pcr_test, conc_three_fedbatch, rtol=1e-6, atol=1e-10
    )


@pytest.mark.unit
def test_weggen_2024_dar2_batch_and_fedbatch_estimation_and_calibration_record(
    tmp_path,
):
    """Fit the Weggen et al. (2024) Simple 1k + sink model and round-trip its record."""
    # Equal observation budget (N_t = 8 sampling times per run, 3 species = 24
    # observations per run, 48 total observations across two runs):
    # - Experiment 1: 1.0 mL batch run (c_H0_1,0 = 6.0e-5 mol/L, c_Drug,0 = 1.5e-4 mol/L).
    # - Experiment 2: 4.0 mL fed-batch run with 600 s syringe-pump dosing of 1.5 mM
    #   payload stock at 6.667e-7 L/s (total 0.4 mL added = 6.0e-7 mol payload).
    batch_times = np.array(
        [0.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 900.0]
    )  # [s]
    fedbatch_times = np.array(
        [0.0, 60.0, 180.0, 300.0, 450.0, 600.0, 750.0, 900.0]
    )  # [s]
    assert batch_times.size == fedbatch_times.size == 8
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
    # Synthetic homoscedastic measurement weighting assumption: set per-sample
    # standard deviation sigma to the published cross-validation RMSE_CV
    # (1.863e-6 mol/L, Weggen et al., 2024, Table 2).
    uncertainty_mol_l = 1.863e-6  # [mol/L]

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

    # 1. Propagated measurement-uncertainty covariance (include_mse=False):
    # evaluates (J^T W J)^(-1) directly from the declared measurement
    # standard deviation sigma = 1.863e-6 mol/L without multiplying by the
    # noiseless residual MSE (~0), yielding strictly positive physical
    # parameter standard deviations (~7.7 L/(mol*s) for k1, ~2.9e-5 1/s for k_drug).
    cov_propagated = estimator.get_covariance(
        include_mse=False
    )  # [[L**2/(mol**2*s**2), L/(mol*s**2)], [L/(mol*s**2), 1/s**2]]
    sd_propagated = np.sqrt(np.diag(cov_propagated))  # [L/(mol*s)], [1/s]
    assert sd_propagated[0] > 5.0
    assert sd_propagated[1] > 1.0e-5

    # 2. Deterministic noisy estimation check: with realistic homoscedastic
    # perturbations (0.15 * uncertainty_mol_l), residual MSE > 0 and
    # get_confidence_intervals() produces non-zero half-widths that strictly
    # bracket the true generating parameters (ci_low < true_param < ci_high).
    rng = np.random.default_rng(357)
    noise_scale = 0.15 * uncertainty_mol_l  # [mol/L]
    batch_noisy = batch_obs[:, :3] + rng.normal(
        0.0, noise_scale, size=batch_obs[:, :3].shape
    )  # [mol/L]
    fedbatch_noisy = fedbatch_obs[:, :3] + rng.normal(
        0.0, noise_scale, size=fedbatch_obs[:, :3].shape
    )  # [mol/L]
    noisy_experiments = {
        "batch_1mL": Experiment(
            {
                name: Measurement(
                    name,
                    batch_times,
                    batch_noisy[:, idx],
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
                    fedbatch_noisy[:, idx],
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
    noisy_estimator = ParameterEstimation(
        _solve_dar2_heavy_chain_experiment,
        param_seed=param_seed,
        x_data=noisy_experiments,
        name_params=["k1", "k_drug"],
        output_names=output_names,
    )
    noisy_fitted, noisy_cov, noisy_info = noisy_estimator.estimate_parameters(
        method="LM", verbose=False
    )
    noisy_intervals = noisy_estimator.get_confidence_intervals(verbose=False)
    n_obs_total = (batch_times.size + fedbatch_times.size) * len(measured_species)
    dof = n_obs_total - len(true_params)
    t_val = float(student_t.ppf(0.975, dof))  # [-]
    # Derive expected noisy covariance and confidence-interval half-widths
    # independently from the weighted residual vector noisy_info["fun"] and the
    # sensitivity Jacobian at noisy_fitted rather than reusing noisy_cov.
    jac_noisy_indep = noisy_estimator.get_gradient(
        noisy_fitted, out_array=True
    )  # [(mol*s)/L for row 0; s for row 1]
    mse_noisy_indep = float(np.sum(np.square(noisy_info["fun"])) / dof)  # [-]
    cov_noisy_indep = mse_noisy_indep * np.linalg.inv(
        jac_noisy_indep @ jac_noisy_indep.T
    )  # [[L**2/(mol**2*s**2), L/(mol*s**2)], [L/(mol*s**2), 1/s**2]]
    np.testing.assert_allclose(noisy_cov, cov_noisy_indep, rtol=1e-5)
    expected_noisy_half_widths = t_val * np.sqrt(
        np.diag(cov_noisy_indep)
    )  # [L/(mol*s)], [1/s]
    assert expected_noisy_half_widths[0] > 1.0
    assert expected_noisy_half_widths[1] > 1.0e-6
    for idx, pname in enumerate(("k1", "k_drug")):
        low, high = noisy_intervals[pname]
        assert low < float(true_params[idx]) < high
        assert 0.5 * (high + low) == pytest.approx(noisy_fitted[idx], rel=1e-10)
        assert 0.5 * (high - low) == pytest.approx(
            expected_noisy_half_widths[idx], rel=1e-5
        )

    # Compare Fisher Information Matrix F = J^T W J at equal sample budget
    # (N_t = 8 per run, 48 total observations in two-run designs) across:
    # (1) a single 8-point batch run (24 observations),
    # (2) a duplicated two-batch baseline (48 observations, which scales F by
    #     2x and det(F) by 2^2 = 4x while leaving normalized off-diagonal
    #     parameter collinearity |F_01| / sqrt(F_00 * F_11) unchanged), and
    # (3) the combined 8-point batch + 8-point 600 s fed-batch design (48
    #     observations): at equal sample budget, the duplicated two-batch design
    #     has the larger raw determinant (det(F_combined) / det(F_two_batch) =
    #     0.5136, det(F_combined) / det(F_batch) = 2.054), whereas the fed-batch
    #     run flips the sign of the cross-sensitivity between k1 and k_drug
    #     (F_01 > 0 in batch vs F_01 < 0 in fed-batch) so their combination
    #     cancels off-diagonal correlation and reduces normalized parameter
    #     collinearity by >30x compared to the equal-budget two-batch design
    #     (Weggen et al., 2024, Sec. 3.3.2).
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
    jac_batch = batch_only_est.get_gradient(
        true_params, out_array=True
    )  # [(mol*s)/L for row 0; s for row 1]
    jac_two_batch = two_batch_est.get_gradient(
        true_params, out_array=True
    )  # [(mol*s)/L for row 0; s for row 1]
    jac_combined = estimator.get_gradient(
        true_params, out_array=True
    )  # [(mol*s)/L for row 0; s for row 1]
    fim_batch = (
        jac_batch @ jac_batch.T
    )  # [[(mol*s/L)**2, mol*s**2/L], [mol*s**2/L, s**2]]
    fim_two_batch = (
        jac_two_batch @ jac_two_batch.T
    )  # [[(mol*s/L)**2, mol*s**2/L], [mol*s**2/L, s**2]]
    fim_combined = (
        jac_combined @ jac_combined.T
    )  # [[(mol*s/L)**2, mol*s**2/L], [mol*s**2/L, s**2]]
    np.testing.assert_allclose(
        cov_propagated, np.linalg.inv(fim_combined), rtol=1e-5
    )
    assert np.linalg.det(fim_two_batch) == pytest.approx(
        4.0 * np.linalg.det(fim_batch), rel=1e-10
    )
    assert np.linalg.det(fim_combined) / np.linalg.det(fim_batch) == pytest.approx(
        2.054, rel=1e-2
    )
    assert np.linalg.det(fim_combined) / np.linalg.det(
        fim_two_batch
    ) == pytest.approx(0.5136, rel=1e-2)
    fim_fedbatch = (
        fim_combined - fim_batch
    )  # [[(mol*s/L)**2, mol*s**2/L], [mol*s**2/L, s**2]]
    assert fim_batch[0, 1] > 0.0 and fim_fedbatch[0, 1] < 0.0
    collinearity_two_batch = abs(fim_two_batch[0, 1]) / np.sqrt(
        fim_two_batch[0, 0] * fim_two_batch[1, 1]
    )  # [-]
    collinearity_combined = abs(fim_combined[0, 1]) / np.sqrt(
        fim_combined[0, 0] * fim_combined[1, 1]
    )  # [-]
    assert collinearity_combined < 0.035 * collinearity_two_batch

    # Capture CalibrationRecord with explicit literature + synthetic provenance.
    # Pin the SHA-256 digest of the canonical JSON serialization of the full
    # synthetic generator specification (generator source SHA-256, path/base
    # revision, operating kwargs, solver options, measurement-weighting
    # assumption, sampling times, and observations).
    generator_rel_path = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    generator_source_sha256 = hashlib.sha256(
        inspect.getsource(_solve_dar2_heavy_chain_experiment).encode("utf-8")
    ).hexdigest()
    generator_revision = (
        f"{generator_rel_path}:_solve_dar2_heavy_chain_experiment"
        f"@sha256:{generator_source_sha256[:16]}"
    )
    synthetic_payload_bytes = json.dumps(
        {
            "generator": "_solve_dar2_heavy_chain_experiment",
            "generator_path": generator_rel_path,
            "generator_source_sha256": generator_source_sha256,
            "generator_revision": generator_revision,
            "model_subcase": (
                "mono-thiol limiting sub-case (c_H0_0 = c_H0_2 = 0) of Weggen "
                "et al. (2024) Simple 1k + k_drug model (Eqs. 1-10)"
            ),
            "solver_options": {
                "method": "RK45",
                "rtol": 1e-10,
                "atol": 1e-13,
                "piecewise_switch": "exact equality at feed_duration_s",
            },
            "true_params": {
                "k1_l_mol_s": float(true_params[0]),
                "k_drug_1_s": float(true_params[1]),
            },
            "uncertainty_mol_l": uncertainty_mol_l,
            "uncertainty_assumption": (
                "synthetic homoscedastic sigma set to published RMSE_CV "
                "(1.863e-6 mol/L, Weggen et al., 2024, Table 2)"
            ),
            "batch_1mL": {
                "kwargs": batch_kwargs,
                "times_s": batch_times.tolist(),
                "obs_mol_l": np.round(batch_obs, 15).tolist(),
            },
            "fedbatch_4mL_600s": {
                "kwargs": fedbatch_kwargs,
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
            revision="2024-v1-table2",
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
                "Two-piece RK45 forward simulation of the mono-thiol limiting "
                "sub-case (c_H0_0 = c_H0_2 = 0) of the Weggen et al. (2024) "
                "Simple 1k + k_drug model for 8-point 1 mL batch and 8-point "
                "4 mL 600 s syringe-pump fed-batch experiments"
            ),
            uri=REVIEW_NOTE.relative_to(REPO_ROOT).as_posix(),
            revision=generator_revision,
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
            "8-point batch and 8-point fed-batch sampling grids with synthetic "
            "homoscedastic per-sample uncertainty sigma set to the published "
            "cross-validation RMSE (1.863e-6 mol/L)",
            fields=("H0_1", "H1_1", "Drug"),
            source="wp15_m6_synthetic_forward_trajectories",
        ),
    ]
    model = ModelRevision(
        "_solve_dar2_heavy_chain_experiment",
        revision=generator_revision,
        description=(
            "Isothermal batch and fed-batch mono-thiol limiting sub-case "
            "(c_H0_0 = c_H0_2 = 0) of the heavy-chain DAR 2 Simple (1k + "
            "k_drug) bioconjugation ODE model of Weggen et al. (2024), Eqs. 1-10"
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
            "Simple (1k + k_drug) conjugation model (mono-thiol sub-case) "
            "across batch and fed-batch operating modes; synthetic forward "
            "trajectory from published parameters, not an empirical calibration"
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
    """Verify WP4 STN/RTN balances, Tier 1 UF/DF+DHAA expansion, and review-note sync."""
    # Densities [kg/L] for converting liquid volumes [L] to solution masses [kg]:
    # - DMSO-free dilute aqueous phosphate buffer / reagent streams: 1.000 kg/L
    # - Neat DMSO payload stock in T2: 1.1004 kg/L (DMSO_DENSITY)
    # Combining 20.00 L aqueous mAb + 2.25 L neat DMSO + 0.25 L aqueous NAC in
    # T2 yields 22.7259 kg in 22.50 L -> rho_mix = 1.01004 kg/L (~1010.5 kg/m**3).
    rho_aq_kg_l = 1.000  # [kg/L], DMSO-free dilute aqueous phosphate buffer
    rho_dmso_kg_l = DMSO_DENSITY * 1.0e-3  # [kg/L], 1.1004 kg/L

    t2_m_s1_kg = 20.00 * rho_aq_kg_l  # [kg], 20.0000 kg
    t2_m_dmso_kg = 2.25 * rho_dmso_kg_l  # [kg], 2.4759 kg (exact 10.00% v/v of 22.50 L)
    t2_m_nac_kg = 0.25 * rho_aq_kg_l  # [kg], 0.2500 kg
    t2_m_total_kg = t2_m_s1_kg + t2_m_dmso_kg + t2_m_nac_kg  # [kg], 22.7259 kg

    t3_m_buffer_kg = 140.00 * rho_aq_kg_l  # [kg], 140.0000 kg
    t3_m_total_kg = t2_m_total_kg + t3_m_buffer_kg  # [kg], 162.7259 kg
    t3_m_s3_kg = 20.00 * rho_aq_kg_l  # [kg], 20.0000 kg
    t3_m_waste_kg = t3_m_total_kg - t3_m_s3_kg  # [kg], 142.7259 kg

    # 1. Canonical 4-task WP4 sequence (Tier 2 partial-TCEP baseline) from
    # Section 8.2 of doc/review_notes/bioconjugation_case_selection.md.
    wp4_tasks = (
        {
            "task_id": "T1_reduction",
            "inputs_vol_frac": {
                "S0_native_mAb": 19.60 / 20.00,  # [L/L], 0.9800
                "S_TCEP_reductant": 0.40 / 20.00,  # [L/L], 0.0200
            },
            "outputs_vol_frac": {
                "S1_reduced_mAb": 1.0000,  # [L/L]
            },
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
            "mass_in_kg": 20.00 * rho_aq_kg_l,  # [kg], 20.000 kg
            "mass_out_main_kg": 20.00 * rho_aq_kg_l,  # [kg], 20.000 kg
            "mass_out_waste_kg": 0.00,  # [kg]
            "protein_in_kg": 0.200,  # [kg]
            "protein_out_kg": 0.200,  # [kg]
            "duration_s": 5400.0,  # [s]
            "duration_range_s": (3600.0, 7200.0),  # [s]
            "storage_policy": "FW",
            "t_hold_max_s": 1800.0,  # [s], synthetic scheduling assumption
        },
        {
            "task_id": "T2_conjugation",
            "inputs_vol_frac": {
                "S1_reduced_mAb": 20.00 / 22.50,  # [L/L], 0.8889
                "S_payload_DMSO": 2.25 / 22.50,  # [L/L], 0.1000 (exact 10.00% v/v DMSO)
                "S_NAC_quench": 0.25 / 22.50,  # [L/L], 0.0111
            },
            "outputs_vol_frac": {
                "S2_crude_ADC": 1.0000,  # [L/L]
            },
            "inputs_kg_kg": {
                "S1_reduced_mAb": t2_m_s1_kg / t2_m_total_kg,  # [kg/kg], 0.8801
                "S_payload_DMSO": t2_m_dmso_kg / t2_m_total_kg,  # [kg/kg], 0.1089
                "S_NAC_quench": t2_m_nac_kg / t2_m_total_kg,  # [kg/kg], 0.0110
            },
            "outputs_kg_kg": {
                "S2_crude_ADC": 1.0000,  # [kg/kg]
            },
            "vol_in_l": 22.50,  # [L]
            "vol_out_main_l": 22.50,  # [L]
            "vol_out_waste_l": 0.00,  # [L]
            "mass_in_kg": t2_m_total_kg,  # [kg], 22.7259 kg
            "mass_out_main_kg": t2_m_total_kg,  # [kg], 22.7259 kg
            "mass_out_waste_kg": 0.00,  # [kg]
            "protein_in_kg": 0.200,  # [kg]
            "protein_out_kg": 0.200,  # [kg]
            "duration_s": 2700.0,  # [s]
            "duration_range_s": (1800.0, 3600.0),  # [s]
            "storage_policy": "FW",
            "t_hold_max_s": 3600.0,  # [s]
        },
        {
            "task_id": "T3_desalting_ufdf",
            "inputs_vol_frac": {
                "S2_crude_ADC": 22.50 / 162.50,  # [L/L], 0.1385
                "S_diafiltration_buffer": 140.00 / 162.50,  # [L/L], 0.8615 (7 diavolumes)
            },
            "outputs_vol_frac": {
                "S3_desalted_ADC": 20.00 / 162.50,  # [L/L], 0.1231
                "S_ufdf_permeate_waste": 142.50 / 162.50,  # [L/L], 0.8769
            },
            "inputs_kg_kg": {
                "S2_crude_ADC": t2_m_total_kg / t3_m_total_kg,  # [kg/kg], 0.1397
                "S_diafiltration_buffer": t3_m_buffer_kg / t3_m_total_kg,  # [kg/kg], 0.8603
            },
            "outputs_kg_kg": {
                "S3_desalted_ADC": t3_m_s3_kg / t3_m_total_kg,  # [kg/kg], 0.1229
                "S_ufdf_permeate_waste": t3_m_waste_kg / t3_m_total_kg,  # [kg/kg], 0.8771
            },
            "vol_in_l": 162.50,  # [L]
            "vol_out_main_l": 20.00,  # [L]
            "vol_out_waste_l": 142.50,  # [L]
            "mass_in_kg": t3_m_total_kg,  # [kg], 162.7259 kg
            "mass_out_main_kg": t3_m_s3_kg,  # [kg], 20.0000 kg
            "mass_out_waste_kg": t3_m_waste_kg,  # [kg], 142.7259 kg
            "protein_in_kg": 0.200,  # [kg]
            "protein_out_kg": 0.196,  # [kg], 98.0% TFF retentate step recovery
            "duration_s": 4500.0,  # [s]
            "duration_range_s": (3600.0, 5400.0),  # [s]
            "storage_policy": "UIS",
            "t_hold_max_s": 86400.0,  # [s], configurable multi-day FW bound
        },
        {
            "task_id": "T4_purification",
            "inputs_vol_frac": {
                "S3_desalted_ADC": 20.00 / 80.00,  # [L/L], 0.2500
                "S_chrom_mobile_phase": 60.00 / 80.00,  # [L/L], 0.7500
            },
            "outputs_vol_frac": {
                "S4_purified_ADC_DS": 18.00 / 80.00,  # [L/L], 0.2250
                "S_chrom_strip_waste": 62.00 / 80.00,  # [L/L], 0.7750
            },
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
            "mass_in_kg": 80.00 * rho_aq_kg_l,  # [kg], 80.000 kg
            "mass_out_main_kg": 18.00 * rho_aq_kg_l,  # [kg], 18.000 kg
            "mass_out_waste_kg": 62.00 * rho_aq_kg_l,  # [kg], 62.000 kg
            "protein_in_kg": 0.196,  # [kg]
            "protein_out_kg": 0.180,  # [kg], 90.0% overall four-step protein yield
            "duration_s": 3600.0,  # [s]
            "duration_range_s": (2700.0, 5400.0),  # [s]
            "storage_policy": "UIS",
            "t_hold_max_s": np.inf,  # [s]
        },
    )

    for task in wp4_tasks:
        assert sum(task["inputs_vol_frac"].values()) == pytest.approx(
            1.0, abs=CLOSED_FORM_RTOL
        )
        assert sum(task["outputs_vol_frac"].values()) == pytest.approx(
            1.0, abs=CLOSED_FORM_RTOL
        )
        assert sum(task["inputs_kg_kg"].values()) == pytest.approx(
            1.0, abs=CLOSED_FORM_RTOL
        )
        assert sum(task["outputs_kg_kg"].values()) == pytest.approx(
            1.0, abs=CLOSED_FORM_RTOL
        )
        assert (
            task["vol_out_main_l"] + task["vol_out_waste_l"]
        ) == pytest.approx(task["vol_in_l"], rel=CLOSED_FORM_RTOL)
        assert (
            task["mass_out_main_kg"] + task["mass_out_waste_kg"]
        ) == pytest.approx(task["mass_in_kg"], rel=CLOSED_FORM_RTOL)
        low_s, high_s = task["duration_range_s"]
        assert low_s <= task["duration_s"] <= high_s
        assert task["storage_policy"] in {"FW", "UIS"}

    # Verify inter-task continuity of both liquid volumes [L] and solution
    # masses [kg] across T1 -> T2 -> T3 -> T4.
    for idx, state_key in enumerate(
        ("S1_reduced_mAb", "S2_crude_ADC", "S3_desalted_ADC")
    ):
        assert wp4_tasks[idx]["vol_out_main_l"] == pytest.approx(
            wp4_tasks[idx + 1]["vol_in_l"]
            * wp4_tasks[idx + 1]["inputs_vol_frac"][state_key],
            rel=CLOSED_FORM_RTOL,
        )
        assert wp4_tasks[idx]["mass_out_main_kg"] == pytest.approx(
            wp4_tasks[idx + 1]["mass_in_kg"]
            * wp4_tasks[idx + 1]["inputs_kg_kg"][state_key],
            rel=CLOSED_FORM_RTOL,
        )
        assert wp4_tasks[idx]["protein_out_kg"] == pytest.approx(
            wp4_tasks[idx + 1]["protein_in_kg"], rel=CLOSED_FORM_RTOL
        )
    overall_protein_yield = (
        wp4_tasks[-1]["protein_out_kg"] / wp4_tasks[0]["protein_in_kg"]
    )  # [-]
    assert overall_protein_yield == pytest.approx(0.90, rel=CLOSED_FORM_RTOL)

    # 2. Verify Tier 1 3-sub-task expansion (T1a_tcep_decap -> T1b_decap_ufdf
    # -> T1c_dhaa_reox) that explicitly tracks the 100.00 L intermediate UF/DF
    # exchange buffer input, 100.40 L TCEP-permeate waste output on U_ufdf_skid,
    # and 0.40 L DHAA reoxidation addition before T2_conjugation.
    tier1_t1_subtasks = (
        {
            "task_id": "T1a_tcep_decap",
            "vol_in_l": 20.00,  # [L], 19.60 L S0 + 0.40 L TCEP (15 eq)
            "vol_out_main_l": 20.00,  # [L], S1a_decapped_mAb
            "vol_out_waste_l": 0.00,  # [L]
            "duration_s": 3600.0,  # [s]
        },
        {
            "task_id": "T1b_decap_ufdf",
            "vol_in_l": 120.00,  # [L], 20.00 L S1a + 100.00 L exchange buffer (5 DV)
            "vol_out_main_l": 19.60,  # [L], S1b_exchanged_mAb
            "vol_out_waste_l": 100.40,  # [L], S_t1b_permeate_waste
            "duration_s": 3600.0,  # [s]
        },
        {
            "task_id": "T1c_dhaa_reox",
            "vol_in_l": 20.00,  # [L], 19.60 L S1b + 0.40 L DHAA (4.5 eq)
            "vol_out_main_l": 20.00,  # [L], S1_reduced_mAb delivered to T2
            "vol_out_waste_l": 0.00,  # [L]
            "duration_s": 3600.0,  # [s]
        },
    )
    for subtask in tier1_t1_subtasks:
        assert (
            subtask["vol_out_main_l"] + subtask["vol_out_waste_l"]
        ) == pytest.approx(subtask["vol_in_l"], rel=CLOSED_FORM_RTOL)
    assert tier1_t1_subtasks[-1]["vol_out_main_l"] == pytest.approx(
        wp4_tasks[0]["vol_out_main_l"], rel=CLOSED_FORM_RTOL
    )

    # 3. Verify reagent stock molar charges and constant-volume diafiltration
    # small-molecule clearance against the 1.333e-3 mol mAb basis
    # (0.200 kg / 150 kg/mol = 1.333333e-3 mol mAb).
    moles_mab_batch = 200.0 / MAB_MW  # [mol], 1.333333e-3 mol
    moles_tcep_tier2 = 0.40 * (2.20 * moles_mab_batch / 0.40)  # [mol], 2.933e-3 mol (2.20 eq)
    moles_tcep_tier1 = 0.40 * 50.0e-3  # [mol], 20.00e-3 mol (15.0 eq)
    moles_dhaa_tier1 = 0.40 * 15.0e-3  # [mol], 6.00e-3 mol (4.5 eq)
    moles_payload_t2 = 2.25 * (5.0 * moles_mab_batch / 2.25)  # [mol], 6.667e-3 mol (5.0 eq)
    moles_nac_t2 = 0.25 * 40.0e-3  # [mol], 10.00e-3 mol (1.5 eq per payload)
    assert moles_tcep_tier2 / moles_mab_batch == pytest.approx(2.20, rel=CLOSED_FORM_RTOL)
    assert moles_tcep_tier1 / moles_mab_batch == pytest.approx(15.0, rel=CLOSED_FORM_RTOL)
    assert moles_dhaa_tier1 / moles_mab_batch == pytest.approx(4.5, rel=CLOSED_FORM_RTOL)
    assert moles_payload_t2 / moles_mab_batch == pytest.approx(5.0, rel=CLOSED_FORM_RTOL)
    assert moles_nac_t2 / moles_payload_t2 == pytest.approx(1.5, rel=CLOSED_FORM_RTOL)
    diavolumes_t1b = 100.00 / 20.00  # [-], 5.0 diavolumes in T1b
    diavolumes_t3 = 140.00 / 20.00  # [-], 7.0 diavolumes in T3
    assert diavolumes_t1b == pytest.approx(5.0, rel=CLOSED_FORM_RTOL)
    assert diavolumes_t3 == pytest.approx(7.0, rel=CLOSED_FORM_RTOL)
    assert float(np.exp(-diavolumes_t1b)) < 0.01  # <1% residual TCEP after 5 DV
    assert float(np.exp(-diavolumes_t3)) < 0.001  # <0.1% residual small molecules after 7 DV

    # 4. Direct synchronization check against doc/review_notes/bioconjugation_case_selection.md:
    # verify that all WP4 task IDs, key Section 8.2 proportions, and all Section 7.1
    # test file paths referenced in the review note match the repository.
    note_text = REVIEW_NOTE.read_text(encoding="utf-8")
    assert round(wp4_tasks[1]["inputs_kg_kg"]["S_payload_DMSO"], 4) == 0.1089
    for required_token in (
        "T1_reduction",
        "T2_conjugation",
        "T3_desalting_ufdf",
        "T4_purification",
        "T1a_tcep_decap",
        "T1b_decap_ufdf",
        "T1c_dhaa_reox",
        "22.50",
        "22.7259",
        "162.50",
        "162.726",
        "100.40",
        "0.8801",
        "0.1089",
        "0.0110",
    ):
        assert required_token in note_text, (
            f"Expected WP4 Section 8.2 token {required_token!r} in {REVIEW_NOTE}"
        )

    referenced_test_paths = sorted(
        set(re.findall(r"tests/[A-Za-z0-9_/]+\.py", note_text))
    )
    assert len(referenced_test_paths) >= 14
    for rel_test_path in referenced_test_paths:
        assert (REPO_ROOT / rel_test_path).is_file(), (
            f"Non-existent test path {rel_test_path!r} cited in {REVIEW_NOTE}"
        )


@pytest.mark.assimulo
@pytest.mark.integration
def test_native_batch_and_semibatch_reactors_execute_dar2_conjugation(tmp_path):
    """Run native BatchReactor and SemibatchReactor solves and reproduce Section 7.2 blockers (#486, #531)."""
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

    # 1a. Reproduce Section 7.2 blocker (tracked in #531): default CVode
    # tolerance (atol = 1e-6 mol/L) on micromolar antibody states (~3.33e-5
    # mol/L) allows -1.25e-5 mol/L (38% of c_mAb) negative-concentration
    # undershoot and unphysical DAR(300 s) = 2.57 > 2.0 overshoot. Provisional
    # assertion while #531 is open: once #531 scales default tolerances to the
    # initial state magnitude or enforces non-negativity, update this block to
    # assert prof_default >= 0 and dar_default_final <= 2.0 without custom
    # sundials_opts.
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
    assert np.min(prof_default[:, :3]) == pytest.approx(-1.25e-5, rel=0.05)
    assert dar_default_final == pytest.approx(2.57, rel=0.05)

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
    micromolar_sundials_opts = {
        "rtol": 1e-8,  # [-], relative CVode integration tolerance
        "atol": 1e-11,  # [mol/L], absolute CVode state tolerance for micromolar mAb
    }
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
    # and non-zero with respect to k1, k2, k3.
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

    # 2a. Reproduce Section 7.2 blocker (tracked in #486): SemibatchReactor.solve_unit
    # raises NotImplementedError when eval_sens=True is requested. Provisional
    # assertion while #486 is open: once #486 implements fed-batch forward
    # sensitivities, replace this raises check with a finite sensitivity assertion.
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
    with pytest.raises(NotImplementedError, match="sensitivity.*not supported"):
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
