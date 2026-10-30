"""Verification contracts for the WP1.5 M6 bioconjugation case selection (#357).

Scope: exercises the representative cysteine-maleimide bioconjugation reaction
networks, unit conversions, mixing Damkohler timescale scaling, dual-wavelength
UV-Vis and multivariate spectral PAT contracts, multi-experiment (batch and
fed-batch) parameter estimation, and ``CalibrationRecord`` provenance round-trip
documented in ``doc/review_notes/bioconjugation_case_selection.md``.

Data provenance: kinetic rate constants, initial concentrations, vessel mixing
times, and molar extinction coefficients are taken from peer-reviewed literature
(Andris et al., 2019, doi:10.1016/j.jbiotec.2019.09.013; Weggen et al., 2023,
doi:10.3389/fbioe.2023.1123842; Weggen et al., 2024,
doi:10.3389/fbioe.2024.1403644; Hamblett et al., 2004,
doi:10.1158/1078-0432.CCR-04-0789; Schiemer et al., 2023,
doi:10.1002/bit.28428) and tagged with ``origin='literature'``. Time-series
observations generated from those published parameters are tagged with
``origin='synthetic'``; they verify the model and PAT contracts and are not
claimed as an empirical calibration dataset.

Core tests run in milliseconds without Assimulo; the final integration test
exercises native ``BatchReactor`` and ``SemibatchReactor`` integration when
Assimulo is installed.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

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


HAS_ASSIMULO = importlib.util.find_spec("assimulo") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_NOTE = REPO_ROOT / "doc" / "review_notes" / "bioconjugation_case_selection.md"

# Exact unit conversion from literature bimolecular rate constants
# [L/(mmol*s)] to PharmaPy's [L/(mol*s)] basis (1 mol = 1000 mmol).
MMOL_PER_MOL = 1000.0  # [mmol/mol], exact metric prefix

# Model 1A rate constants: intact DAR 2 conjugation of ADC1 with surrogate
# maleimide payload NPM in 20 mM NaH2PO4, 1 mM EDTA, pH 7.0, 10% (v/v) DMSO
# at 295.15 K (Andris et al., 2019; Weggen et al., 2023, Eqs. 3-5).
K1_LIT_MMOL = 0.797  # [L/(mmol*s)], Weggen et al. (2023), Eq. (3)
K2_LIT_MMOL = 1.476  # [L/(mmol*s)], Weggen et al. (2023), Eq. (4)
K1_MODEL_1A = K1_LIT_MMOL * MMOL_PER_MOL  # [L/(mol*s)]
K2_MODEL_1A = K2_LIT_MMOL * MMOL_PER_MOL  # [L/(mol*s)]
K3_MODEL_1A = 1.55e-3  # [1/s], payload sink, Weggen et al. (2023), Eq. (5)

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
# Weggen et al., 2023, Sec. 2.2.2). Units: mw [g/mol], rho_liq [kg/m**3],
# cp_liq [J/(mol*K)], t_crit [K].
BUFFER_DENSITY = 1010.5  # [kg/m**3], Weggen et al. (2023), Sec. 2.2.2
REACTION_TEMP = 295.15  # [K], 22 degC isothermal laboratory condition
BIOCONJUGATION_DATABASE = {
    "mAb": {
        "mw": 150000.0,
        "rho_liq": BUFFER_DENSITY,
        "cp_liq": [210000.0],
        "t_crit": 800.0,
    },
    "mAb_1Drug": {
        "mw": 150700.0,
        "rho_liq": BUFFER_DENSITY,
        "cp_liq": [211000.0],
        "t_crit": 800.0,
    },
    "mAb_2Drug": {
        "mw": 151400.0,
        "rho_liq": BUFFER_DENSITY,
        "cp_liq": [212000.0],
        "t_crit": 800.0,
    },
    "Drug": {
        "mw": 700.0,
        "rho_liq": BUFFER_DENSITY,
        "cp_liq": [950.0],
        "t_crit": 750.0,
    },
    "Drug_depl": {
        "mw": 718.0,
        "rho_liq": BUFFER_DENSITY,
        "cp_liq": [975.0],
        "t_crit": 750.0,
    },
    "buffer": {
        "mw": 20.5,
        "rho_liq": BUFFER_DENSITY,
        "cp_liq": [78.0],
        "t_crit": 650.0,
    },
}

# Model 1B rate constants: reduced heavy-chain DAR 2 Simple (1k + sink) model
# for ADC1 + Drug1 across batch (1 mL) and fed-batch (4 mL) runs (Weggen et
# al., 2024, Front. Bioeng. Biotechnol. 12:1403644, Table 2 & Sec. 3.1).
K1_DAR2_DRUG1_MMOL = 0.251  # [L/(mmol*s)], Weggen et al. (2024), Table 2
K1_DAR2_DRUG1_CI_HALF_MMOL = 0.023  # [L/(mmol*s)], 95% CI half-width
K1_DAR2_DRUG1 = K1_DAR2_DRUG1_MMOL * MMOL_PER_MOL  # [L/(mol*s)]
K_DRUG_DRUG1 = 1.0e-3  # [1/s], Weggen et al. (2024), Sec. 3.1

# Mixing vessel 95% homogenization times and standard reactant concentrations
# from Weggen et al. (2023, Table 1, Table 2, Sec. 2.2.5 & Sec. 3.2.1).
TAU_MIX_TUBE = 0.5  # [s], internally stirred 1.5 mL reaction tube
TAU_MIX_GST1 = 9.4  # [s], 300 mL lab glass stirred tank at 60 rpm
TAU_MIX_SUM = 17.6  # [s], 25 L single-use mixer at 400 rpm
TAU_MIX_GST2 = 32.2  # [s], 22 L glass stirred tank at 60 rpm
MAB_MASS_CONC_STD = 5.0  # [g/L], Weggen et al. (2023), Table 2
MAB_MW = 150000.0  # [g/mol], intact IgG1 molar mass
C_MAB_STD = MAB_MASS_CONC_STD / MAB_MW  # [mol/L], 3.333e-5 mol/L
PAYLOAD_EXCESS_STD = 5.0  # [-], 5x molar payload excess, Table 2
C_DRUG_BULK_STD = PAYLOAD_EXCESS_STD * C_MAB_STD  # [mol/L], 1.667e-4 mol/L
C_DRUG_FEED_PLUME = 1.0e-2  # [mol/L], 10 mM stock feed plume, Sec. 2.2.5

# Dual-wavelength UV-Vis molar extinction coefficients at 248 nm and 280 nm
# for cAC10 IgG1 mAb and vc-MMAE (SGD-1006) drug-linker at 1 cm pathlength
# (Hamblett et al., 2004, Clin. Cancer Res. 10(20):7063-7070, p. 7064).
PATHLENGTH_CM = 1.0  # [cm], standard UV-Vis optical pathlength
EPS_MAB_248 = 9.413e4  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_MAB_280 = 2.3415e5  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_DRUG_248 = 1.59e4  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064
EPS_DRUG_280 = 1.50e3  # [L/(mol*cm)], Hamblett et al. (2004), p. 7064

# Numerical tolerances.
CLOSED_FORM_RTOL = 1e-12  # [-], IEEE floating-point evaluation agreement
ODE_FIT_RTOL = 1e-4  # [-], ODE-integrated LM parameter recovery tolerance
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

    def rhs(time_s, state):
        c_h0, c_h1, c_drug, c_depl, vol_l = state  # [mol/L x 4, L]
        flow_l_s = (
            feed_rate_l_s if (0.0 <= time_s <= feed_duration_s and feed_rate_l_s > 0.0) else 0.0
        )  # [L/s]
        dilution_s = flow_l_s / vol_l  # [1/s]
        r_conj = k1_val * c_h0 * c_drug  # [mol/(L*s)]
        r_depl = k_drug_val * c_drug  # [mol/(L*s)]
        dc_h0 = -r_conj - dilution_s * c_h0  # [mol/(L*s)]
        dc_h1 = r_conj - dilution_s * c_h1  # [mol/(L*s)]
        dc_drug = -r_conj - r_depl + dilution_s * (c_drug_feed_mol_l - c_drug)  # [mol/(L*s)]
        dc_depl = r_depl - dilution_s * c_depl  # [mol/(L*s)]
        return [dc_h0, dc_h1, dc_drug, dc_depl, flow_l_s]

    initial_state = [
        float(c_h0_init_mol_l),
        0.0,
        float(c_drug_init_mol_l),
        0.0,
        float(v0_l),
    ]  # [mol/L x 4, L]
    events = None
    if feed_duration_s > 0.0 and feed_rate_l_s > 0.0:

        def feed_off_event(time_s, _state):
            return time_s - feed_duration_s

        events = [feed_off_event]

    sol = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        initial_state,
        t_eval=times,
        method="RK45",
        rtol=1e-10,
        atol=1e-13,
        events=events,
         max_step=max(1.0, float(times[-1]) / 200.0),
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
    det_eps = EPS_MAB_248 * EPS_DRUG_280 - EPS_MAB_280 * EPS_DRUG_248  # [L**2/(mol**2*cm**2)]
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
    """Verify Model 1A stoichiometry, rates, Jacobians, and moiety conservation."""
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

    rates = kinetics.get_rxn_rates(conc, temp=REACTION_TEMP, overall_rates=True, jac=False)
    jac_states = kinetics.get_rxn_rates(conc, temp=REACTION_TEMP, jac=True)
    jac_params = kinetics.derivatives(conc, temp=REACTION_TEMP, dstates=False)

    np.testing.assert_allclose(rates, expected_species_rates, rtol=CLOSED_FORM_RTOL)

    # Exact linear invariants:
    # 1. Antibody core conservation: r_mAb + r_mAb_1Drug + r_mAb_2Drug == 0
    # 2. Drug moiety conservation: r_Drug + r_Drug_depl + r_mAb_1Drug + 2*r_mAb_2Drug == 0
    assert rates[0] + rates[1] + rates[2] == pytest.approx(0.0, abs=1e-20)
    assert (rates[3] + rates[4] + rates[1] + 2.0 * rates[2]) == pytest.approx(
        0.0, abs=1e-20
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
    """Verify Vicum et al. (2004) / Weggen et al. (2023) Eq. (9) timescale ratios."""
    # Characteristic bimolecular reaction time tau_R = 1 / (k1 * (c_mAb + c_Drug)) [s].
    tau_rxn_bulk = 1.0 / (K1_MODEL_1A * (C_MAB_STD + C_DRUG_BULK_STD))  # [s]
    tau_rxn_plume = 1.0 / (K1_MODEL_1A * (C_MAB_STD + C_DRUG_FEED_PLUME))  # [s]

    assert tau_rxn_bulk == pytest.approx(30000.0 / (797.0 * 6.0), rel=CLOSED_FORM_RTOL)
    assert tau_rxn_plume == pytest.approx(30000.0 / (797.0 * 301.0), rel=CLOSED_FORM_RTOL)

    vessel_mixing_times = {
        "tube_1_5mL": TAU_MIX_TUBE,
        "GST_1_300mL": TAU_MIX_GST1,
        "SUM_25L": TAU_MIX_SUM,
        "GST_2_22L": TAU_MIX_GST2,
    }  # [s]
    da_bulk = {
        name: tau_mix / tau_rxn_bulk for name, tau_mix in vessel_mixing_times.items()
    }  # [-]
    da_plume = {
        name: tau_mix / tau_rxn_plume for name, tau_mix in vessel_mixing_times.items()
    }  # [-]

    # Monotone ordering across vessels matches the CFD Delta-DAR ranking
    # reported in Weggen et al. (2023, Sec. 3.2.3: tube < GST-1 < SUM < GST-2).
    assert (
        da_bulk["tube_1_5mL"]
        < da_bulk["GST_1_300mL"]
        < da_bulk["SUM_25L"]
        < da_bulk["GST_2_22L"]
    )
    assert da_bulk["tube_1_5mL"] < 0.1
    assert da_plume["GST_2_22L"] > 250.0


@pytest.mark.unit
def test_dual_wavelength_uv_vis_contract_and_in_situ_mcr_deconvolution():
    """Verify Hamblett et al. (2004) UV-Vis DAR inversion and MCR/PLS PAT contracts."""
    c_mab_true = 3.5e-5  # [mol/L], 5.25 g/L intact antibody
    dar_targets = np.array([0.0, 1.0, 1.9, 2.0, 4.0, 8.0])  # [-]
    c_drug_conj_true = c_mab_true * dar_targets  # [mol/L]

    # Forward Beer-Lambert absorbance prediction at 248 nm and 280 nm [-].
    a_248 = PATHLENGTH_CM * (
        EPS_MAB_248 * c_mab_true + EPS_DRUG_248 * c_drug_conj_true
    )  # [-]
    a_280 = PATHLENGTH_CM * (
        EPS_MAB_280 * c_mab_true + EPS_DRUG_280 * c_drug_conj_true
    )  # [-]
    spectra_desalted = np.column_stack((a_248, a_280))  # [-]

    c_mab_rec, c_drug_rec, dar_rec = _dual_wavelength_uv_invert(spectra_desalted)
    np.testing.assert_allclose(
        c_mab_rec, np.full_like(dar_targets, c_mab_true), rtol=CLOSED_FORM_RTOL
    )
    np.testing.assert_allclose(c_drug_rec, c_drug_conj_true, rtol=CLOSED_FORM_RTOL, atol=1e-18)
    np.testing.assert_allclose(dar_rec, dar_targets, rtol=CLOSED_FORM_RTOL, atol=1e-14)

    # In-situ aliasing property (Schiemer et al., 2023, doi:10.1002/bit.28428):
    # if unconjugated and conjugated payload had identical extinction at 248 and
    # 280 nm, closed-batch drug-moiety conservation keeps (A_248, A_280) constant
    # across all reaction extents.
    c_drug_total = 4.0 * c_mab_true  # [mol/L], fixed 4x initial payload charge
    c_free_drug = c_drug_total - c_drug_conj_true[:4]  # [mol/L], for DAR 0..2
    a_248_aliased = PATHLENGTH_CM * (
        EPS_MAB_248 * c_mab_true + EPS_DRUG_248 * (c_drug_conj_true[:4] + c_free_drug)
    )  # [-]
    np.testing.assert_allclose(
        a_248_aliased, np.full(4, a_248_aliased[0]), rtol=CLOSED_FORM_RTOL
    )

    # Multi-wavelength spectral resolution when maleimide thioether addition
    # alters the 310 nm maleimide pi->pi* band: exercise MultipleCurveResolution
    # (S = C @ A) and PCR_calibration on a reaction time course.
    times_s = np.linspace(0.0, 900.0, 25)  # [s]
    traj = _solve_dar2_heavy_chain_experiment(
        (K1_DAR2_DRUG1, K_DRUG_DRUG1),
        times_s,
        c_h0_init_mol_l=6.0e-5,
        c_drug_init_mol_l=1.5e-4,
    )  # [mol/L], columns (H0_1, H1_1, Drug, Drug_depl)
    # 2-chromophore dynamic basis for MCR: [conjugated heavy chain H1_1,
    # free maleimide payload Drug] across 4 diagnostic wavelengths
    # (248, 280, 310, 342 nm) after subtracting constant protein/buffer background.
    conc_two = traj[:, [1, 2]]  # [mol/L], shape (25, 2)
    pure_absorptivities = np.array(
        [
            [1.59e4, 1.50e3, 0.80e3, 0.15e3],
            [1.42e4, 1.85e3, 3.60e3, 1.95e3],
        ]
    )  # [L/(mol*cm)], rows = (H1_1 conjugated site, free maleimide Drug)
    spectral_matrix = conc_two @ pure_absorptivities  # [-], shape (25, 4)

    mcr_true_params = np.array([K1_DAR2_DRUG1, K_DRUG_DRUG1])  # [L/(mol*s)], [1/s]
    mcr = MultipleCurveResolution(
        _solve_dar2_heavy_chain_experiment,
        param_seed=mcr_true_params,
        time_data=times_s,
        y_spectra=spectral_matrix,
        measured_ind=[1, 2],
        kwargs_fun={"c_h0_init_mol_l": 6.0e-5, "c_drug_init_mol_l": 1.5e-4},
    )
    mcr_obj = mcr.get_objective(mcr_true_params, out_array=False)
    assert mcr_obj == pytest.approx(0.0, abs=1e-20)
    np.testing.assert_allclose(mcr.epsilon_mcr, pure_absorptivities, rtol=1e-10)
    y_runs, _resid_runs, _weighted_resid, absorptivity_pure = mcr.get_global_analysis(
        mcr_true_params
    )
    np.testing.assert_allclose(absorptivity_pure, pure_absorptivities, rtol=1e-10)
    np.testing.assert_allclose(y_runs[0], spectral_matrix, rtol=1e-10)

    pcr = PCR_calibration(spectral_matrix, num_comp=2, standardize=True, snv=False)
    pcr.get_regression(conc_two, num_comp=2)
    conc_pcr_pred = pcr.predict(spectral_matrix)  # [mol/L]
    np.testing.assert_allclose(conc_pcr_pred, conc_two, rtol=1e-7, atol=1e-10)


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
        "c_h0_init_mol_l": 6.0e-5,
        "c_drug_init_mol_l": 1.5e-4,
        "v0_l": 1.0e-3,
        "feed_rate_l_s": 0.0,
        "feed_duration_s": 0.0,
        "c_drug_feed_mol_l": 0.0,
    }
    fedbatch_kwargs = {
        "c_h0_init_mol_l": 6.0e-5,
        "c_drug_init_mol_l": 0.0,
        "v0_l": 4.0e-3,
        "feed_rate_l_s": 4.0e-4 / 600.0,
        "feed_duration_s": 600.0,
        "c_drug_feed_mol_l": 1.5e-3,
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
    assert intervals["k1"][1] >= intervals["k1"][0]
    assert intervals["k_drug"][1] >= intervals["k_drug"][0]

    # Compare Fisher Information Matrix determinant det(J^T W J) of batch-only
    # vs combined batch + fed-batch to verify the MBDoE identifiability gain
    # from syringe-pump payload dosing (Weggen et al., 2024, Sec. 3.3.2).
    batch_only_est = ParameterEstimation(
        _solve_dar2_heavy_chain_experiment,
        param_seed=true_params,
        x_data={"batch_1mL": experiments["batch_1mL"]},
        name_params=["k1", "k_drug"],
        output_names=output_names,
    )
    jac_batch = batch_only_est.get_gradient(true_params, out_array=True)
    jac_combined = estimator.get_gradient(true_params, out_array=True)
    fim_batch = jac_batch @ jac_batch.T
    fim_combined = jac_combined @ jac_combined.T
    assert np.linalg.det(fim_combined) > 2.0 * np.linalg.det(fim_batch)

    # Capture CalibrationRecord with explicit literature + synthetic provenance.
    note_sha256 = hashlib.sha256(REVIEW_NOTE.read_bytes()).hexdigest()
    sources = [
        DataSource(
            "weggen_2024_table2",
            "literature",
            "Published Open Access under CC BY 4.0 in Frontiers in Bioengineering "
            "and Biotechnology 12:1403644 (PMC11274341)",
            description=(
                "Published rate constants and experimental conditions for the "
                "DAR 2 Simple (1k + k_drug) heavy-chain conjugation model of "
                "ADC1 + Drug1: k1 = 0.251 +/- 0.023 L/(mmol*s) (251.0 L/(mol*s)) "
                "and k_drug = 0.001 1/s (Table 2 and Section 3.1)"
            ),
            citation="Weggen et al. (2024), Front. Bioeng. Biotechnol. 12:1403644",
            uri="https://doi.org/10.3389/fbioe.2024.1403644",
            role="literature parameters",
        ),
        DataSource(
            "bioconjugation_case_selection_note",
            "synthetic",
            "Shipped in the PharmaPy repository (doc/review_notes/"
            "bioconjugation_case_selection.md); synthetic forward-simulated "
            "trajectories generated from the published Weggen et al. (2024) "
            "parameters for M6 contract verification, not an empirical dataset",
            description=(
                "Deterministic RK45 forward simulation of the Weggen et al. "
                "(2024) Simple 1k + k_drug model for 1 mL batch and 4 mL "
                "600 s syringe-pump fed-batch experiments"
            ),
            uri=REVIEW_NOTE.relative_to(REPO_ROOT).as_posix(),
            sha256=note_sha256,
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
            source="bioconjugation_case_selection_note",
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
    assert "not an empirical" in reloaded.description

    replayed = reloaded.rerun(
        _solve_dar2_heavy_chain_experiment, rtol=REPLAY_RTOL, atol=0.0
    )
    np.testing.assert_allclose(replayed.params_convg, fitted, rtol=REPLAY_RTOL, atol=0.0)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.skipif(
    not HAS_ASSIMULO,
    reason="assimulo is not installed; native reactor integration test skipped",
)
def test_native_batch_and_semibatch_reactors_execute_dar2_conjugation(tmp_path):
    """Run native BatchReactor and SemibatchReactor solves on Model 1A."""
    db_path = _write_bioconjugation_db(tmp_path)

    # 1. BatchReactor native solve at 5 g/L mAb (3.333e-5 mol/L) and 5x payload
    # in GST-1 (0.300 L = 3.0e-4 m**3).
    gst1_vol_m3 = 3.0e-4  # [m**3], 0.300 L lab stirred tank GST-1
    gst1_vol_l = 0.300  # [L]
    batch_kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )
    batch_conc_init = [
        C_MAB_STD,
        0.0,
        0.0,
        C_DRUG_BULK_STD,
        0.0,
        0.0,
    ]  # [mol/L], mAb, mAb_1Drug, mAb_2Drug, Drug, Drug_depl, buffer
    batch_reactor = BatchReactor(isothermal=True)
    batch_reactor.Kinetics = batch_kinetics
    batch_reactor.Phases = LiquidPhase(
        db_path,
        temp=REACTION_TEMP,
        vol=gst1_vol_m3,
        mole_conc=batch_conc_init,
        name_solv="buffer",
    )
    time_grid = np.linspace(0.0, 300.0, 31)  # [s]
    # Micromolar protein states (c_mAb ~ 3.33e-5 mol/L) require an absolute
    # tolerance well below CVode's 1e-6 mol/L default to prevent O(1e-6) mol/L
    # negative-concentration undershoot and DAR > 2 overshoot.
    micromolar_sundials_opts = {"rtol": 1e-8, "atol": 1e-11}
    batch_reactor.solve_unit(
        time_grid=time_grid,
        eval_sens=True,
        verbose=False,
        sundials_opts=micromolar_sundials_opts,
    )

    prof = batch_reactor.result.mole_conc  # [mol/L], shape (31, 6)
    mab_total = prof[:, 0] + prof[:, 1] + prof[:, 2]  # [mol/L]
    drug_moiety_total = prof[:, 3] + prof[:, 4] + prof[:, 1] + 2.0 * prof[:, 2]  # [mol/L]
    dar_final = (prof[-1, 1] + 2.0 * prof[-1, 2]) / mab_total[-1]  # [-]

    np.testing.assert_allclose(mab_total, C_MAB_STD, rtol=1e-6)
    np.testing.assert_allclose(drug_moiety_total, C_DRUG_BULK_STD, rtol=1e-6)
    assert np.all(prof[:, :5] >= -1e-11)
    assert 1.98 < dar_final <= 2.0 + 1e-6

    # 2. SemibatchReactor native solve with 60 s payload dosing into 0.300 L
    # (3.0e-4 m**3) inside a 0.500 L (5.0e-4 m**3) tank.
    semibatch_kinetics = RxnKinetics(
        db_path,
        rxn_list=list(MODEL_1A_REACTIONS),
        k_params=[K1_MODEL_1A, K2_MODEL_1A, K3_MODEL_1A],
        ea_params=[0.0, 0.0, 0.0],
    )
    tank_capacity_m3 = 5.0e-4  # [m**3], 0.500 L vessel capacity
    feed_time_s = 60.0  # [s], standard GST-1/GST-2 addition time, Table 2
    feed_vol_m3 = 5.0e-6  # [m**3], 5 mL (0.005 L) payload stock addition
    feed_vol_l = 0.005  # [L]
    feed_flow_m3_s = feed_vol_m3 / feed_time_s  # [m**3/s]
    total_drug_moles = gst1_vol_l * C_DRUG_BULK_STD  # [mol]
    feed_drug_conc = total_drug_moles / feed_vol_l  # [mol/L], 10 mM stock

    semibatch = SemibatchReactor(vol_tank=tank_capacity_m3, isothermal=True)
    semibatch.Kinetics = semibatch_kinetics
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
