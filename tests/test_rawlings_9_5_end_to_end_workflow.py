"""End-to-end integration tests for the Rawlings & Ekerdt 9.5 workflow (#337).

Scope: verifies the complete named-interface and parameter-estimation cohort
(#339, #338, #336, #341, #347, #534, #536) on the six-reaction, ten-species
catalytic batch-reactor network from Rawlings & Ekerdt (2002), *Chemical
Reactor Analysis and Design Fundamentals*, Chapter 9, Example 9.5
(``rawlings_9-5.ipynb``):
- ``RxnKinetics`` with reaction-ID-keyed ``rxns``, ``k_params``, and nested
  species reaction-order ``params_f`` mappings (including catalytic species
  appearing on both sides of a reaction arrow and omitted reactions defaulting
  to elementary stoichiometry).
- ``LiquidPhase`` initialized from sparse species-keyed ``mole_conc`` and a
  solvent ``mass_frac`` mapping, plus ``phase.retrieve_field('mole_conc')``.
- ``DynamicResult.retrieve_state('mole_conc')`` and bounded component-list
  display with ellipsis in ``DynamicResult.pprint()``.
- ``Measurement`` and ``Experiment`` objects (with ``name`` and ``author``
  metadata) fed to ``ParameterEstimation`` using ``'scipy'`` and a custom
  optimizer callable adapter, followed by ``CalibrationRecord.from_estimation``
  capture, JSON round-trip, and ``CalibrationRecord.rerun``.
- Full ``BatchReactor`` integration in the optional ``assimulo`` lane.
"""

import json
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision
from PharmaPy.Results import DynamicResult


SPECIES_ORDER = ('A', 'B', 'C', 'D', 'S', 'V', 'N', 'M', 'AA', 'solvent')
PARTICIPATING_SPECIES = ('A', 'B', 'C', 'D', 'S', 'V', 'N', 'M', 'AA')
MEASURED_SPECIES = ('A', 'B', 'C')

# Rawlings & Ekerdt (2002), Example 9.5 reaction network:
#   rxn_1: N + A --> B + A   (r_1 = k_1 * C_A, zero-order in excess N)
#   rxn_2: M + B --> A       (r_2 = k_2 * C_B, zero-order in excess M)
#   rxn_3: AA + A --> C + A  (r_3 = k_3 * C_A, zero-order in excess AA)
#   rxn_4: A --> D           (r_4 = k_4 * C_A)
#   rxn_5: C --> S           (r_5 = k_5 * C_C)
#   rxn_6: B + C --> V       (r_6 = k_6 * C_B * C_C)
RAWLINGS_9_5_RXNS = {
    'rxn_1': 'N + A --> B + A',
    'rxn_2': 'M + B --> A',
    'rxn_3': 'AA + A --> C + A',
    'rxn_4': 'A --> D',
    'rxn_5': 'C --> S',
    'rxn_6': 'B + C --> V',
}

SECONDS_PER_DAY = 86400.0  # [s/day], exact unit conversion

# Ground-truth rate constants from Rawlings & Ekerdt Example 9.5
# [1/day, 1/day, 1/day, 1/day, 1/day, L/(mol*day)]
RAWLINGS_9_5_TRUE_RATES_DAY = np.array(
    [2.0, 0.025, 1000.0, 0.25, 2.0, 7.5e-6], dtype=float
)
RAWLINGS_9_5_K_TRUE_SEC = {
    name: float(val / SECONDS_PER_DAY)
    for name, val in zip(RAWLINGS_9_5_RXNS.keys(), RAWLINGS_9_5_TRUE_RATES_DAY)
}  # [1/s] for rxn_1..rxn_5; [L/(mol*s)] for rxn_6

# Initial guess theta_0 = log10(k_0) from rawlings_9-5.ipynb Cell 4 [-]
RAWLINGS_9_5_LOG10_SEED = np.array(
    [0.8, -1.13, 3.15, -0.77, -0.16, -5.46], dtype=float
)  # [-]

RAWLINGS_9_5_PARAMS_F = {
    'rxn_1': {'N': 0.0, 'A': 1.0},  # [-], zero-order in N, first-order in A
    'rxn_2': {'M': 0.0, 'B': 1.0},  # [-], zero-order in M, first-order in B
    'rxn_3': {'AA': 0.0, 'A': 1.0},  # [-], zero-order in AA, first-order in A
}

# Initial liquid charge from rawlings_9-5.ipynb (#338 comment 6349032667)
INITIAL_MOLE_CONC = {
    'A': 1.0,   # [mol/L]
    'N': 50.0,  # [mol/L]
    'M': 50.0,  # [mol/L]
    'AA': 50.0,  # [mol/L]
}
INITIAL_SOLVENT_MASS_FRAC = {'solvent': 1.0}  # [-], pure solvent basis
INITIAL_VOLUME_M3 = 0.1  # [m**3]
ISOTHERMAL_TEMP_K = 298.15  # [K]

# 51-point time grid from Rawlings & Ekerdt Example 9.5 [days]
RAWLINGS_9_5_TIME_DAYS = np.linspace(0.0, 100.0, 51)  # [days]
RAWLINGS_9_5_TIME_SEC = RAWLINGS_9_5_TIME_DAYS * SECONDS_PER_DAY  # [s]

# Diagonal residual weights in rawlings_9-5.ipynb Cell 4 balancing the
# O(1) A, O(10) B, and O(500) C concentration scales [(L/mol)**2]
RAWLINGS_9_5_WEIGHT_DIAG = np.array([1.0, 1e2, 1e4], dtype=float)  # [(L/mol)**2]

ODE_RTOL = 1e-6  # [-], BDF integration relative tolerance
ODE_ATOL = 1e-8  # [mol/L], BDF integration absolute tolerance
IDENTIFIABLE_INDICES = [0, 1, 3, 5]  # [-], k_1, k_2, k_4, k_6
IDENTIFIABLE_RTOL = 0.05  # [-], recovery tolerance on noise-free trajectory
RATIO_LOG10_ATOL = 0.02  # [-], log10(k_3 / k_5) ratio tolerance
TRAJECTORY_RTOL = 1e-3  # [-], CVode vs SciPy trajectory agreement tolerance
TRAJECTORY_ATOL = 1e-4  # [mol/L], CVode vs SciPy trajectory absolute tolerance


@pytest.fixture()
def rawlings_purecomp_json(tmp_path) -> str:
    """Write the 10-species pure-component database for Rawlings 9.5.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    str
        Filesystem path to the generated JSON database.
    """
    species_template = {
        'mw': 1.0,  # [g/mol], textbook scaled solute molar mass
        't_crit': 500.0,  # [K]
        'p_crit': 40.0e5,  # [Pa]
        'rho_liq': 1000.0,  # [kg/m**3], constant liquid density
        'cp_liq': [100.0, 0.0, 0.0, 0.0, 0.0],  # [J/(mol*K)]
        'cp_vapor': [50.0, 0.0, 0.0, 0.0],  # [J/(mol*K)]
        'deltah_f': 0.0,  # [J/mol]
        'ant_coeff': [0.0, 0.0, 0.0],  # [-]
        'T_flash': 300.0,  # [K]
        'nu_groups': [1],  # [-]
        'id_groups': [1],  # [-]
        'mu_liq': [-10.0, 1000.0, 0.0, 0.0],  # [-]
        'sigma_st': [0.05, 0.0],  # [N/m]
    }
    db = {}
    for name in SPECIES_ORDER:
        entry = dict(species_template)
        if name == 'solvent':
            entry['mw'] = 18.015  # [g/mol], water-like solvent molar mass
            entry['rho_liq'] = 1000.0  # [kg/m**3]
        db[name] = entry

    json_path = tmp_path / 'rawlings_9_5_purecomp.json'
    json_path.write_text(json.dumps(db), encoding='utf-8')
    return str(json_path)


def _integrate_kinetics_batch(
    kinetics: RxnKinetics,
    initial_partic_conc: np.ndarray,
    time_sec: np.ndarray,
    temp_k: float = ISOTHERMAL_TEMP_K,
) -> np.ndarray:
    """Integrate isothermal batch kinetics using ``RxnKinetics.get_rxn_rates``.

    Parameters
    ----------
    kinetics : RxnKinetics
        Configured reaction kinetics instance.
    initial_partic_conc : np.ndarray
        Initial participating-species molar concentrations [mol/L], shape ``(9,)``.
    time_sec : np.ndarray
        Sample times [s], shape ``(n_time,)``.
    temp_k : float, optional
        Isothermal reactor temperature [K].

    Returns
    -------
    np.ndarray
        Participating-species concentration profiles [mol/L], shape
        ``(n_time, 9)``.
    """
    t_eval = np.asarray(time_sec, dtype=float)  # [s]

    def rhs(_t: float, conc: np.ndarray) -> np.ndarray:
        """Evaluate batch species production rates [mol/(L*s)]."""
        return kinetics.get_rxn_rates(np.maximum(conc, 0.0), temp_k)

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(initial_partic_conc, dtype=float),
        t_eval=t_eval,
        method='BDF',
        rtol=ODE_RTOL,
        atol=ODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"Batch ODE integration failed: {sol.message}")
    return sol.y.T  # [mol/L]


@pytest.mark.integration
def test_rawlings_9_5_named_interface_core_workflow(
    rawlings_purecomp_json: str,
) -> None:
    """Run the full Rawlings 9.5 named-interface workflow in the core lane."""
    # 1. RxnKinetics with dict rxns, k_params, and nested species order params_f (#339)
    kinetics_true = RxnKinetics(
        rawlings_purecomp_json,
        rxns=RAWLINGS_9_5_RXNS,
        k_params=RAWLINGS_9_5_K_TRUE_SEC,
        params_f=RAWLINGS_9_5_PARAMS_F,
    )
    assert kinetics_true.rxn_names == tuple(RAWLINGS_9_5_RXNS.keys())
    assert tuple(kinetics_true.partic_species) == PARTICIPATING_SPECIES

    # Catalytic species 'A' appears as both reactant and product in rxn_1 and rxn_3:
    # net stoichiometric coefficient is 0 while raw reactant/product entries are 1.
    idx_a = kinetics_true.partic_species.index('A')
    assert kinetics_true.stoich_matrix[0, idx_a] == pytest.approx(0.0)
    assert kinetics_true.stoich_react[0, idx_a] == pytest.approx(1.0)
    assert kinetics_true.stoich_prod[0, idx_a] == pytest.approx(1.0)
    assert kinetics_true.stoich_matrix[2, idx_a] == pytest.approx(0.0)
    assert kinetics_true.stoich_react[2, idx_a] == pytest.approx(1.0)
    assert kinetics_true.stoich_prod[2, idx_a] == pytest.approx(1.0)

    # Reordered dictionary keys align by reaction identifier, never insertion order
    reversed_k = dict(reversed(list(RAWLINGS_9_5_K_TRUE_SEC.items())))
    reversed_f = dict(reversed(list(RAWLINGS_9_5_PARAMS_F.items())))
    kinetics_reordered = RxnKinetics(
        rawlings_purecomp_json,
        rxns=RAWLINGS_9_5_RXNS,
        k_params=reversed_k,
        params_f=reversed_f,
    )
    np.testing.assert_allclose(kinetics_reordered.k_params, kinetics_true.k_params)
    np.testing.assert_allclose(kinetics_reordered.params_f, kinetics_true.params_f)

    # Also verify sequence rxns with 1-based 'rxn_i' keys in params_f (#339 comment 6349042099)
    kinetics_seq = RxnKinetics(
        rawlings_purecomp_json,
        rxns=list(RAWLINGS_9_5_RXNS.values()),
        k_params=list(RAWLINGS_9_5_K_TRUE_SEC.values()),
        params_f=RAWLINGS_9_5_PARAMS_F,
    )
    np.testing.assert_allclose(kinetics_seq.k_params, kinetics_true.k_params)
    np.testing.assert_allclose(kinetics_seq.params_f, kinetics_true.params_f)

    # Mismatched reaction identifier raises an actionable ValueError
    mismatched_k = dict(RAWLINGS_9_5_K_TRUE_SEC)
    mismatched_k['coupling'] = mismatched_k.pop('rxn_6')
    with pytest.raises(ValueError, match="k_params"):
        RxnKinetics(
            rawlings_purecomp_json,
            rxns=RAWLINGS_9_5_RXNS,
            k_params=mismatched_k,
            params_f=RAWLINGS_9_5_PARAMS_F,
        )

    # 2. LiquidPhase with sparse mole_conc dict + solvent mass_frac dict (#338, #336)
    phase = LiquidPhase(
        rawlings_purecomp_json,
        temp=ISOTHERMAL_TEMP_K,
        vol=INITIAL_VOLUME_M3,
        mole_conc=INITIAL_MOLE_CONC,
        mass_frac=INITIAL_SOLVENT_MASS_FRAC,
    )
    conc_arr = phase.retrieve_field('mole_conc')
    assert conc_arr.shape == (len(SPECIES_ORDER),)
    conc_map = phase.retrieve_field('mole_conc', as_dict=True)
    assert tuple(conc_map.keys()) == SPECIES_ORDER
    assert conc_map['A'] == pytest.approx(1.0)
    assert conc_map['N'] == pytest.approx(50.0)
    assert conc_map['M'] == pytest.approx(50.0)
    assert conc_map['AA'] == pytest.approx(50.0)
    for omitted in ('B', 'C', 'D', 'S', 'V'):
        assert conc_map[omitted] == pytest.approx(0.0)
    assert conc_map['solvent'] > 0.0

    measured_initial = phase.retrieve_field(
        'mole_conc', species=MEASURED_SPECIES, as_dict=True
    )
    assert tuple(measured_initial.keys()) == MEASURED_SPECIES
    assert measured_initial['A'] == pytest.approx(1.0)
    assert measured_initial['B'] == pytest.approx(0.0)
    assert measured_initial['C'] == pytest.approx(0.0)

    # 3. Integrate batch trajectory and verify DynamicResult (#341)
    initial_partic_conc = np.array(
        [conc_map[sp] for sp in PARTICIPATING_SPECIES], dtype=float
    )  # [mol/L]
    partic_prof = _integrate_kinetics_batch(
        kinetics_true, initial_partic_conc, RAWLINGS_9_5_TIME_SEC
    )  # [mol/L]
    solvent_prof = np.full(
        (RAWLINGS_9_5_TIME_SEC.size, 1), conc_map['solvent'], dtype=float
    )  # [mol/L]
    full_conc_prof = np.hstack([partic_prof, solvent_prof])  # [mol/L]

    di_states = {
        'mole_conc': {
            'dim': len(SPECIES_ORDER),
            'units': 'mol/L',
            'index': list(SPECIES_ORDER),
            'type': 'diff',
        },
        'temp': {
            'dim': 1,
            'units': 'K',
            'type': 'alg',
        },
    }
    states_data = {
        'mole_conc': full_conc_prof,
        'temp': np.full(RAWLINGS_9_5_TIME_SEC.size, ISOTHERMAL_TEMP_K),
    }
    result = DynamicResult(
        di_states, time=RAWLINGS_9_5_TIME_SEC, **states_data
    )

    retrieved_conc = result.retrieve_state('mole_conc')
    assert tuple(retrieved_conc.keys()) == SPECIES_ORDER
    np.testing.assert_allclose(retrieved_conc['A'], partic_prof[:, 0])
    np.testing.assert_allclose(retrieved_conc['B'], partic_prof[:, 1])
    np.testing.assert_allclose(retrieved_conc['C'], partic_prof[:, 2])

    # Bounded component-list display in pprint truncates 10 species with '...'
    printed_default = result.pprint()
    assert 'A, B, C, D, ..., solvent' in printed_default
    assert result.di_states['mole_conc']['index'] == list(SPECIES_ORDER)

    # 4. Experiment / Measurement + ParameterEstimation + CalibrationRecord (#347, #534)
    exp = Experiment(
        name='exp1',
        author='James Rawlings/John Ekerdt',
        measurements={
            'A': Measurement(
                x=RAWLINGS_9_5_TIME_DAYS,
                y=retrieved_conc['A'],
                unit_x='days',
                unit_y='mol/L',
                basis='molar concentration',
            ),
            'B': Measurement(
                x=RAWLINGS_9_5_TIME_DAYS,
                y=retrieved_conc['B'],
                unit_x='days',
                unit_y='mol/L',
                basis='molar concentration',
            ),
            'C': Measurement(
                x=RAWLINGS_9_5_TIME_DAYS,
                y=retrieved_conc['C'],
                unit_x='days',
                unit_y='mol/L',
                basis='molar concentration',
            ),
        },
    )
    assert exp.name == 'exp1'
    assert exp.author == 'James Rawlings/John Ekerdt'

    weights_df = pd.DataFrame(
        np.diag(RAWLINGS_9_5_WEIGHT_DIAG),
        index=list(MEASURED_SPECIES),
        columns=list(MEASURED_SPECIES),
    )  # [(L/mol)**2]

    true_log10_rates = np.log10(RAWLINGS_9_5_TRUE_RATES_DAY)  # [-]

    def model_wrapper(log10_params: np.ndarray, t_days: np.ndarray) -> np.ndarray:
        """Evaluate measured species [A, B, C] concentrations at ``t_days``.

        Parameters
        ----------
        log10_params : numpy.ndarray
            Base-10 logarithms of rate constants on a per-day basis [-], shape
            ``(6,)``, where reactions 1..5 are in ``1/day`` and reaction 6 is
            in ``L/(mol*day)``.
        t_days : numpy.ndarray
            Sample times [day], shape ``(num_times,)``.

        Returns
        -------
        numpy.ndarray
            Predicted molar concentrations of ``('A', 'B', 'C')`` [mol/L],
            shape ``(num_times, 3)``.
        """
        k_sec = (
            (10.0 ** np.asarray(log10_params, dtype=float)) / SECONDS_PER_DAY
        )  # [1/s] for rxn_1..rxn_5; [L/(mol*s)] for rxn_6
        kinetics_step = RxnKinetics(
            rawlings_purecomp_json,
            rxns=RAWLINGS_9_5_RXNS,
            k_params=k_sec,
            params_f=RAWLINGS_9_5_PARAMS_F,
        )
        prof = _integrate_kinetics_batch(
            kinetics_step,
            initial_partic_conc,
            np.asarray(t_days, dtype=float) * SECONDS_PER_DAY,
        )  # [mol/L]
        return prof[:, :3]

    def custom_lsq_optimizer(seed, objective_fun, gradient_fun, **options):
        """Wrap ``scipy.optimize.least_squares`` as a custom optimizer callable.

        Parameters
        ----------
        seed : numpy.ndarray
            Initial free parameter vector [-], shape ``(num_params,)``.
        objective_fun : callable
            Residual callback ``objective_fun(theta, out_array=True)`` [-].
        gradient_fun : callable
            Jacobian callback ``gradient_fun(theta, out_array=True)`` [-],
            shape ``(num_params, n_residuals)``.
        **options
            Forwarded solver options such as ``submethod`` and ``max_nfev``.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Converged SciPy least-squares result object.
        """
        return least_squares(
            lambda theta: objective_fun(theta, out_array=True),
            seed,
            jac=lambda theta: gradient_fun(theta, out_array=True).T,
            method=options.get('submethod', 'trf'),
            max_nfev=options.get('max_nfev', 80),
        )

    param_units_map = {name: '-' for name in RAWLINGS_9_5_RXNS.keys()}

    for method_arg, method_label, opt_kwargs in (
        ('scipy', 'scipy', {'max_nfev': 80}),
        (custom_lsq_optimizer, 'custom', {'submethod': 'trf', 'max_nfev': 80}),
    ):
        estimator = ParameterEstimation(
            model_wrapper,
            param_seed=RAWLINGS_9_5_LOG10_SEED.copy(),
            experiments=[exp],
            output_names=MEASURED_SPECIES,
            name_params=list(RAWLINGS_9_5_RXNS.keys()),
            weight_matrix=weights_df,
        )
        log_fit, cov_fit, info_fit = estimator.optimize_fn(
            method=method_arg,
            optim_options=opt_kwargs,
            verbose=False,
        )
        k_fit = 10.0 ** log_fit  # [1/day] for rxn_1..rxn_5; [L/(mol*day)] for rxn_6
        np.testing.assert_allclose(
            k_fit[IDENTIFIABLE_INDICES],
            RAWLINGS_9_5_TRUE_RATES_DAY[IDENTIFIABLE_INDICES],
            rtol=IDENTIFIABLE_RTOL,
        )
        assert (log_fit[2] - log_fit[4]) == pytest.approx(
            true_log10_rates[2] - true_log10_rates[4], abs=RATIO_LOG10_ATOL
        )
        assert cov_fit.shape == (len(RAWLINGS_9_5_RXNS), len(RAWLINGS_9_5_RXNS))
        np.testing.assert_allclose(info_fit['x'], log_fit)

        record = CalibrationRecord.from_estimation(
            estimator,
            {'exp1': exp},
            case=f'rawlings-9-5-{method_label}',
            parameter_units=param_units_map,
            model=ModelRevision('rawlings_9_5_batch', revision='v1'),
            sources=[DataSource('rawlings_9_5', 'synthetic', 'test suite')],
            weight_matrix=weights_df,
        )
        assert record.solver.method == method_label
        reloaded = CalibrationRecord.from_json(record.to_json())
        np.testing.assert_allclose(reloaded.fitted_values, log_fit)

        rerun_kwargs = (
            {'optimizer': custom_lsq_optimizer}
            if method_label == 'custom'
            else {}
        )
        replayed = reloaded.rerun(
            model_wrapper,
            rtol=1e-4,
            atol=1e-6,
            **rerun_kwargs,
        )
        np.testing.assert_allclose(
            replayed.params_convg, log_fit, rtol=1e-4, atol=1e-6
        )


@pytest.mark.assimulo
@pytest.mark.integration
def test_rawlings_9_5_batch_reactor_assimulo_workflow(
    rawlings_purecomp_json: str,
) -> None:
    """Run the Rawlings 9.5 workflow through BatchReactor in the Assimulo lane."""
    pytest.importorskip('assimulo')
    from PharmaPy.Reactors import BatchReactor

    kinetics = RxnKinetics(
        rawlings_purecomp_json,
        rxns=RAWLINGS_9_5_RXNS,
        k_params=RAWLINGS_9_5_K_TRUE_SEC,
        params_f=RAWLINGS_9_5_PARAMS_F,
    )
    phase = LiquidPhase(
        rawlings_purecomp_json,
        temp=ISOTHERMAL_TEMP_K,
        vol=INITIAL_VOLUME_M3,
        mole_conc=INITIAL_MOLE_CONC,
        mass_frac=INITIAL_SOLVENT_MASS_FRAC,
    )

    initial_partic = np.array(
        [
            phase.retrieve_field('mole_conc', as_dict=True)[sp]
            for sp in PARTICIPATING_SPECIES
        ],
        dtype=float,
    )  # [mol/L]

    reactor = BatchReactor(isothermal=True, return_sens=False)
    reactor.Phases = phase
    reactor.Kinetics = kinetics
    reactor.solve_unit(
        time_grid=RAWLINGS_9_5_TIME_SEC,
        sundials_opts={'maxh': SECONDS_PER_DAY},
        verbose=False,
    )

    conc = reactor.result.retrieve_state('mole_conc')
    assert tuple(conc.keys()) == PARTICIPATING_SPECIES

    # Compare BatchReactor trajectory against independent SciPy integration
    ref_prof = _integrate_kinetics_batch(
        kinetics, initial_partic, RAWLINGS_9_5_TIME_SEC
    )  # [mol/L]
    for idx, sp_name in enumerate(MEASURED_SPECIES):
        np.testing.assert_allclose(
            conc[sp_name],
            ref_prof[:, idx],
            rtol=TRAJECTORY_RTOL,
            atol=TRAJECTORY_ATOL,
        )

    # Verify bounded component display on reactor.result.pprint()
    printed_table = reactor.result.pprint()
    assert 'A, B, C, D, ..., AA' in printed_table

    # Fit parameters through the notebook's BatchReactor callback (#341 comment 6349046301)
    exp = Experiment(
        name='exp1',
        author='James Rawlings/John Ekerdt',
        measurements={
            'A': Measurement(
                x=RAWLINGS_9_5_TIME_DAYS,
                y=conc['A'],
                unit_x='days',
                unit_y='mol/L',
                basis='molar concentration',
            ),
            'B': Measurement(
                x=RAWLINGS_9_5_TIME_DAYS,
                y=conc['B'],
                unit_x='days',
                unit_y='mol/L',
                basis='molar concentration',
            ),
            'C': Measurement(
                x=RAWLINGS_9_5_TIME_DAYS,
                y=conc['C'],
                unit_x='days',
                unit_y='mol/L',
                basis='molar concentration',
            ),
        },
    )

    weights_df = pd.DataFrame(
        np.diag(RAWLINGS_9_5_WEIGHT_DIAG),
        index=list(MEASURED_SPECIES),
        columns=list(MEASURED_SPECIES),
    )  # [(L/mol)**2]

    def batch_reactor_callback(log10_params: np.ndarray, t_days: np.ndarray) -> np.ndarray:
        """Run ``BatchReactor`` and extract ``('A', 'B', 'C')`` concentrations.

        Parameters
        ----------
        log10_params : numpy.ndarray
            Base-10 logarithms of rate constants on a per-day basis [-], shape
            ``(6,)``, where reactions 1..5 are in ``1/day`` and reaction 6 is
            in ``L/(mol*day)``.
        t_days : numpy.ndarray
            Sample times [day], shape ``(num_times,)``.

        Returns
        -------
        numpy.ndarray
            Predicted molar concentrations of ``('A', 'B', 'C')`` [mol/L],
            shape ``(num_times, 3)``.
        """
        k_sec = (
            (10.0 ** np.asarray(log10_params, dtype=float)) / SECONDS_PER_DAY
        )  # [1/s] for rxn_1..rxn_5; [L/(mol*s)] for rxn_6
        kin_step = RxnKinetics(
            rawlings_purecomp_json,
            rxns=RAWLINGS_9_5_RXNS,
            k_params=k_sec,
            params_f=RAWLINGS_9_5_PARAMS_F,
        )
        liq_step = LiquidPhase(
            rawlings_purecomp_json,
            temp=ISOTHERMAL_TEMP_K,
            vol=INITIAL_VOLUME_M3,
            mole_conc=INITIAL_MOLE_CONC,
            mass_frac=INITIAL_SOLVENT_MASS_FRAC,
        )
        unit = BatchReactor(isothermal=True, return_sens=False)
        unit.Phases = liq_step
        unit.Kinetics = kin_step
        unit.solve_unit(
            time_grid=np.asarray(t_days, dtype=float) * SECONDS_PER_DAY,
            sundials_opts={'maxh': SECONDS_PER_DAY},
            verbose=False,
        )
        conc_step = unit.result.retrieve_state('mole_conc')
        return np.column_stack([conc_step[ky] for ky in MEASURED_SPECIES])

    estimator = ParameterEstimation(
        batch_reactor_callback,
        param_seed=RAWLINGS_9_5_LOG10_SEED.copy(),
        experiments=[exp],
        output_names=MEASURED_SPECIES,
        name_params=list(RAWLINGS_9_5_RXNS.keys()),
        weight_matrix=weights_df,
    )
    log_fit, _, _ = estimator.optimize_fn(
        method='scipy',
        optim_options={'max_nfev': 80},
        verbose=False,
    )
    k_fit = 10.0 ** log_fit  # [1/day] for rxn_1..rxn_5; [L/(mol*day)] for rxn_6
    np.testing.assert_allclose(
        k_fit[IDENTIFIABLE_INDICES],
        RAWLINGS_9_5_TRUE_RATES_DAY[IDENTIFIABLE_INDICES],
        rtol=IDENTIFIABLE_RTOL,
    )

    param_units_map = {name: '-' for name in RAWLINGS_9_5_RXNS.keys()}
    record = CalibrationRecord.from_estimation(
        estimator,
        {'exp1': exp},
        case='rawlings-9-5-batch-reactor-scipy',
        parameter_units=param_units_map,
        model=ModelRevision('BatchReactor.rawlings_9_5', revision='v1'),
        sources=[DataSource('rawlings_9_5', 'synthetic', 'test suite')],
        weight_matrix=weights_df,
    )
    replayed = record.rerun(batch_reactor_callback, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(replayed.params_convg, log_fit, rtol=1e-4, atol=1e-6)
