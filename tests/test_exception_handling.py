"""Regressions for narrowed exception handling in three package modules.

The crystallizer fallback test uses the real Assimulo problem in the optional
backend lane. Warning-attribution cases construct production crystallizers in
the core lane because they do not invoke a solver boundary. Bootstrap exception
tests inject failures at the user-model callback because this small linear fit
does not naturally construct a singular Levenberg--Marquardt system.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy import Crystallizers
from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
from PharmaPy.ThermoModule import ParseDatabase

pytestmark = pytest.mark.unit


def _statistics_with_failing_model(error):
    """Build real estimation and statistics objects for a model failure.

    Parameters
    ----------
    error : BaseException
        Exception raised by the user-supplied model after the nominal fit.

    Returns
    -------
    StatisticsClass
        Production bootstrap-statistics object configured with a fitted
        :class:`ParameterEstimation` collaborator.
    """
    failure_state = {"enabled": False}

    def linear_model(params, time):
        """Evaluate a fitted linear concentration model or its failure path.

        Parameters
        ----------
        params : numpy.ndarray
            Linear concentration rate [mol/L/s].
        time : numpy.ndarray
            Measurement times [s].

        Returns
        -------
        numpy.ndarray
            Predicted concentration [mol/L].

        Raises
        ------
        BaseException
            Requested model failure after the nominal fit is complete.
        """
        if failure_state["enabled"]:
            raise error
        return params[0] * time

    time = np.array([0.0, 1.0, 2.0])  # [s]
    # The deliberately imperfect fit gives residual bootstrapping a nonzero
    # error distribution instead of a degenerate all-zero sample.
    observed_concentration = np.array([0.1, 1.0, 2.1])  # [mol/L]
    fit_evaluation_cap = 5  # [-], sufficient for this one-parameter linear fit
    estimator = ParameterEstimation(
        linear_model,
        param_seed=np.array([1.0]),  # [mol/L/s]
        x_data=time,
        y_data=observed_concentration,
        name_params=["rate_mol_l_s"],
        name_states=["concentration_mol_l"],
    )
    estimator.optimize_fn(
        method="LM",
        verbose=False,
        optim_options={"max_fun_eval": fit_evaluation_cap},
    )
    statistics = StatisticsClass(estimator)
    failure_state["enabled"] = True
    return statistics


def test_parse_database_converts_numeric_fields_to_float_arrays(tmp_path):
    """Numeric properties become float arrays while retaining their basis.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the compound database fixture.
    """
    database = {
        "water": {"mw": 18.02, "cas": "7732-18-5"},
        "ethanol": {"mw": 46.07, "cas": "64-17-5"},
    }  # mw [g/mol]; CAS identifier [-]
    path = tmp_path / "compounds.json"
    path.write_text(json.dumps(database))

    parsed = ParseDatabase(str(path))

    expected_molecular_weights = [18.02, 46.07]  # [g/mol]
    assert isinstance(parsed["mw"], np.ndarray)
    assert parsed["mw"].dtype == np.float64
    np.testing.assert_allclose(sorted(parsed["mw"]), expected_molecular_weights)


def test_parse_database_keeps_non_numeric_fields_as_lists(tmp_path):
    """Text identifiers retain the established list representation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the compound database fixture.
    """
    database = {
        "water": {"mw": 18.02, "cas": "7732-18-5"},
        "ethanol": {"mw": 46.07, "cas": "64-17-5"},
    }  # mw [g/mol]; CAS identifier [-]
    path = tmp_path / "compounds.json"
    path.write_text(json.dumps(database))

    parsed = ParseDatabase(str(path))

    assert isinstance(parsed["cas"], list)
    assert sorted(parsed["cas"]) == ["64-17-5", "7732-18-5"]


def test_parse_database_propagates_float_overflow(tmp_path):
    """A float64 overflow is not mistaken for a non-numeric property.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the compound database fixture.
    """
    maximum_float64 = np.finfo(np.float64).max  # [g/mol], fixture boundary
    overflowing_molecular_weight = int(maximum_float64) * 2  # [g/mol]
    database = {
        "water": {"mw": overflowing_molecular_weight},
        "ethanol": {"mw": 46.07},
    }  # mw [g/mol]
    path = tmp_path / "compounds.json"
    path.write_text(json.dumps(database))

    with pytest.raises(OverflowError, match="too large to convert to float"):
        ParseDatabase(str(path))


def test_parse_database_propagates_malformed_nested_values(tmp_path):
    """Malformed nested property values retain their diagnostic TypeError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the compound database fixture.
    """
    malformed_molecular_weight = {"unexpected": 18.02}  # value [g/mol]
    database = {
        "water": {"mw": {"value": malformed_molecular_weight}},
        "ethanol": {"mw": {"value": 46.07}},
    }  # mw [g/mol]
    path = tmp_path / "compounds.json"
    path.write_text(json.dumps(database))

    with pytest.raises(TypeError, match="float"):
        ParseDatabase(str(path))


def test_bootstrap_params_records_nan_rows_for_linear_algebra_failures():
    """User-model linear-algebra failures warn and produce NaN rows."""
    stats = _statistics_with_failing_model(
        np.linalg.LinAlgError("singular bootstrap Hessian")
    )

    with pytest.warns(RuntimeWarning, match="singular bootstrap Hessian") as caught:
        boot_params = stats.bootstrap_params(num_samples=3)

    assert len(caught) == 3
    for index, warning in enumerate(caught):
        assert "sample {}".format(index) in str(warning.message)
    assert boot_params.shape == (3, 1)
    assert np.isnan(boot_params).all()


def test_bootstrap_params_propagates_programming_errors():
    """User-model programming errors are not converted to NaN rows."""
    stats = _statistics_with_failing_model(
        AttributeError("missing optimizer state")
    )

    with pytest.raises(AttributeError, match="missing optimizer state"):
        stats.bootstrap_params(num_samples=3)


def test_bootstrap_params_does_not_swallow_keyboard_interrupt():
    """User interrupts continue to abort a bootstrap run immediately."""
    stats = _statistics_with_failing_model(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        stats.bootstrap_params(num_samples=3)


@pytest.mark.assimulo
def test_batch_cryst_ad_fallback_configures_finite_difference_problem():
    """Unsupported AD requests configure an operational NumPy fallback."""
    assimulo_problem = pytest.importorskip("assimulo.problem")

    with pytest.warns(RuntimeWarning, match="finite-difference") as caught:
        crystallizer = Crystallizers.BatchCryst(
            target_comp="solute", jac_type="AD"
        )

    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert crystallizer.jac_type == "finite_diff"

    initial_states = np.array([1.0])  # [-], unevaluated handoff sentinel
    kinetic_params = np.array([2.0])  # [-], unevaluated handoff sentinel
    problem = crystallizer.set_ode_problem(
        eval_sens=True,
        states_init=initial_states,
        params_mergd=kinetic_params,
        jacv_prod=False,
    )

    assert isinstance(problem, assimulo_problem.Explicit_Problem)
    assert problem.jac == crystallizer.jac_states_numerical
    assert crystallizer.jac_params_fn == crystallizer.jac_params_numerical
    assert problem.rhs_sens == crystallizer.rhs_sensitivity


@pytest.mark.parametrize(
    "crystallizer_name",
    ("BatchCryst", "MSMPR", "SemibatchCryst"),
)
def test_ad_fallback_warning_points_at_caller(crystallizer_name):
    """The AD fallback warning is attributed to the constructing caller.

    Covers all three public crystallizers because they sit at different depths:
    ``BatchCryst`` and ``MSMPR`` derive from ``_BaseCryst`` directly, while
    ``SemibatchCryst`` subclasses ``MSMPR``. A fixed ``stacklevel`` is therefore
    correct for at most one of these hierarchies, and pointed at
    ``Crystallizers.py`` rather than at user code for ``SemibatchCryst``.

    Parameters
    ----------
    crystallizer_name : str
        Attribute name of the public crystallizer class under test.
    """
    crystallizer_class = getattr(Crystallizers, crystallizer_name)

    with pytest.warns(RuntimeWarning, match="finite-difference") as caught:
        construction_line = inspect.currentframe().f_lineno + 1
        crystallizer = crystallizer_class(target_comp="solute", jac_type="AD")

    # Assert the exact construction line, not only this file: a stacklevel that
    # is short by one frame still lands inside Crystallizers.py, but one that is
    # too deep could land elsewhere in this module and pass a file-only check.
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert caught[0].lineno == construction_line
    assert crystallizer.jac_type == "finite_diff"
