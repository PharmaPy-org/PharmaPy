"""Regressions for narrowed exception handling in three package modules.

The crystallizer test uses the shared Assimulo import helper because the core
test lane intentionally excludes that optional solver. Its monkeypatching is
limited to the optional import and problem-construction boundaries; the real
``BatchCryst.set_ode_problem`` routing remains under test.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from assimulo_helpers import import_module_with_assimulo_stub
from PharmaPy.StatsModule import StatisticsClass
from PharmaPy.ThermoModule import ParseDatabase

pytestmark = pytest.mark.unit


class _FakeEstimationInstance:
    """Minimal parameter-estimation collaborator for bootstrap tests."""

    def __init__(self, optimize_fn):
        """Store the optimizer callable used by ``bootstrap_params``.

        Parameters
        ----------
        optimize_fn : callable
            Optimizer double returning model parameters in their configured
            physical units or raising a test-specific exception.
        """
        self.num_params = 2
        self.opt_method = "LM"
        self.optim_options = {}
        self.y_data = None
        self.optimize_fn = optimize_fn


class _BootstrapStatistics(StatisticsClass):
    """Statistics test double with deterministic dimensionless samples."""

    def __init__(self, optimize_fn):
        """Initialize only the state needed for bootstrap optimization.

        Parameters
        ----------
        optimize_fn : callable
            Optimizer double returning model parameters in their configured
            physical units or raising a test-specific exception.
        """
        self.inst = _FakeEstimationInstance(optimize_fn)

    def get_bootsamples(self, num_samples, fix_initial=False):
        """Return one deterministic response sample per bootstrap iteration.

        Parameters
        ----------
        num_samples : int
            Number of bootstrap samples to construct.
        fix_initial : bool, optional
            Unused compatibility argument matching the production helper.

        Returns
        -------
        list of numpy.ndarray
            One synthetic response vector with ``num_samples`` entries [-].
        """
        del fix_initial
        response_samples = np.zeros(num_samples)  # [-]
        return [response_samples]


class _ExplicitProblemStub:
    """Record the sensitivity callbacks assigned at the Assimulo boundary."""

    def __init__(self, rhs, y0, t0, p0):
        """Store the ODE problem inputs without invoking an optional solver.

        Parameters
        ----------
        rhs : callable
            Crystallizer right-hand-side function.
        y0 : numpy.ndarray
            Initial model state vector [state-dependent units].
        t0 : float
            Initial simulation time [s].
        p0 : numpy.ndarray
            Kinetic parameter vector [parameter-dependent units].
        """
        self.rhs = rhs
        self.y0 = y0  # [state-dependent units]
        self.t0 = t0  # [s]
        self.p0 = p0  # [parameter-dependent units]
        self.jac = None
        self.rhs_sens = None


def _import_crystallizers(monkeypatch):
    """Import crystallizers with only optional Assimulo boundaries replaced.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Cleanup fixture for temporary module and problem-class substitutions.

    Returns
    -------
    module
        Imported ``PharmaPy.Crystallizers`` module.

    Notes
    -----
    Replacing ``Explicit_Problem`` keeps this regression in the core lane and
    lets it assert the exact callbacks configured by the real
    ``set_ode_problem`` method. Solver execution remains covered by the
    Assimulo-marked integration lane.
    """
    module = import_module_with_assimulo_stub(
        monkeypatch,
        "PharmaPy.Crystallizers",
        solvers={"CVode": object},
        problem={"Explicit_Problem": _ExplicitProblemStub},
    )
    monkeypatch.setattr(module, "Explicit_Problem", _ExplicitProblemStub)
    return module


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
    """Singular optimizer systems warn and produce NaN parameter rows."""

    def failing_optimize(**kwargs):
        """Raise the documented NumPy failure for a singular LM system.

        Parameters
        ----------
        **kwargs
            Optimizer options accepted but unused by this test double.

        Raises
        ------
        numpy.linalg.LinAlgError
            Always, to represent a singular approximate Hessian.
        """
        del kwargs
        raise np.linalg.LinAlgError("singular bootstrap Hessian")

    stats = _BootstrapStatistics(failing_optimize)

    with pytest.warns(RuntimeWarning, match="singular bootstrap Hessian") as caught:
        boot_params = stats.bootstrap_params(num_samples=3)

    assert len(caught) == 3
    for index, warning in enumerate(caught):
        assert "sample {}".format(index) in str(warning.message)
    assert boot_params.shape == (3, 2)
    assert np.isnan(boot_params).all()


def test_bootstrap_params_propagates_programming_errors():
    """Unrelated optimizer programming errors are not converted to NaNs."""

    def broken_optimize(**kwargs):
        """Raise a programming error unrelated to numerical convergence.

        Parameters
        ----------
        **kwargs
            Optimizer options accepted but unused by this test double.

        Raises
        ------
        AttributeError
            Always, to represent an invalid optimizer implementation.
        """
        del kwargs
        raise AttributeError("missing optimizer state")

    stats = _BootstrapStatistics(broken_optimize)

    with pytest.raises(AttributeError, match="missing optimizer state"):
        stats.bootstrap_params(num_samples=3)


def test_bootstrap_params_does_not_swallow_keyboard_interrupt():
    """User interrupts continue to abort a bootstrap run immediately."""

    def interrupted_optimize(**kwargs):
        """Raise a user interrupt from the optimizer boundary.

        Parameters
        ----------
        **kwargs
            Optimizer options accepted but unused by this test double.

        Raises
        ------
        KeyboardInterrupt
            Always, to emulate a user cancelling optimization.
        """
        del kwargs
        raise KeyboardInterrupt

    stats = _BootstrapStatistics(interrupted_optimize)

    with pytest.raises(KeyboardInterrupt):
        stats.bootstrap_params(num_samples=3)


def test_batch_cryst_ad_fallback_configures_finite_difference_problem(monkeypatch):
    """Unsupported AD requests configure an operational NumPy fallback.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Cleanup fixture for the optional Assimulo problem boundary.
    """
    crystallizers = _import_crystallizers(monkeypatch)

    with pytest.warns(RuntimeWarning, match="finite-difference") as caught:
        crystallizer = crystallizers.BatchCryst(target_comp="solute", jac_type="AD")

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

    assert problem.jac == crystallizer.jac_states_numerical
    assert crystallizer.jac_params_fn == crystallizer.jac_params_numerical
    assert problem.rhs_sens == crystallizer.rhs_sensitivity


@pytest.mark.parametrize(
    "crystallizer_name",
    ("BatchCryst", "MSMPR", "SemibatchCryst"),
)
def test_ad_fallback_warning_points_at_caller(crystallizer_name, monkeypatch):
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
    monkeypatch : pytest.MonkeyPatch
        Cleanup fixture for the optional Assimulo problem boundary.
    """
    crystallizers = _import_crystallizers(monkeypatch)
    crystallizer_class = getattr(crystallizers, crystallizer_name)

    with pytest.warns(RuntimeWarning, match="finite-difference") as caught:
        construction_line = inspect.currentframe().f_lineno + 1
        crystallizer = crystallizer_class(target_comp="solute", jac_type="AD")

    # Assert the exact construction line, not only this file: a stacklevel that
    # is short by one frame still lands inside Crystallizers.py, but one that is
    # too deep could land elsewhere in this module and pass a file-only check.
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert caught[0].lineno == construction_line
    assert crystallizer.jac_type == "finite_diff"
