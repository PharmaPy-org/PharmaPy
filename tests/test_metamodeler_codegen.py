"""Regression tests for MetaModeler-generated unit-operation templates."""

from importlib import util
import sys
from types import ModuleType

import numpy as np
import pytest

from PharmaPy.MetaModeler import MetaModelingClass


pytestmark = pytest.mark.unit


def _install_fake_assimulo(monkeypatch):
    """Install minimal Assimulo modules needed to import generated templates.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Active pytest monkeypatch fixture used to restore ``sys.modules``.

    Returns
    -------
    None
    """
    assimulo = ModuleType("assimulo")
    problem = ModuleType("assimulo.problem")
    solvers = ModuleType("assimulo.solvers")

    problem.Explicit_Problem = object
    problem.Implicit_Problem = object
    solvers.CVode = object
    solvers.IDA = object

    monkeypatch.setitem(sys.modules, "assimulo", assimulo)
    monkeypatch.setitem(sys.modules, "assimulo.problem", problem)
    monkeypatch.setitem(sys.modules, "assimulo.solvers", solvers)


def _generated_class(tmp_path, monkeypatch, model_type, has_stages=False):
    """Generate and import a template class for one model type.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory where the generated Python module is written.
    monkeypatch : pytest.MonkeyPatch
        Active pytest monkeypatch fixture used to install fake imports.
    model_type : {"ODE", "DAE", "PDE"}
        Template model type emitted by :class:`MetaModelingClass`.
    has_stages : bool, default=False
        Whether the generated template declares staged state handling [-].

    Returns
    -------
    type
        Generated unit-operation class.
    """
    _install_fake_assimulo(monkeypatch)

    module_path = tmp_path / f"generated_{model_type.lower()}.py"
    generator = MetaModelingClass(
        module_path,
        f"Generated{model_type}",
        model_type=model_type,
        name_states=["material", "energy"],
        has_stages=has_stages,
    )

    generator.CreatePharmaPyTemplate()

    spec = util.spec_from_file_location(module_path.stem, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, f"Generated{model_type}")


@pytest.mark.parametrize("model_type", ["ODE", "DAE", "PDE"])
def test_generated_unit_model_concatenates_material_and_energy_rates(
        tmp_path, monkeypatch, model_type):
    generated_class = _generated_class(tmp_path, monkeypatch, model_type)

    if model_type == "PDE":
        unit = generated_class(num_nodes=3)
        unit.num_states = 3
        state_vector = np.array([
            1.0, 2.0, 300.0,
            3.0, 4.0, 310.0,
            5.0, 6.0, 320.0,
        ])  # [-], [-], [K] repeated by finite-volume node
        expected_material = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        expected_energy = [[300.0], [310.0], [320.0]]  # [K]
        material_rate = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ])  # [1/s]
        energy_rate = np.array([[3.0], [4.0], [5.0]])  # [K/s]
        expected_rates = [0.1, 0.2, 3.0, 0.3, 0.4, 4.0, 0.5, 0.6, 5.0]
    else:
        unit = generated_class()
        state_vector = np.array([1.0, 2.0, 300.0])  # [-], [-], [K]
        expected_material = [1.0, 2.0]
        expected_energy = [300.0]  # [K]
        material_rate = np.array([0.1, 0.2])  # [1/s]
        energy_rate = np.array([3.0])  # [K/s]
        expected_rates = [0.1, 0.2, 3.0]

    unit.acum_len = np.array([2])  # [-]

    def material_balances(time, material, energy):
        """Return a known material-rate vector from generated state inputs.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        material : numpy.ndarray
            Material state values [-].
        energy : numpy.ndarray
            Energy state values [K].

        Returns
        -------
        numpy.ndarray
            Material-state rates [1/s].
        """
        assert time == pytest.approx(5.0)
        np.testing.assert_allclose(material, expected_material)
        np.testing.assert_allclose(energy, expected_energy)
        return material_rate

    def energy_balances(time, material, energy):
        """Return a known energy-rate vector from generated state inputs.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        material : numpy.ndarray
            Material state values [-].
        energy : numpy.ndarray
            Energy state values [K].

        Returns
        -------
        numpy.ndarray
            Energy-state rates [K/s].
        """
        assert time == pytest.approx(5.0)
        np.testing.assert_allclose(material, expected_material)
        np.testing.assert_allclose(energy, expected_energy)
        return energy_rate

    unit.material_balances = material_balances
    unit.energy_balances = energy_balances

    rates = unit.unit_model(5.0, state_vector)

    np.testing.assert_allclose(rates, expected_rates)


@pytest.mark.parametrize("model_type", ["ODE", "DAE"])
def test_generated_staged_unit_model_flattens_stage_rate_blocks(
        tmp_path, monkeypatch, model_type):
    generated_class = _generated_class(tmp_path, monkeypatch, model_type,
                                       has_stages=True)
    unit = generated_class(num_stages=3)
    unit.len_states_orig = np.array([1, 1])  # [-]
    unit.acum_len = np.array([3])  # [-]
    state_vector = np.array([1.0, 2.0, 3.0, 300.0, 310.0, 320.0])
    material_rate = np.array([[0.1], [0.2], [0.3]])  # [1/s]
    energy_rate = np.array([[3.0], [4.0], [5.0]])  # [K/s]

    def material_balances(time, material, energy):
        """Return staged material-rate blocks from staged state inputs.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        material : numpy.ndarray
            Per-stage material state values [-].
        energy : numpy.ndarray
            Per-stage energy state values [K].

        Returns
        -------
        numpy.ndarray
            Per-stage material-state rates [1/s].
        """
        assert time == pytest.approx(5.0)
        np.testing.assert_allclose(material, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(energy, [300.0, 310.0, 320.0])
        return material_rate

    def energy_balances(time, material, energy):
        """Return staged energy-rate blocks from staged state inputs.

        Parameters
        ----------
        time : float
            Evaluation time [s].
        material : numpy.ndarray
            Per-stage material state values [-].
        energy : numpy.ndarray
            Per-stage energy state values [K].

        Returns
        -------
        numpy.ndarray
            Per-stage energy-state rates [K/s].
        """
        assert time == pytest.approx(5.0)
        np.testing.assert_allclose(material, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(energy, [300.0, 310.0, 320.0])
        return energy_rate

    unit.material_balances = material_balances
    unit.energy_balances = energy_balances

    rates = unit.unit_model(5.0, state_vector)

    if model_type == "ODE":
        expected_rates = [0.1, 0.2, 0.3, 3.0, 4.0, 5.0]
    else:
        # Current generated staged DAE behavior is pinned here for the
        # #82 flattening fix; #196 tracks aligning state and residual order.
        expected_rates = [0.1, 3.0, 0.2, 4.0, 0.3, 5.0]

    np.testing.assert_allclose(rates, expected_rates)


def test_generated_solve_model_declares_cumulative_lengths_for_helpers(
        tmp_path, monkeypatch):
    generated_class = _generated_class(tmp_path, monkeypatch, "ODE")
    unit = generated_class()

    with pytest.raises(ValueError, match="need at least one array"):
        unit.solve_model(np.array([0.0, 1.0]))

    np.testing.assert_array_equal(unit.acum_len, [1])
    np.testing.assert_array_equal(unit.len_states, [1, 1])
    assert unit.num_states == 2


@pytest.mark.parametrize("model_type", ["ODE", "DAE"])
def test_generated_retrieve_results_splits_states_by_instance_lengths(
        tmp_path, monkeypatch, model_type):
    generated_class = _generated_class(tmp_path, monkeypatch, model_type)
    unit = generated_class()
    unit.acum_len = np.array([2])  # [-]
    states_by_time = np.array([
        [1.0, 2.0, 300.0],
        [3.0, 4.0, 310.0],
    ])  # [-], [-], [K]

    result = unit.retrieve_results(np.array([0.0, 1.0]), states_by_time)

    assert list(result) == ["material", "energy"]
    assert unit.outputs is result
    np.testing.assert_allclose(result["material"], states_by_time[:, :2])
    np.testing.assert_allclose(result["energy"], states_by_time[:, 2:])


def test_generated_pde_retrieve_results_returns_reordered_outputs(
        tmp_path, monkeypatch):
    generated_class = _generated_class(tmp_path, monkeypatch, "PDE")
    unit = generated_class(num_nodes=2)
    unit.len_states = np.array([2, 1])  # [-]
    states_by_time = np.array([
        [1.0, 2.0, 300.0, 3.0, 4.0, 310.0],
        [5.0, 6.0, 320.0, 7.0, 8.0, 330.0],
    ])  # [-], [-], [K] repeated by finite-volume node

    states_per_node, individual_states = unit.retrieve_results(
        np.array([0.0, 1.0]), states_by_time)

    assert len(states_per_node) == 2
    assert unit.outputs[0] is states_per_node
    assert unit.outputs[1] is individual_states
    np.testing.assert_allclose(states_per_node[0][0], states_by_time[:, :2])
    np.testing.assert_allclose(states_per_node[1][1], states_by_time[:, 5:])
    assert list(individual_states) == ["material", "energy"]
    np.testing.assert_allclose(individual_states["material"][0],
                               [[1.0, 3.0], [5.0, 7.0]])
    np.testing.assert_allclose(individual_states["energy"],
                               [[300.0, 310.0], [320.0, 330.0]])
