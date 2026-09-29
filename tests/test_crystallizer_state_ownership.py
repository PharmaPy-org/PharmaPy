"""#312 solver-owned concentration buffers with real phases and CVode.

The deliberately displaced solvent is independent of the phase volume closure.
Reported states remain integrated values; named-solvent phase objects retain
their existing ideal-volume completion contract. This does not validate the
physical equivalence of those two representations.
"""
import json
import numpy as np
import pytest
from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase
from test_crystallizer_moment_inventory import inventory_unit

RTOL = 1e-12  # [-], allowance for direct converter arithmetic roundoff
SOLVENT_OFFSET = 5.0  # [kg/m**3], synthetic off-closure state exposes aliasing


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR, SemibatchCryst])
@pytest.mark.parametrize('named', [False, True])
def test_rhs_does_not_write_caller_concentrations(data_path, unit_type, named):
    """Keep solver buffers isolated from property-phase completion.

    Parameters
    ----------
    data_path : dict
        Public thermodynamic database paths.
    unit_type : type
        Batch, semibatch or continuous crystallizer.
    named : bool
        Enable ideal-volume solvent completion in the owned phase.
    """
    unit, states = inventory_unit(data_path, unit_type)
    unit.Liquid_1.ind_solv = 4 if named else None
    states[8] += SOLVENT_OFFSET  # [kg/m**3], solvent is the fifth species
    original = states.copy()  # [mixed state units], ownership oracle
    derivative = unit.unit_model(0.0, states)  # [state unit/s]
    assert np.isfinite(derivative).all()
    np.testing.assert_array_equal(states, original)
    assert not np.shares_memory(unit.Liquid_1.mass_conc, states)


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR, SemibatchCryst])
@pytest.mark.parametrize('named', [False, True])
def test_retrieval_preserves_all_concentration_rows(data_path, unit_type, named):
    """Retain integrated profiles while explicitly constructing the phase outlet.

    Parameters
    ----------
    data_path : dict
        Public thermodynamic database paths.
    unit_type : type
        Batch, semibatch or continuous crystallizer.
    named : bool
        Enable ideal-volume solvent completion in the owned phase.
    """
    unit, initial = inventory_unit(data_path, unit_type)
    unit.Liquid_1.ind_solv = 4 if named else None
    times = np.array([0.0, 0.01, 0.03])  # [s], asymmetric three-row profile
    states = np.tile(initial, (3, 1))  # [mixed state units]
    states[:, 8] += SOLVENT_OFFSET * np.arange(1, 4)  # [kg/m**3]
    original = states[:, 4:9].copy()  # [kg/m**3], every solver concentration
    unit.retrieve_results(times, states)
    np.testing.assert_array_equal(states[:, 4:9], original)
    np.testing.assert_array_equal(unit.result.mass_conc, original)
    np.testing.assert_array_equal(unit.outputs['mass_conc'], original)
    np.testing.assert_array_equal(unit.profiles_runs[0]['mass_conc'], original)
    expected = original[-1].copy()  # [kg/m**3], phase-owned representation
    if named:
        densities, _ = unit.Liquid_1.getDensityPure(phase='liquid')  # [kg/m**3]
        expected[4] = densities[4] * (1 - np.sum(expected[:4] / densities[:4]))
    np.testing.assert_allclose(unit.Outlet.Liquid_1.mass_conc, expected,
                               rtol=RTOL, atol=0)
    assert not np.shares_memory(unit.Liquid_1.mass_conc, states)


@pytest.mark.assimulo
@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('equal_density', [False, True])
def test_real_growth_solve_publishes_raw_concentrations(data_path, named, equal_density, tmp_path):
    """Preserve actual native-solver reporting values through retrieval.

    Parameters
    ----------
    data_path : dict
        Public thermodynamic database paths.
    named : bool
        Enable the solvent completion contract; False is the control.
    equal_density : bool
        Replace every liquid density with the same synthetic 1000 kg/m**3.
    tmp_path : Path
        Isolated directory for that explicit diagnostic property variant.
    """
    pytest.importorskip('assimulo')
    # Exact public issue #312 growth fixture, plus named/unnamed and
    # equal-density controls. No private exercise values are used.
    path = data_path['flowsheet'] / 'compound_database.json'
    if equal_density:
        properties = json.loads(path.read_text())
        for component in properties.values():
            component['rho_liq'] = 1000.0  # [kg/m**3], equal-density diagnostic
        path = tmp_path/'equal_density.json'
        path.write_text(json.dumps(properties))
    temperature = 310.0  # [K], issue #312's isothermal diagnostic
    volume = 2.0  # [m**3], issue's liquid charge
    grid = np.geomspace(1.0,500.0,40)  # [um], issue's seed grid
    population = 1e7*np.exp(-((grid-100.0)/40.0)**2)  # [#/um], issue's Gaussian seed
    unit = BatchCryst('A',method='moments',controls={
        'temp':lambda time:temperature+np.zeros_like(time)})
    liquid = LiquidPhase(str(path),temp=temperature,vol=volume,
                         mass_frac=[.1,.1,.1,.1,.6],name_solv='solvent' if named else None)
    # [-], issue's unequal solute/solvent fractions
    solid = SolidPhase(str(path),temp=temperature,x_distrib=grid,
                       distrib=population,kv=.5,mass_frac=[1,0,0,0,0])
    # [-], issue's non-unit shape factor and pure-A solid composition
    unit.Phases = (liquid,solid)
    unit.Kinetics = CrystKinetics(coeff_solub=[40.0],
                                 growth=(1.0, 0.0, 1.0))
    # [kg/m**3], [um/s], [J/mol], [-]; isothermal seeded diagnostic growth
    original_retrieve = unit.retrieve_results
    captured = {}

    def capture(time, states):
        """Observe the real solver handoff before normal result retrieval.

        Parameters
        ----------
        time : ndarray
            Reporting times [s].
        states : ndarray
            Moments [um**n], concentrations [kg/m**3], liquid volume [m**3].
        """
        captured['concentration'] = states[:, 4:9].copy()  # [kg/m**3]
        original_retrieve(time, states)

    unit.retrieve_results = capture
    times = np.linspace(0.0, 6000.0, 61)  # [s], exact issue #312 reporting interval
    _, states = unit.solve_unit(time_grid=times, verbose=False)
    assert unit.result.mass_conc[-1, 0] < unit.result.mass_conc[0, 0]
    np.testing.assert_array_equal(states[:, 4:9], captured['concentration'])
    np.testing.assert_array_equal(unit.result.mass_conc, captured['concentration'])
