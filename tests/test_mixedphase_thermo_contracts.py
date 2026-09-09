"""B004 regressions using real phases and no optional solver backend.

The shipped five-species database supplies unequal phase properties. These
tests intercept solver initialization at the ODE boundary. Moment-mode solves
and inventory retrieval are covered in test_crystallizer_moment_basis.py and
test_crystallizer_moment_inventory.py.
"""

from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Crystallizers import BatchCryst, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Cake, Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SolidLiquidSep import Filter, get_alpha, get_sat_inf

pytestmark = pytest.mark.unit

LIQUID_COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], solvent-rich
SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A
KV = 0.5  # [-], distinguishes crystal third moment from physical volume
RTOL = 1e-12  # [-], roundoff allowance for short float64 property calculations
TEMPERATURE = 310.0  # [K], above the enthalpy reference so weights are visible


@pytest.fixture
def thermo_path(data_path):
    """Return the shipped five-species property file.

    Parameters
    ----------
    data_path : dict
        Repository test-data paths.

    Returns
    -------
    str
        Thermophysical database path.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


def make_slurry(thermo_path, solid_fraction=0.1):
    """Build a litre of slurry with a prescribed physical solids fraction.

    Parameters
    ----------
    thermo_path : str
        Database path.
    solid_fraction : float, optional
        Solid volume fraction [-]; default ten percent.

    Returns
    -------
    Slurry
        Real liquid/solid mixture with total volume 1e-3 m**3.
    """
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                         temp=TEMPERATURE)
    # Moments construct the #159 inventory case; only the third fixes volume.
    moments = np.array([1e12, 1e6, 1e3, solid_fraction / KV])  # [m**n/m**3]
    solid = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION,
                       moments=moments, kv=KV, temp=TEMPERATURE)
    slurry = Slurry(vol=1e-3, moments=moments)  # [m**3]
    slurry.Phases = [liquid, solid]
    return slurry


def test_total_volume_uses_normalized_moment(thermo_path):
    slurry = make_slurry(thermo_path)
    assert slurry.Solid_1.moments[3] == pytest.approx(2e-4, rel=RTOL, abs=0)
    assert slurry.Liquid_1.vol == pytest.approx(9e-4, rel=RTOL, abs=0)
    assert slurry.getTotalVol() == pytest.approx(1e-3, rel=RTOL, abs=0)


@pytest.mark.parametrize('has_distribution', [False, True])
def test_phase_only_slurry_normalizes_total_solid_moments(thermo_path, has_distribution):
    # Same #159 litre-scale inventory, supplied through real phase amounts.
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                         vol=9e-4, temp=TEMPERATURE)  # [m**3]
    total_moments = np.array([1e9, 1e3, 1.0, 2e-4])  # [m**n], order n
    # Retained number densities without a size grid; moments are supplied
    # independently, as the SolidPhase constructor permits.
    distribution = np.array([2.0, 5.0, 1.0]) if has_distribution else None  # [#/um]
    solid = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION,
                       moments=total_moments, distrib=distribution,
                       kv=KV, temp=TEMPERATURE)
    slurry = Slurry()
    slurry.Phases = [liquid, solid]
    assert slurry.getTotalVol() == pytest.approx(1e-3, rel=RTOL, abs=0)
    assert slurry.moments[3] == pytest.approx(0.2, rel=RTOL, abs=0)
    np.testing.assert_allclose(slurry.moments, total_moments / 1e-3,
                               rtol=RTOL, atol=0)
    np.testing.assert_array_equal(solid.moments, total_moments)
    if has_distribution:
        expected_distribution = [2000.0, 5000.0, 1000.0]  # [#/m**3/um], divide by 1 L
        np.testing.assert_allclose(slurry.distrib, expected_distribution,
                                   rtol=RTOL, atol=0)
        np.testing.assert_array_equal(solid.distrib, distribution)
    else:
        assert slurry.distrib is None


@pytest.mark.parametrize('population', ['zero_moments', 'stored_distribution',
                                      'nonzero_count'])
def test_phase_only_slurry_rejects_zero_volume_before_normalization(
        thermo_path, population):
    """Reject empty phase inventories without corrupting their populations.

    Parameters
    ----------
    thermo_path : str
        Thermophysical database path.
    population : str
        Zero total moments [m**n], a stored zero distribution [#/um], or
        one zero-size nucleus with nonzero count but zero material volume.
    """
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                             temp=TEMPERATURE)
    moments = np.zeros(4)  # [m**n], orders zero through three, no solid volume
    if population == 'nonzero_count':
        moments[0] = 1.0  # [#], one zero-size nucleus still has no material volume
    distribution = np.zeros(3) if population == 'stored_distribution' else None
    # [#/um], optional retained zero population without a size grid
    solid = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION,
                       moments=moments.copy(), distrib=distribution,
                       kv=KV, temp=TEMPERATURE)
    slurry = Slurry()
    with np.errstate(divide='raise', invalid='raise'):
        with pytest.raises(ValueError, match='zero combined phase volume'):
            slurry.Phases = [liquid, solid]
    assert slurry.moments is None
    assert slurry.distrib is None
    np.testing.assert_array_equal(solid.moments, moments)
    if distribution is not None:
        np.testing.assert_array_equal(solid.distrib, distribution)


def test_phase_only_slurry_accepts_liquid_without_crystals(thermo_path):
    """Keep a positive liquid inventory usable with a zero crystal population.

    Parameters
    ----------
    thermo_path : str
        Thermophysical database path.
    """
    liquid_volume = 1e-3  # [m**3], one litre of crystal-free liquid
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                         vol=liquid_volume, temp=TEMPERATURE)
    moments = np.zeros(4)  # [m**n], no crystals
    solid = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION,
                       moments=moments, kv=KV, temp=TEMPERATURE)
    slurry = Slurry()
    with np.errstate(divide='raise', invalid='raise'):
        slurry.Phases = [liquid, solid]
    assert slurry.vol == pytest.approx(liquid_volume, rel=RTOL, abs=0)
    assert slurry.temp == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    np.testing.assert_array_equal(slurry.moments, moments)
    np.testing.assert_array_equal(slurry.getFractions(), [1.0, 0.0])


def test_total_volume_uses_reconciled_phase_volumes(thermo_path):
    slurry = make_slurry(thermo_path)
    # A phase inventory update changes liquid volume independently of stored
    # slurry moments. Total volume must follow the current phase amounts.
    slurry.Liquid_1.updatePhase(vol=1.8e-3)  # [m**3], double the initial liquid
    assert slurry.getTotalVol() == pytest.approx(1.9e-3, rel=RTOL, abs=0)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('solid_fraction', [0.0, 0.1, 0.4])
def test_slurry_capacitance_is_sum_of_real_phases(thermo_path, solid_fraction, basis):
    slurry = make_slurry(thermo_path, solid_fraction)
    total_capacity = sum(phase.mass * phase.getCp(TEMPERATURE, basis='mass')
                         for phase in slurry.Phases)  # [J/K]
    for liquid_basis in (False, True):
        volume = slurry.Liquid_1.vol if liquid_basis else slurry.vol  # [m**3]
        if basis == 'mole':
            volume *= 1000  # [L], exact conversion from m**3 to match mol/L
        actual = slurry.getCp(TEMPERATURE, times_vliq=liquid_basis,
                               basis=basis)  # [J/m**3/K] mass or [J/L/K] mole
        assert actual * volume == pytest.approx(total_capacity, rel=RTOL, abs=0)


class OdeBoundaryReached(Exception):
    """Stop after the real solve_unit packs its initial state."""


@pytest.mark.parametrize('unit_type', [BatchCryst, SemibatchCryst])
def test_crystallizer_initial_state_contains_liquid_volume(
        thermo_path, monkeypatch, unit_type):
    """Seed liquid volume and micrometre moments at the solver boundary.

    Parameters
    ----------
    thermo_path : str
        Thermophysical database path.
    monkeypatch : pytest.MonkeyPatch
        Replace only solver construction.
    unit_type : type
        Batch or semibatch crystallizer class.
    """
    slurry = make_slurry(thermo_path)
    unit = unit_type(target_comp='A', method='moments', adiabatic=True,
                     vol_tank=slurry.vol)
    unit.Phases = slurry
    unit.Kinetics = CrystKinetics()
    captured = []

    def capture_problem(eval_sens, states_init, params, jac_v_prod):
        """Capture packed states and stop before constructing a solver.

        Parameters
        ----------
        eval_sens, jac_v_prod : bool
            Solver options.
        states_init : ndarray
            Total moments [um**n], concentrations [kg/m**3], volume [m**3],
            and temperature [K], in the crystallizer's named state order.
        params : ndarray
            Kinetic parameters in the kinetics provider's units.

        Raises
        ------
        OdeBoundaryReached
            Always, after recording the actual initial state.
        """
        captured.append(states_init.copy())
        raise OdeBoundaryReached

    monkeypatch.setattr(unit, 'set_ode_problem', capture_problem)
    with pytest.raises(OdeBoundaryReached):
        unit.solve_unit(runtime=1.0)  # [s], initialization only
    assert len(captured) == 1
    volume_index = unit.num_distr + unit.num_species
    assert captured[0][volume_index] == pytest.approx(9e-4, rel=RTOL, abs=0)
    moments = slurry.Solid_1.moments  # [m**n], total phase moments
    expected_moments = np.array([moments[0], moments[1] * 1e6,
                                 moments[2] * 1e12, moments[3] * 1e18])
    # [um**n], exact SI phase to raw solver-state length conversion (#224)
    np.testing.assert_allclose(captured[0][:unit.num_distr],
                               expected_moments, rtol=RTOL, atol=0)


@pytest.mark.parametrize('unit_type', [BatchCryst, SemibatchCryst])
def test_crystallizer_energy_storage_uses_real_phase_capacitances(
        thermo_path, unit_type):
    slurry = make_slurry(thermo_path)
    unit = unit_type(target_comp='A', method='moments', adiabatic=True)
    unit.Phases = slurry
    unit.diam_tank = 0.1  # [m], arbitrary vessel geometry; no heat transfer
    unit.area_base = np.pi * unit.diam_tank**2 / 4  # [m**2]
    densities = slurry.getDensity()  # [kg/m**3], liquid then solid
    capacity = sum(phase.mass * phase.getCp(TEMPERATURE, basis='mass')
                   for phase in slurry.Phases)  # [J/K]
    crystallization_rate = 1e-5  # [kg/s], nonzero heat source probe
    heat_release = 1.46e4 * crystallization_rate  # [J/s], existing latent model
    kwargs = dict(time=0.0, params={}, cryst_rate=crystallization_rate,
                  u_inputs={'Inlet': {'vol_flow': 0.0}},
                  rhos=densities, mu_n=slurry.Solid_1.moments,
                  distrib=None, mass_conc=slurry.Liquid_1.mass_conc,
                  temp=TEMPERATURE, temp_ht=None, vol=slurry.Liquid_1.vol)
    if unit_type is SemibatchCryst:
        kwargs['rhos'] = [densities, densities]
        # Zero net feed energy even with the existing epsilon flow floor.
        kwargs['h_in'] = (slurry.getEnthalpy(TEMPERATURE)
                          / slurry.getDensity(total=True) * densities[0])  # [J/m**3]
    actual = unit.energy_balances(**kwargs)  # [K/s]
    assert actual == pytest.approx(heat_release / capacity, rel=RTOL, abs=0)
    if unit_type is BatchCryst:
        unit.adiabatic = False
        unit.controls = {'temp': {}}
        components = unit.energy_balances(**kwargs, heat_prof=True)  # [J/s], [J/K]
        assert components[1] == pytest.approx(capacity, rel=RTOL, abs=0)


def make_solid(thermo_path, kv=KV):
    """Build a two-bin distribution with unequal sizes for packing checks.

    Parameters
    ----------
    thermo_path : str
        Database path.
    kv : float, optional
        Positive volume shape factor [-], default half a cube's volume.

    Returns
    -------
    SolidPhase
        Uniform number distribution over 100--300 um, pure A.
    """
    return SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION, kv=kv,
                      x_distrib=[100.0, 200.0, 300.0],  # [um], two midpoint bins
                      distrib=[1e9, 1e9, 1e9],  # [#/um], resolves epsilon effect
                      temp=TEMPERATURE)


@pytest.mark.parametrize('saturation', [0.2, 1.0, None])
@pytest.mark.parametrize('spatial', [False, True])
def test_cake_enthalpy_weights_packed_phase_masses(thermo_path, saturation, spatial):
    cake = Cake()
    cake.Phases = [LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                              temp=TEMPERATURE), make_solid(thermo_path)]
    cake.saturation = (np.full(len(cake.z_external), saturation)
                       if spatial and saturation is not None else saturation)  # [-]
    # Construct inventories consistent with the specified pore filling.
    solid_volume = cake.Solid_1.vol  # [m**3]
    pore_volume = cake.cake_vol - solid_volume  # [m**3]
    liquid_volume = pore_volume * (1 if saturation is None else saturation)  # [m**3]
    cake.Liquid_1.updatePhase(vol=liquid_volume)
    phase_masses = [phase.mass for phase in cake.Phases]  # [kg]
    energy = sum(phase.mass * phase.getEnthalpy(TEMPERATURE, basis='mass')
                 for phase in cake.Phases)  # [J]
    expected = energy / sum(phase_masses)  # [J/kg liquid plus solid inventory]
    assert cake.getEnthalpy() == pytest.approx(expected, rel=RTOL, abs=0)


@pytest.mark.parametrize('inventory_fill', [0.2, 0.6])
def test_mixer_energy_balance_accepts_real_partial_cake_inlet(thermo_path, inventory_fill):
    """Close inventory energy even when producer saturation metadata differs.

    Parameters
    ----------
    thermo_path : str
        Database path.
    inventory_fill : float
        Fraction of pore volume actually filled by liquid inventory [-].
        0.2 matches the saturation metadata; 0.6 deliberately differs.

    Notes
    -----
    This directly exercises energy_balance's input mapping with real phase
    inventories and no collaborator stubs. Full mixer solves and the
    Cake/Slurry outlet decision are covered separately in
    test_mixer_container_balances.py.
    """
    cake = Cake()
    cake.Phases = [LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                              temp=TEMPERATURE), make_solid(thermo_path)]
    cake.saturation = np.full(len(cake.z_external), 0.2)  # [-], one-fifth pore fill
    pore_volume = cake.cake_vol - cake.Solid_1.vol  # [m**3]
    liquid_volume = pore_volume * inventory_fill  # [m**3], actual charged liquid
    cake.Liquid_1.updatePhase(vol=liquid_volume)
    cake_mass = sum(phase.mass for phase in cake.Phases)  # [kg]
    cake_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                      for phase in cake.Phases)  # [J]
    assert cake_mass * cake.getEnthalpy() == pytest.approx(cake_energy, rel=RTOL, abs=0)
    hot_temperature = 350.0  # [K], hotter than cake to expose weighting errors
    # Added liquid exceeds empty pore space, so the physical outlet is slurry.
    hot_liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                             temp=hot_temperature, vol=cake.cake_vol)
    mixer = Mixer()
    mixer.Inlets = [hot_liquid, cake]
    liquid_mass = hot_liquid.mass + cake.Liquid_1.mass  # [kg]
    solid_mass = cake.Solid_1.mass  # [kg]
    mixer.Outlet = Slurry()
    mixer.Outlet.Phases = [
        LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION, mass=liquid_mass),
        SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION,
                   moments=cake.Solid_1.moments, kv=KV),
    ]
    inputs = {
        'mass_liq': np.array([hot_liquid.mass, cake.Liquid_1.mass]),  # [kg]
        'mass_solid': np.array([0.0, solid_mass]),  # [kg]
        'mass_frac': np.array([hot_liquid.mass_frac, cake.Liquid_1.mass_frac]),  # [-]
        'temp': np.array([hot_temperature, TEMPERATURE]),  # [K]
        'num_distrib': np.array([np.zeros_like(cake.Solid_1.distrib),
                                cake.Solid_1.distrib]),  # [#/um]
    }
    inlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                       for phase in [hot_liquid, *cake.Phases])  # [J]
    outlet_temperature = mixer.energy_balance(inputs)  # [K]
    outlet_energy = sum(phase.mass * phase.getEnthalpy(outlet_temperature, basis='mass')
                        for phase in mixer.Outlet.Phases)  # [J]
    assert TEMPERATURE < outlet_temperature < hot_temperature
    assert outlet_energy == pytest.approx(inlet_energy, rel=RTOL, abs=0)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_liquid_props_match_selected_scalar_providers(thermo_path, basis):
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION,
                         temp=TEMPERATURE)
    expected = (liquid.getCp(basis=basis), liquid.getDensity(basis=basis),
                liquid.getEnthalpy(basis=basis))  # mass or molar provider units
    np.testing.assert_allclose(liquid.getProps(basis), expected, rtol=RTOL, atol=0)


def test_liquid_props_reject_unknown_basis(thermo_path):
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION)
    with pytest.raises(ValueError, match="basis.*mass.*mole"):
        liquid.getProps(basis='volume')


def test_cp_mix_rejects_unknown_basis(thermo_path):
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION)
    with pytest.raises(ValueError, match="basis.*mass.*mole"):
        liquid.getCpMix(TEMPERATURE, mass_frac=LIQUID_COMPOSITION, basis='volume')


@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('profile', [False, True])
def test_vector_cp_matches_stacked_scalar_evaluations(thermo_path, basis, profile):
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION)
    temperatures = np.array([290.0, 300.0, 310.0])  # [K], three rows vs five species
    compositions = np.array([np.roll(LIQUID_COMPOSITION, shift)
                             for shift in range(3)])  # [-], asymmetric rows
    fractions = compositions if profile else LIQUID_COMPOSITION  # [-]
    actual = liquid.getCp(temperatures, mass_frac=fractions, basis=basis)
    expected = [liquid.getCp(temp, mass_frac=composition, basis=basis)
                for temp, composition in zip(
                    temperatures, compositions if profile else
                    np.tile(LIQUID_COMPOSITION, (len(temperatures), 1)))]
    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize('kv', [0.0, -0.5, np.nan, np.inf, -np.inf, [0.5]])
def test_solid_phase_rejects_invalid_shape_factor(thermo_path, kv):
    with pytest.raises(ValueError, match='kv.*finite.*positive.*scalar'):
        make_solid(thermo_path, kv)


def test_porosity_matches_independent_two_bin_model(thermo_path):
    # Yu/Zou/Standish model attributed in commit 9b646f2. For two bins with
    # identical initial specific volume V, the candidates reduce to
    # V-(V-1)*g(r)*w_small and V-V*f(r)*w_large. A common diameter scale
    # cancels from r=150/250; kv remains only in epsilon-regularized weights.
    ratio = 150 / 250  # [-], ratio of midpoint diameters
    large_interaction = (1 - ratio)**2 + 0.4 * ratio * (1 - ratio)**3.7  # [-]
    small_interaction = (1 - ratio)**3.3 + 2.8 * ratio * (1 - ratio)**2.7  # [-]
    count = 2e11  # [-], uniform 1e9 #/um over 200 um
    first_moment = 4e7  # [m], count times mean size 200 um
    initial_porosity = 0.375 + 0.34 * first_moment / (count + np.finfo(float).eps)  # [-]
    specific_volume = 1 / (1 - initial_porosity)  # [-]
    unscaled_bin_volumes = np.array([3.375e-7, 1.5625e-6])  # [proportional m**3]
    shape_factors = [0.3, 0.5, np.pi / 6, 0.524, 0.8, 1.0]  # [-], compact habits
    actual_values = []  # [-]
    for kv in shape_factors:
        weights = unscaled_bin_volumes / (
            unscaled_bin_volumes.sum() + np.finfo(float).eps / kv)  # [-]
        candidates = [specific_volume - (specific_volume - 1) * large_interaction * weights[0],
                      specific_volume - specific_volume * small_interaction * weights[1]]  # [-]
        expected = 1 - 1 / max(candidates)  # [-]
        actual = make_solid(thermo_path, kv).getPorosity()  # [-]
        # Absolute float64 roundoff allowance, much smaller than the kv effect.
        assert actual == pytest.approx(expected, rel=0, abs=4*np.finfo(float).eps)
        assert 0 < actual < 1
        actual_values.append(actual)
    # The packing model is kv-invariant by construction. The only departure
    # is the +eps weight denominator: its largest relative perturbation in
    # this sweep is eps / (sum(unscaled volumes) * min(kv)).
    regularization_tolerance = (np.finfo(float).eps
                                / unscaled_bin_volumes.sum()
                                / min(shape_factors))  # [-]
    np.testing.assert_allclose(actual_values, actual_values[0],
                               rtol=regularization_tolerance, atol=0)


def test_filter_reads_phase_shape_factor(thermo_path):
    solid = make_solid(thermo_path)
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION, vol=1.0)
    filter_unit = Filter(station_diam=1.0)  # [m], existing packing default
    # Isolate the ownership assertion to the helper; other slurry/packing
    # consumers also legitimately read kv when Filter.Phases attaches phases.
    porosity = solid.getPorosity()  # [-]
    with patch.object(SolidPhase, 'kv', new_callable=PropertyMock,
                      create=True, return_value=KV) as factor:
        expected = get_alpha(solid, porosity, 1.0, solid.getDensity())  # [m/kg]
    factor.assert_called_once_with()
    filter_unit.Phases = [liquid, solid]
    assert filter_unit.alpha == pytest.approx(expected, rel=RTOL, abs=0)


def test_porosity_is_exact_at_legacy_shape_factor(thermo_path):
    # Captured on HEAD 40d2f4a for this uniform two-bin fixture. Allow four
    # float64 epsilon for a few ulp of reassociation, as the cake-alpha test does.
    legacy_porosity = 0.3679903311462239  # [-]
    assert make_solid(thermo_path, 0.524).getPorosity() == pytest.approx(
        legacy_porosity, rel=4 * np.finfo(float).eps, abs=0)


def test_saturation_matches_volume_weights_without_shape_factor():
    grid = np.array([1e-4, 2e-4, 3e-4])  # [m], same two midpoint bins
    distribution = np.ones(3)  # [common number basis/um], constant density
    porosity = 0.4  # [-], prescribed packing case
    pressure = 1e5  # [Pa], one-bar driving pressure
    height = 0.01  # [m], centimetre cake
    surface_tension = 0.07  # [N/m], water-like test fluid
    density = 1000.0  # [kg/m**3], water-like test fluid
    midpoints = np.array([1.5e-4, 2.5e-4])  # [m]
    capillary = porosity**3 * midpoints**2 * (density * 9.8 * height + pressure) / (
        (1-porosity)**2 * height * surface_tension)  # [-], existing gravity 9.8 m/s**2
    local = 0.155 * (1 + 0.031 * capillary**(-0.49))  # [-], Destro (2021), eqs 16--18
    expected = np.dot([27/152, 125/152], local)  # [-], volumes proportional to 3**3, 5**3
    actual = get_sat_inf(grid, distribution, pressure, porosity, height,
                         1.0, (surface_tension, density))  # [-], unused zeroth moment
    assert actual == pytest.approx(expected, rel=RTOL, abs=0)


def test_mass_specified_slurry_distribution_reconciles_phases(thermo_path):
    """Exercise the public setter after specifying the internal mass state.

    No public constructor input reaches this branch: Slurry has no
    mass_slurry keyword. Set the attribute directly as the stream regression
    in test_liquid_inventory_flow.py does.
    """
    grid = np.array([0.0, 100.0, 200.0, 300.0])  # [um], uniform quadrature grid
    distribution = np.array([0.0, 1e9, 1.25e8, 0.0])  # [#/m**3/um]
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION)
    solid = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION, kv=KV)
    slurry = Slurry(vol=0, x_distrib=grid, distrib=distribution)
    slurry.mass_slurry = 6.0  # [kg], arbitrary nonzero inventory
    # Direct trapezoidal integration gives normalized mu3=0.2, phi_s=0.1.
    rho_liquid = liquid.getDensity()  # [kg/m**3]
    rho_solid = solid.getDensity()  # [kg/m**3]
    volume = 6.0 / (0.9 * rho_liquid + 0.1 * rho_solid)  # [m**3]
    slurry.Phases = [solid, liquid]  # reversed attachment order
    assert slurry.vol == pytest.approx(volume, rel=RTOL, abs=0)
    assert liquid.vol == pytest.approx(0.9 * volume, rel=RTOL, abs=0)
    assert solid.vol == pytest.approx(0.1 * volume, rel=RTOL, abs=0)
    assert liquid.mass == pytest.approx(0.9 * volume * rho_liquid, rel=RTOL, abs=0)
    assert solid.mass == pytest.approx(0.1 * volume * rho_solid, rel=RTOL, abs=0)
