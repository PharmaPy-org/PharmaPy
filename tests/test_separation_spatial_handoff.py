"""Separation-side #231 spatial fields using real phases and unit operations.

Reduced result fixtures prescribe asymmetric node/species values independently
of the ODE. Solver tests exercise real CVode and are selected by Assimulo.
Drying initialization and continuation remain deferred behind PR #210.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.MixedPhases import Cake, Slurry
from PharmaPy.SolidLiquidSep import DeliquoringStep, DisplacementWashing, Filter
from test_separation_balance_fixes import RTOL, separation_phases


@pytest.fixture(params=[False, True], ids=["original_species", "permuted_species"])
def deliquoring_result(separation_phases, request, tmp_path):
    """Build a three-cell, five-species result with known reduced states.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases with shipped thermophysical data.
    request : pytest.FixtureRequest
        Select whether to reorder the thermophysical database species.
    tmp_path : pathlib.Path
        Directory for the reordered property database.

    Returns
    -------
    tuple
        Deliquoring unit, final concentration [kg/m**3], and saturation [-].
    """
    liquid, solid = separation_phases
    properties = json.loads(Path(liquid.path_data).read_text())
    names = list(properties)
    if request.param:
        names = [names[index] for index in [4, 2, 0, 3, 1]]
    path = tmp_path / 'species.json'
    path.write_text(json.dumps({name: properties[name] for name in names}))
    # A solvent-declared phase exercises the converter's one-dimensional API.
    liquid = LiquidPhase(str(path), mass_frac=[.1, .1, .1, .1, .6],
                         vol=1e-3, name_solv=names[-1])  # [m**3], excess inlet liquid
    solid = SolidPhase(str(path), mass_frac=[1, 0, 0, 0, 0],
                       x_distrib=solid.x_distrib, distrib=solid.distrib)
    cake = Cake(num_discr=3)
    cake.Phases = [liquid, solid]
    unit = DeliquoringStep(num_nodes=3, diam_unit=0.1)  # diameter [m]
    unit.Phases = cake
    cake.z_external = unit.z_centers * unit.cake_height  # [m]
    unit.theta_conv = 2.0  # [1/s], synthetic dimensional time conversion
    unit.sat_inf = 0.2  # [-], prescribed irreducible saturation
    unit.rho_s = cake.Solid_1.getDensity()  # [kg/m**3]
    unit.rho_j = cake.Liquid_1.getDensityPure()[0]  # [kg/m**3]
    # Equal-volume five-species reference: C_ref,j = rho_j/5. The contrast
    # is 4*rho_j/5, so volume fractions 0.1, 0.2, 0.3 reduce to -1/8, 0, 1/8.
    unit.conc_mean_init = np.tile(unit.rho_j / 5, (3, 1))  # [kg/m**3]
    # Each row is a distinct, normalized liquid volume composition. Multiplying
    # by pure density gives physically consistent mass concentration per node.
    volume_fractions = np.array([
        [0.1, 0.2, 0.3, 0.1, 0.3],
        [0.3, 0.1, 0.1, 0.3, 0.2],
        [0.2, 0.3, 0.1, 0.2, 0.2],
    ])  # [-]
    reduced_saturation = np.array([0.25, 0.5, 0.75])  # [-]
    reduced_concentration = np.array([
        [-1, 0, 1, -1, 1], [1, -1, -1, 1, 0], [0, 1, -1, 0, 0],
    ]) / 8  # [-], independently derived above
    final_states = np.column_stack((reduced_saturation, reduced_concentration))  # [-]
    initial_states = np.column_stack((np.ones(3), np.zeros((3, 5))))  # [-], saturated reference
    states = np.vstack((initial_states.ravel(), final_states.ravel()))  # [-]
    unit.inlet_mass_test = liquid.mass  # [kg], pre-retrieval snapshot
    unit.inlet_fraction_test = liquid.mass_frac.copy()  # [-]
    unit.retrieve_results(np.array([0., 2.]), states)
    return unit, volume_fractions * unit.rho_j, np.array([0.4, 0.6, 0.8])


@pytest.mark.unit
@pytest.mark.parametrize('container', [list, tuple])
def test_deliquoring_phase_list_constructs_cake(separation_phases, container):
    """Construct through both supported phase containers.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    container : type
        List or tuple constructor.
    """
    unit = DeliquoringStep(num_nodes=3, diam_unit=0.1)  # diameter [m]
    unit.Phases = container(separation_phases)
    assert isinstance(unit.CakePhase, Cake)
    assert len(unit.CakePhase.z_external) == unit.num_nodes
    expected_centers = unit.cake_height * np.array([1/6, 1/2, 5/6])  # [m]
    np.testing.assert_allclose(unit.CakePhase.z_external, expected_centers, rtol=RTOL)
    np.testing.assert_allclose(unit.CakePhase.saturation, np.ones(3), rtol=RTOL)


@pytest.mark.unit
def test_deliquoring_retains_nodes_and_species_order(deliquoring_result):
    """Preserve spatial fields separately from solvent-declared bulk liquid.

    Parameters
    ----------
    deliquoring_result : tuple
        Unit, expected concentration [kg/m**3], and saturation [-].
    """
    unit, concentration, saturation = deliquoring_result
    np.testing.assert_allclose(unit.Outlet.mass_concentr, concentration, rtol=RTOL)
    np.testing.assert_allclose(unit.Outlet.saturation, saturation, rtol=RTOL)
    assert unit.Outlet.Liquid_1.name_species == unit.name_species
    assert unit.Outlet.Liquid_1.mass_frac.shape == (5,)
    # The three cells carry saturation weights 2:3:4, independently of density.
    species_weights = (2 * concentration[0] + 3 * concentration[1]
                       + 4 * concentration[2])  # [kg/m**3], proportional masses
    np.testing.assert_allclose(unit.Outlet.Liquid_1.mass_frac,
                               species_weights / species_weights.sum(), rtol=RTOL)
    for column, name in enumerate(unit.name_species):
        expected_history = np.vstack((np.full(3, unit.rho_j[column] / 5),
                                      concentration[:, column]))  # [kg/m**3]
        np.testing.assert_allclose(unit.concPerVolElement[name], expected_history, rtol=RTOL)


@pytest.mark.unit
def test_deliquoring_external_grid_is_dimensional(deliquoring_result):
    """Expose cake coordinates in metres while retaining normalized results.

    Parameters
    ----------
    deliquoring_result : tuple
        Unit and expected final fields [kg/m**3] and [-].
    """
    unit, _, _ = deliquoring_result
    # Three equal control volumes have centers L/6, L/2, 5L/6.
    expected = unit.cake_height * np.array([1/6, 1/2, 5/6])  # [m]
    np.testing.assert_allclose(unit.Outlet.z_external, expected, rtol=RTOL)
    np.testing.assert_allclose(unit.result.z, [1/6, 1/2, 5/6], rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('dynamic', [False, True])
def test_washing_saturation_matches_outlet_grid(separation_phases, dynamic):
    """Store the saturated washing state on its concentration grid.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    dynamic : bool
        Select time-dependent or final-only washing results.
    """
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    separation_phases[0].diffusivity = np.full((5, 5), 1e-4)  # [m**2/s], synthetic low-Peclet probe
    washer.Phases = list(separation_phases)
    # The analytical washing model assumes saturated pores. Its old default
    # cake has 50 saturation entries even when concentration has four nodes.
    washer.solve_unit(deltaP=1., wash_ratio=0.5, dynamic=dynamic)  # [Pa], [-]
    assert washer.Outlet.saturation.shape == washer.Outlet.z_external.shape == (4,)
    np.testing.assert_allclose(washer.Outlet.saturation, np.ones(4), rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('uniform_composition', [False, True])
def test_deliquoring_interpolates_distributed_saturation(separation_phases, uniform_composition):
    """Remap spatial fields from 50 to 10 nodes and reuse the final cake.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    uniform_composition : bool
        Select uniform or spatially varying liquid concentration.
    """
    pytest.importorskip('assimulo')
    cake = Cake(num_discr=50)
    cake.Phases = separation_phases
    unit = DeliquoringStep(num_nodes=10, diam_unit=0.1)  # diameter [m]
    unit.Phases = cake
    cake.z_external = np.linspace(0., unit.cake_height, 50)  # [m]
    # A linear profile has exact interpolated values at all receiving centers.
    cake.saturation = np.linspace(0.6, 0.9, 50)  # [-], safely above residual
    initial_concentration = cake.Liquid_1.mass_conc.copy()  # [kg/m**3]
    if not uniform_composition:
        pure_density = cake.Liquid_1.getDensityPure()[0]  # [kg/m**3]
        transfer = np.zeros(5)  # [kg/m**3], move volume from solvent to species zero
        volume_transfer = 0.05  # [-], small enough to keep all species positive
        transfer[0] = volume_transfer * pure_density[0]
        transfer[4] = -volume_transfer * pure_density[4]
        concentrations = (initial_concentration
                          + np.linspace(0., 1., 50)[:, None] * transfer)  # [kg/m**3]
        cake.mass_concentr = concentrations  # [kg/m**3], spatial field
        # Keep the attached inlet's bulk species inventory consistent with its
        # distributed field; the litre-scale excess retains this same mixture.
        concentration_weights = (np.linspace(.6, .9, 50)[:, None]
                                 * concentrations).sum(axis=0)  # [kg/m**3]
        cake.Liquid_1.updatePhase(mass_frac=concentration_weights / concentration_weights.sum())

    unit.solve_unit(deltaP=5e4, runtime=0.01, verbose=False)  # [Pa], [s], short setup probe
    expected = 0.6 + 0.3 * (np.arange(10) + 0.5) / 10  # [-], linear cell-center values
    np.testing.assert_allclose(unit.satProf[0], expected, rtol=RTOL)
    expected_concentration = np.tile(initial_concentration, (10, 1))  # [kg/m**3]
    if not uniform_composition:
        expected_concentration += ((np.arange(10) + 0.5) / 10)[:, None] * transfer
    actual_concentration = np.column_stack([
        unit.result.mass_conc[name][0] for name in unit.name_species])  # [kg/m**3]
    np.testing.assert_allclose(actual_concentration, expected_concentration, rtol=RTOL)
    previous_saturation = unit.Outlet.saturation.copy()  # [-]
    previous_concentration = unit.Outlet.mass_concentr.copy()  # [kg/m**3]
    previous_mass = unit.Outlet.Liquid_1.mass  # [kg]
    unit.solve_unit(deltaP=5e4, runtime=0.01, verbose=False)  # [Pa], [s]
    np.testing.assert_allclose(unit.satProf[0], previous_saturation, rtol=RTOL)
    np.testing.assert_allclose(unit.liquid_initial_adjustment_species, np.zeros(5),
                               rtol=0, atol=RTOL * previous_mass)
    np.testing.assert_allclose(np.column_stack([
        unit.result.mass_conc[name][0] for name in unit.name_species]),
        previous_concentration, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_filter_washing_deliquoring_handoff(separation_phases):
    """Pass a filtered, washed cake into real deliquoring integration.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    """
    pytest.importorskip('assimulo')
    filtr = Filter(station_diam=0.1)  # [m]
    separation_phases[0].diffusivity = np.full((5, 5), 1e-4)  # [m**2/s], synthetic low-Peclet probe
    filtr.Phases = list(separation_phases)
    filtr.solve_unit(deltaP=5e4, verbose=False)  # [Pa]
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    washer.Phases = filtr.Outlet
    washer.solve_unit(deltaP=1., wash_ratio=0.5, dynamic=False)  # [Pa], [-]
    unit = DeliquoringStep(num_nodes=3, diam_unit=0.1)  # [m]
    unit.Phases = washer.Outlet
    unit.solve_unit(deltaP=5e4, runtime=0.01, verbose=False)  # [Pa], [s]
    np.testing.assert_allclose(unit.satProf[0], np.ones(3), rtol=RTOL)
    assert unit.Outlet.mass_concentr.shape == (3, 5)
    np.testing.assert_allclose(unit.Outlet.mass_concentr,
                               np.column_stack([unit.result.mass_conc[name][-1]
                                                for name in unit.name_species]), rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('saturation', [0.8, np.array([0.8])])
def test_deliquoring_uniform_saturation(separation_phases, saturation):
    """Accept scalar and singleton saturation on the public solve path.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    saturation : float or ndarray
        Uniform saturation [-]; 0.8 is safely above residual saturation.
    """
    pytest.importorskip('assimulo')
    unit = DeliquoringStep(num_nodes=3, diam_unit=0.1)  # diameter [m]
    unit.Phases = list(separation_phases)
    unit.CakePhase.saturation = saturation  # [-]
    unit.solve_unit(deltaP=5e4, runtime=0.01, verbose=False)  # [Pa], [s]
    np.testing.assert_allclose(unit.satProf[0], np.full(3, 0.8), rtol=RTOL)


@pytest.mark.unit
def test_deliquoring_inventory_and_removed_liquid(deliquoring_result):
    """Close species masses using unequal cell liquid contents.

    Parameters
    ----------
    deliquoring_result : tuple
        Real unit, final concentrations [kg/m**3], and saturation [-].
    """
    unit, concentration, _ = deliquoring_result
    cake = unit.Outlet
    # Three equal cells, saturations 2/5, 3/5, 4/5: coefficients 2/15,3/15,4/15.
    species_mass = cake.porosity * cake.cake_vol * (
        2 * concentration[0] + 3 * concentration[1] + 4 * concentration[2]) / 15  # [kg]
    expected_mass = species_mass.sum()  # [kg]
    assert cake.Liquid_1.mass == pytest.approx(expected_mass, rel=RTOL)
    initial_pore_mass = cake.porosity * cake.cake_vol * unit.rho_j.sum() / 5  # [kg]
    assert unit.liquid_removed == pytest.approx(initial_pore_mass - expected_mass, rel=RTOL)
    assert unit.liquid_initial_adjustment == pytest.approx(unit.inlet_mass_test - initial_pore_mass, rel=RTOL)
    np.testing.assert_allclose(
        cake.Liquid_1.mass * cake.Liquid_1.mass_frac
        + unit.liquid_removed * unit.liquid_removed_mass_frac
        + unit.liquid_initial_adjustment_species,
        unit.inlet_mass_test * unit.inlet_fraction_test, rtol=RTOL)
    initial_pore_mass = cake.porosity * cake.cake_vol * unit.rho_j.sum() / 5  # [kg]
    expected_removed = initial_pore_mass - np.array([initial_pore_mass, expected_mass])  # [kg]
    np.testing.assert_allclose(unit.result.mass_liquid_removed, expected_removed, rtol=RTOL)
    assert np.ndim(cake.Liquid_1.vol) == np.ndim(cake.Liquid_1.moles) == 0
    np.testing.assert_allclose(unit.result.liquid_initial_adjustment_species,
                               unit.liquid_initial_adjustment_species, rtol=RTOL)
    assert unit.result.liquid_initial_adjustment == pytest.approx(
        unit.liquid_initial_adjustment_species.sum(), rel=RTOL)
    assert not hasattr(unit, 'liquid_initial_adjustment_mass_frac')
    assert not hasattr(unit.result, 'liquid_initial_adjustment_mass_frac')


@pytest.mark.unit
@pytest.mark.parametrize('fill', [0.8, 1.2])
def test_deliquored_cake_mixer_and_enthalpy(deliquoring_result, fill):
    """Use the reconciled bulk inventory in cake energy and Mixer output type.

    Parameters
    ----------
    deliquoring_result : tuple
        Real unit and prescribed final fields [kg/m**3], [-].
    fill : float
        Mixed liquid volume divided by pore volume [-]; straddles saturation.
    """
    unit, _, _ = deliquoring_result
    cake = unit.Outlet
    liquid = cake.Liquid_1
    temperature = 320.0  # [K], above the enthalpy reference for nonzero energy
    expected_enthalpy = (
        liquid.mass * liquid.getEnthalpy(temp=temperature, basis='mass')
        + cake.Solid_1.mass * cake.Solid_1.getEnthalpy(temp=temperature)
    ) / (liquid.mass + cake.Solid_1.mass)  # [J/kg cake]
    assert np.ndim(cake.getEnthalpy(temp=temperature)) == 0
    assert cake.getEnthalpy(temp=temperature) == pytest.approx(expected_enthalpy, rel=RTOL)
    pore_volume = cake.porosity * cake.cake_vol  # [m**3]
    added = LiquidPhase(liquid.path_data, mass_frac=liquid.mass_frac,
                        vol=fill * pore_volume - liquid.vol)  # [m**3]
    total_liquid = added.mass + liquid.mass  # [kg]
    mixer = Mixer()
    # Liquid first follows the existing Mixer entry contract for name metadata.
    mixer.Inlets = [added, cake]
    mixer.solve_unit()
    assert isinstance(mixer.Outlet, Cake if fill < 1 else Slurry)
    assert mixer.Outlet.Liquid_1.mass == pytest.approx(total_liquid, rel=RTOL)
    np.testing.assert_allclose(mixer.Outlet.Liquid_1.mass_frac, liquid.mass_frac, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('case', ['same_grid', 'finer_grid', 'outside_span', 'clip'])
def test_deliquoring_bounded_remap(separation_phases, case):
    """Hold end states and avoid spline overshoot on concentration and saturation.

    Parameters
    ----------
    separation_phases : tuple
        Real phases with shipped thermophysical data.
    case : str
        Same-grid identity, finer-grid step, diameter mismatch, or clipping.
    """
    source = DeliquoringStep(num_nodes=4, diam_unit=0.1)  # [m]
    source.Phases = list(separation_phases)
    cake = source.CakePhase
    cake.saturation = np.array([.4, .4, 1., 1.])  # [-], bounded step for overshoot probe
    profile_scale = np.array([1., 1., 2., 2.])  # [-], independent positive concentration step
    cake.mass_concentr = profile_scale[:, None] * cake.Liquid_1.mass_conc  # [kg/m**3]
    unit = DeliquoringStep(num_nodes=4 if case == 'same_grid' else 8,
                           diam_unit=0.01 if case == 'outside_span' else 0.1)  # [m]
    unit.Phases = cake
    unit.sat_inf = .2  # [-], prescribed residual saturation for remapping only
    if case == 'clip':
        cake.saturation = np.array([.1, .1, 1.1, 1.1])  # [-], outside physical reduced-state interval
    if case == 'same_grid':
        states = unit.initialize_states().reshape(unit.num_nodes, -1)  # [-]
        expected_saturation = np.array([.4, .4, 1., 1.])  # [-]
        expected_scale = profile_scale  # [-]
    else:
        if case == 'outside_span':
            with pytest.warns(UserWarning, match='span.*holding'):
                states = unit.initialize_states().reshape(unit.num_nodes, -1)  # [-]
        elif case == 'clip':
            with pytest.warns(UserWarning, match='residual floor'):
                states = unit.initialize_states().reshape(unit.num_nodes, -1)  # [-]
        else:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                states = unit.initialize_states().reshape(unit.num_nodes, -1)  # [-]
            assert not recorded
        if case == 'outside_span':
            expected_saturation = np.ones(8)  # [-], all new nodes beyond the source outlet
            expected_scale = np.full(8, 2.)  # [-]
        else:
            expected_saturation = np.array([.4, .4, .4, .55, .85, 1., 1., 1.])  # [-], linear step
            expected_scale = np.array([1., 1., 1., 1.25, 1.75, 2., 2., 2.])  # [-]
            if case == 'clip':
                expected_saturation = np.array([.2, .2, .2, .35, .85, 1., 1., 1.])  # [-]
    np.testing.assert_allclose(states[:, 0] * .8 + .2, expected_saturation, rtol=RTOL)
    concentration = (states[:, 1:] * (unit.rho_j - unit.conc_mean_init)
                     + unit.conc_mean_init)  # [kg/m**3], decode reduced state
    np.testing.assert_allclose(concentration,
        expected_scale[:, None] * cake.Liquid_1.mass_conc, rtol=RTOL)
    np.testing.assert_allclose(unit.conc_mean_init,
        np.tile(np.mean(expected_scale) * cake.Liquid_1.mass_conc, (unit.num_nodes, 1)), rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_deliquoring_saturated_mean_and_mass_balance(separation_phases):
    """A saturated twenty-cell cake has unit mean and closes attached inventory.

    Parameters
    ----------
    separation_phases : tuple
        Real phases; the excess liquid is included in the initial adjustment.
    """
    pytest.importorskip('assimulo')
    unit = DeliquoringStep(num_nodes=20, diam_unit=.1)  # [m]
    unit.Phases = list(separation_phases)
    initial_mass = unit.Liquid_1.mass  # [kg]
    concentration = unit.Liquid_1.mass_conc.copy()  # [kg/m**3]
    unit.solve_unit(deltaP=5e4, runtime=.01, verbose=False)  # [Pa], [s], measurable drainage
    assert unit.mean_sat[0] == pytest.approx(1., rel=RTOL)
    np.testing.assert_allclose(unit.conc_mean_init, np.tile(concentration, (20, 1)), rtol=RTOL)
    assert initial_mass == pytest.approx(unit.Outlet.Liquid_1.mass + unit.liquid_removed + unit.liquid_initial_adjustment, rel=RTOL)
    assert unit.result.mass_liquid_removed[0] == 0
    assert unit.liquid_initial_adjustment > 0  # initial excess above the pores


@pytest.mark.unit
@pytest.mark.parametrize('saturation', [0.4, 1 - 5e-10])
def test_washing_resaturation_warning(separation_phases, saturation):
    """Warn for partial filling while tolerating saturation roundoff.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    saturation : float
        Inlet pore filling [-]; the near-one case is within the 1e-9 tolerance.
    """
    separation_phases[0].diffusivity = np.full((5, 5), 1e-4)  # [m**2/s], low-Peclet probe
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=.1)  # [m]
    washer.Phases = list(separation_phases)
    height = washer.CakePhase.cake_vol / washer.cross_area  # [m]
    np.testing.assert_allclose(washer.CakePhase.z_external,
                               np.linspace(0., height, 4), rtol=RTOL)
    washer.CakePhase.saturation = np.full(4, saturation)  # [-]
    if saturation < .5:  # [-], distinguish the deliberately partial filling case
        with pytest.warns(UserWarning, match='assumes saturated pores; re-saturating'):
            washer.solve_unit(deltaP=1., wash_ratio=.5, dynamic=False)  # [Pa], [-]
    else:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            washer.solve_unit(deltaP=1., wash_ratio=.5, dynamic=False)  # [Pa], [-]
        assert not [warning for warning in recorded if 're-saturating' in str(warning.message)]
    np.testing.assert_allclose(washer.Outlet.saturation, np.ones(4), rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('distance', [0.5e-9, 2e-9])
def test_deliquoring_span_warning_tolerance(separation_phases, distance):
    """Distinguish coordinate roundoff from a material span mismatch.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    distance : float
        Receiving node overrun divided by source height [-]; straddles 1e-9.
    """
    unit = DeliquoringStep(num_nodes=3, diam_unit=.1)  # [m]
    unit.Phases = list(separation_phases)
    unit.sat_inf = .2  # [-], prescribed residual saturation
    unit.CakePhase.saturation = np.array([.4, .6, .8])  # [-], nonconstant field
    source_height = unit.cake_height  # [m]
    unit.z_centers[-1] = 1 + distance  # [-], receiving node beyond the source domain
    assert unit.CakePhase.cake_height == source_height
    if distance > 1e-9:  # [-], documented span tolerance
        with pytest.warns(UserWarning, match='span.*holding'):
            unit.initialize_states()
    else:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            unit.initialize_states()
        assert not recorded


@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['negative_concentration', 'nan_concentration', 'grid_order'])
def test_deliquoring_rejects_invalid_remap_fields(separation_phases, invalid):
    """Reject invalid fields before interpolation can make them plausible.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    invalid : str
        Select a physical concentration error or unordered coordinates.
    """
    unit = DeliquoringStep(num_nodes=3, diam_unit=.1)  # [m]
    unit.Phases = list(separation_phases)
    unit.sat_inf = .2  # [-]
    unit.CakePhase.mass_concentr = np.tile(unit.Liquid_1.mass_conc, (3, 1))  # [kg/m**3]
    match = 'concentration'
    if invalid == 'negative_concentration':
        unit.CakePhase.mass_concentr[0, 0] = -1.  # [kg/m**3], material violation
    elif invalid == 'nan_concentration':
        unit.CakePhase.mass_concentr[0, 0] = np.nan  # [kg/m**3]
    else:
        unit.CakePhase.z_external = unit.CakePhase.z_external[::-1]  # [m]
        match = 'increasing coordinates'
    with pytest.raises(ValueError, match=match):
        unit.initialize_states()


@pytest.mark.unit
def test_deliquoring_records_attached_inventory_mismatch(deliquoring_result):
    """Account for a smaller attached inventory as a signed initial adjustment.

    Parameters
    ----------
    deliquoring_result : tuple
        Unit and known final fields [kg/m**3] and [-].
    """
    unit, _, _ = deliquoring_result
    unit.Outlet.Liquid_1.updatePhase(mass=unit.Outlet.Liquid_1.mass / 2)  # [kg]
    before = unit.Outlet.Liquid_1.mass * unit.Outlet.Liquid_1.mass_frac  # [kg]
    unit.retrieve_results(np.array([0., 2.]), unit.outputs)  # [-], reduced times
    assert unit.liquid_initial_adjustment < 0
    np.testing.assert_allclose(
        unit.Outlet.Liquid_1.mass * unit.Outlet.Liquid_1.mass_frac
        + unit.liquid_removed * unit.liquid_removed_mass_frac
        + unit.liquid_initial_adjustment_species, before, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_filter_cake_grid_is_dimensional(separation_phases):
    """Filter produces fifty dimensional cell centers over the recovered cake.

    Parameters
    ----------
    separation_phases : tuple
        Real phases with a coarse crystal population.
    """
    pytest.importorskip('assimulo')
    unit = Filter(station_diam=.1)  # [m]
    unit.Phases = list(separation_phases)
    unit.solve_unit(deltaP=5e4, verbose=False)  # [Pa]
    height = unit.Outlet.cake_vol / unit.area_filt  # [m]
    expected_centers = (np.arange(50) + .5) / 50 * height  # [m], Cake default grid count
    np.testing.assert_allclose(unit.Outlet.z_external, expected_centers, rtol=RTOL)


@pytest.mark.unit
def test_deliquoring_pure_component_has_finite_reduced_state(separation_phases):
    """A pure, uniform solvent has zero contrast and a finite zero reduced value.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases; the liquid is changed to pure solvent.
    """
    separation_phases[0].updatePhase(mass_frac=[0, 0, 0, 0, 1])  # [-], pure solvent
    unit = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    unit.Phases = list(separation_phases)
    unit.sat_inf = .2  # [-], fixed residual saturation for the initialization probe
    with np.errstate(all='raise'):
        states = unit.initialize_states().reshape(4, 6)  # [-], saturation + five species
    np.testing.assert_allclose(states[:, 1:], np.zeros((4, 5)), rtol=RTOL)
    np.testing.assert_allclose(states[:, 0], np.ones(4), rtol=RTOL)


@pytest.mark.unit
def test_deliquoring_rejects_empty_retained_inventory(deliquoring_result):
    """Do not pass a zero mass to updatePhase, where zero means unspecified.

    Parameters
    ----------
    deliquoring_result : tuple
        Unit and known final fields [kg/m**3] and [-].
    """
    unit, _, _ = deliquoring_result
    states = unit.outputs.copy()  # [-]
    states[:, ::6] = -.25  # [-], S=0 maps to (0 - 0.2)/(1 - 0.2)
    before = unit.Outlet.Liquid_1.mass  # [kg]
    with pytest.raises(ValueError, match='positive retained liquid'):
        unit.retrieve_results(np.array([0., 2.]), states)  # [-]
    assert unit.Outlet.Liquid_1.mass == before


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('runtime', [1e-4, 1e-2])
def test_regrid_accounts_initial_adjustment(separation_phases, runtime):
    """Regridding a physical four-cell cake to five cells preserves bookkeeping.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases.
    runtime : float
        Short physical drainage duration [s], exposing initial remap offsets.
    """
    pytest.importorskip('assimulo')
    source = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    source.Phases = list(separation_phases)
    cake = source.CakePhase
    cake.saturation = np.array([.4, .4, .6, 1.])  # [-], asymmetric pore filling
    cake.Liquid_1.updatePhase(vol=.6 * cake.porosity * cake.cake_vol)  # [m**3], four-cell mean
    initial_species = cake.Liquid_1.mass * cake.Liquid_1.mass_frac  # [kg]
    unit = DeliquoringStep(num_nodes=5, diam_unit=.1)  # [m]
    unit.Phases = cake
    unit.solve_unit(deltaP=5e4, runtime=runtime, verbose=False)  # [Pa], [s]
    # Linear samples at 0.1,0.3,0.5,0.7,0.9 give this independent profile;
    # its mean is 0.604, while the original four-cell mean is 0.6.
    np.testing.assert_allclose(unit.satProf[0], [.4, .4, .5, .72, 1.], rtol=RTOL)
    assert unit.liquid_initial_adjustment < 0
    assert np.all(unit.result.mass_liquid_removed >= 0)
    np.testing.assert_allclose(
        unit.Outlet.Liquid_1.mass * unit.Outlet.Liquid_1.mass_frac
        + unit.liquid_removed * unit.liquid_removed_mass_frac
        + unit.liquid_initial_adjustment_species, initial_species, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_below_residual_saturation_solves(separation_phases):
    """A below-residual inlet starts above the singular capillary state.

    Parameters
    ----------
    separation_phases : tuple
        Real phases; S=0.01 is deliberately below the residual correlation.
    """
    pytest.importorskip('assimulo')
    unit = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    unit.Phases = list(separation_phases)
    unit.CakePhase.saturation = np.full(4, .01)  # [-]
    with np.errstate(divide='raise', invalid='raise', over='raise'):
        with pytest.warns(UserWarning, match='residual floor'):
            theta, states = unit.solve_unit(deltaP=5e4, runtime=.01, verbose=False)  # [Pa], [s]
        derivative = unit.unit_model(theta[0], states[0])  # [-], reduced time derivative
        zero_states = states[0].copy()  # [-], exercise the RHS zero-state guard itself
        zero_states[::6] = 0  # [-], zero reduced saturation in each five-species node
        zero_derivative = unit.unit_model(theta[0], zero_states)  # [-]
    assert np.all(np.isfinite(derivative))
    assert np.all(np.isfinite(zero_derivative))
    assert np.all(unit.satProf > unit.sat_inf)
    assert np.all(unit.satProf <= 1)


@pytest.mark.unit
def test_washing_consumes_midpoint_field(deliquoring_result):
    """Recover the incoming midpoint concentration from the washing solution.

    Parameters
    ----------
    deliquoring_result : tuple
        Three-cell cake with a distinct middle-node concentration [kg/m**3].
    """
    source, expected, _ = deliquoring_result
    cake = source.Outlet
    cake.Liquid_1.diffusivity = np.full((5, 5), 1e-4)  # [m**2/s], low-Peclet probe
    washer = DisplacementWashing(solvent_idx=4, num_nodes=3, diam_unit=.1)  # [m]
    washer.Phases = cake
    with pytest.warns(UserWarning, match='re-saturating'):
        concentration, normalized, _, _ = washer.solve_unit(
            deltaP=1., wash_ratio=.5, dynamic=False)  # [Pa], [-]
    # Species zero is not the washing solvent: C_out = C_star * C_initial.
    assert normalized[1, 0] > 0
    assert concentration[1, 0] / normalized[1, 0] == pytest.approx(expected[1, 0], rel=RTOL)


@pytest.mark.unit
def test_published_fields_are_independent(deliquoring_result):
    """Editing one diagnostic must not overwrite other results or the outlet.

    Parameters
    ----------
    deliquoring_result : tuple
        Real unit with known concentration and saturation fields.
    """
    unit, expected, saturation = deliquoring_result
    name = unit.name_species[0]
    unit.concPerVolElement[name][-1, 0] = -1.  # [kg/m**3], deliberate client mutation
    assert unit.concPerSpecies[name][-1, 0] == pytest.approx(expected[0, 0], rel=RTOL)
    unit.concPerSpecies[name][-1, 1] = -2.  # [kg/m**3]
    assert unit.result.mass_conc[name][-1, 1] == pytest.approx(expected[1, 0], rel=RTOL)
    unit.result.saturation[-1, 0] = -1.  # [-]
    np.testing.assert_allclose(unit.Outlet.saturation, saturation, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('solvent', [0, 4])
def test_washing_pore_inventory_on_identical_grid(separation_phases, solvent):
    """A washed cake carries its own species inventory into an identical grid.

    Parameters
    ----------
    separation_phases : tuple
        Real Filter feed with shipped liquid and solid properties.
    solvent : int
        Washing solvent index; both opposite ends of the species order.
    """
    pytest.importorskip('assimulo')
    liquid, solid = separation_phases
    liquid.diffusivity = np.full((5, 5), 1e-4)  # [m**2/s], resolved washing gradient
    filt = Filter(station_diam=.1)  # [m]
    filt.Phases = [liquid, solid]
    filt.solve_unit(deltaP=5e4, verbose=False)  # [Pa]
    washer = DisplacementWashing(solvent_idx=solvent, num_nodes=4, diam_unit=.1)  # [m]
    washer.Phases = filt.Outlet
    washer.solve_unit(deltaP=1., wash_ratio=.5, dynamic=False)  # [Pa], [-]
    cake = washer.Outlet
    field = cake.mass_concentr.copy()  # [kg/m**3]
    # Four endpoint nodes represent widths L/6, L/3, L/3, L/6.
    expected_species = cake.porosity * cake.cake_vol * (
        field[0] + 2 * field[1] + 2 * field[2] + field[3]) / 6  # [kg]
    np.testing.assert_allclose(cake.Liquid_1.mass * cake.Liquid_1.mass_frac,
                               expected_species, rtol=RTOL, atol=RTOL * cake.Liquid_1.mass)
    assert cake.Liquid_1.mass_frac.shape == (5,)
    unit = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    unit.Phases = cake
    # This grid override is a bookkeeping fixture matching washing positions
    # and cell widths. Deliquoring transport assumes uniform cells; this test
    # checks inventory accounting, not transport accuracy on this grid.
    unit.z_centers = washer.zProf / unit.cake_height  # [-]
    unit.z_grid = np.array([0., 1/6, 1/2, 5/6, 1.])  # [-], faces for the washing grid
    unit.delta_z = np.diff(unit.z_grid)  # [-]
    unit.solve_unit(deltaP=5e4, runtime=1e-4, verbose=False)  # [Pa], [s]
    np.testing.assert_allclose(unit.liquid_initial_adjustment_species, np.zeros(5),
                               atol=RTOL * expected_species.sum(), rtol=0)
    assert np.all(unit.result.mass_liquid_removed >= 0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_filter_excess_is_adjustment_not_removed_liquid(separation_phases):
    """Separate Filter liquid above the cake from the pore drainage history.

    Parameters
    ----------
    separation_phases : tuple
        Real feed with enough liquid for partial filtration.
    """
    pytest.importorskip('assimulo')
    filt = Filter(station_diam=.1)  # [m]
    filt.Phases = list(separation_phases)
    filt.solve_unit(deltaP=5e4, time_grid=[0., .01], verbose=False)  # [Pa], [s], partial filtration
    cake = filt.Outlet
    attached = cake.Liquid_1.mass  # [kg]
    pore_mass = cake.porosity * cake.cake_vol * cake.Liquid_1.mass_conc.sum()  # [kg]
    assert attached > pore_mass
    unit = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    unit.Phases = cake
    unit.solve_unit(deltaP=5e4, runtime=1e-4, verbose=False)  # [Pa], [s]
    assert unit.liquid_initial_adjustment == pytest.approx(attached - pore_mass, rel=RTOL)
    assert unit.result.mass_liquid_removed[0] == 0
    assert unit.liquid_removed == pytest.approx(pore_mass - unit.Outlet.Liquid_1.mass, rel=RTOL)


@pytest.mark.unit
def test_constant_field_geometry_mismatch_does_not_warn(separation_phases):
    """Constant fields need no extrapolation warning when the diameter changes.

    Parameters
    ----------
    separation_phases : tuple
        Real phases with uniform saturation and liquid composition.
    """
    source = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    source.Phases = list(separation_phases)
    unit = DeliquoringStep(num_nodes=5, diam_unit=.01)  # [m], source-domain mismatch
    unit.Phases = source.CakePhase
    unit.sat_inf = .2  # [-], prescribed residual saturation
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        unit.initialize_states()
    assert not recorded


@pytest.mark.unit
def test_residual_floor_roundoff_does_not_warn(separation_phases):
    """Sub-tolerance saturation clipping needs no modeling-assumption warning.

    Parameters
    ----------
    separation_phases : tuple
        Real phases with uniform composition and saturation.
    """
    unit = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    unit.Phases = list(separation_phases)
    unit.sat_inf = .2  # [-], prescribed residual saturation
    roundoff_offset = 5e-10  # [-], half the documented 1e-9 warning tolerance
    unit.CakePhase.saturation = np.full(4, unit.sat_inf - roundoff_offset)  # [-]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        states = unit.initialize_states().reshape(unit.num_nodes, -1)  # [-]
    assert not recorded
    assert np.all(states[:, 0] > 0)


@pytest.mark.unit
def test_washing_rejects_zero_pore_inventory(separation_phases):
    """Reject an empty retained liquid field before dividing by its mass.

    Parameters
    ----------
    separation_phases : tuple
        Real phases with a positive attached liquid inventory.
    """
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=.1)  # [m]
    washer.Phases = list(separation_phases)
    washer.num_z = 4
    washer.num_t = 1
    concentration = np.zeros((4, 1, washer.Liquid_1.num_species))  # [kg/m**3]
    initial_mass = washer.Liquid_1.mass  # [kg]
    initial_fractions = washer.Liquid_1.mass_frac.copy()  # [-]
    with np.errstate(divide='raise', invalid='raise'):
        with pytest.raises(ValueError, match='positive retained liquid inventory'):
            washer.retrieve_results(washer.CakePhase.z_external,
                                    np.array([0.]), concentration)  # [m], [s], [kg/m**3]
    assert washer.Liquid_1.mass == pytest.approx(initial_mass, rel=RTOL)
    np.testing.assert_allclose(washer.Liquid_1.mass_frac, initial_fractions, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('dynamic', [False, True])
def test_washing_bulk_concentrations_close_species_inventory(separation_phases, uniform, dynamic):
    """Close species balances using actual inlet and outlet liquid inventories.

    Parameters
    ----------
    separation_phases : tuple
        Real five-species phases from the shipped database.
    uniform : bool
        Use a uniform initial field to check the previous analytical formula.
    dynamic : bool
        Select time-series or final analytical profile evaluation.

    Notes
    -----
    A 1 cm unit and one tenth of the fixture population give a cake about
    6 cm high. This makes unit-diameter packing differ measurably from the
    Cake-owned porosity, which must set both geometry and liquid inventories.
    The initial attached liquid exactly fills the pores; no remap or
    re-saturation adjustment is needed on this unchanged four-node grid.
    """
    liquid, solid = separation_phases
    liquid.diffusivity = np.full((5, 5), 1e-4)  # [m**2/s], resolved review-case dispersion
    population_scale = .1  # [-], gives a 6 cm cake in the 1 cm diameter unit
    solid.updatePhase(distrib=solid.distrib * population_scale)
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=.01)  # [m]
    washer.Phases = [liquid, solid]
    # Asymmetric normalized volume fractions give a physically consistent
    # nonuniform field with four positions and five species.
    fractions = np.array([
        [.1, .2, .3, .1, .3], [.3, .1, .1, .3, .2],
        [.2, .3, .1, .2, .2], [.4, .1, .2, .1, .2],
    ])  # [-]
    if uniform:
        fractions[:] = fractions[0]
    initial = fractions * liquid.getDensityPure()[0]  # [kg/m**3]
    washer.CakePhase.mass_concentr = initial.copy()  # [kg/m**3]
    inlet = np.zeros(liquid.num_species)  # [kg/m**3], pure washing solvent
    inlet[washer.solvent_idx] = liquid.rho_liq[washer.solvent_idx]
    cake_volume = washer.CakePhase.cake_vol  # [m**3]
    porosity = washer.CakePhase.porosity  # [-], also owns cake volume and resistance
    pore_volume = porosity * cake_volume  # [m**3], saturated initial and final pores
    initial_mean = (initial[0] + 2 * initial[1]
                    + 2 * initial[2] + initial[3]) / 6  # [kg/m**3]
    initial_species_mass = pore_volume * initial_mean  # [kg]
    liquid.updatePhase(mass=initial_species_mass.sum(),
                       mass_frac=initial_species_mass / initial_species_mass.sum())
    attached_initial_species = liquid.mass * liquid.mass_frac  # [kg]
    wash_ratio = .5  # [-], half a cake volume of wash liquid (superficial basis)
    concentration, normalized, retained, effluent = washer.solve_unit(
        deltaP=1e5, wash_ratio=wash_ratio, dynamic=dynamic)  # [Pa], review pressure drop
    assert retained.shape == effluent.shape == (liquid.num_species,)
    # Four equally spaced endpoint nodes have trapezoidal weights 1:2:2:1.
    expected_mean = (concentration[0] + 2 * concentration[1]
                     + 2 * concentration[2] + concentration[3]) / 6  # [kg/m**3]
    np.testing.assert_allclose(retained, expected_mean, rtol=RTOL)
    np.testing.assert_allclose(washer.concProf[:, -1], concentration, rtol=RTOL)
    outlet_liquid = washer.Outlet.Liquid_1
    attached_final_species = outlet_liquid.mass * outlet_liquid.mass_frac  # [kg]
    wash_volume = wash_ratio * cake_volume  # [m**3], also effluent volume at saturation one
    np.testing.assert_allclose(
        attached_initial_species + wash_volume * inlet,
        attached_final_species + wash_volume * effluent, rtol=RTOL)
    if uniform:
        mean_normalized = (normalized[0] + 2 * normalized[1]
                           + 2 * normalized[2] + normalized[3]) / 6  # [-]
        previous_retained = (initial[0] - inlet) * mean_normalized + inlet  # [kg/m**3]
        previous_effluent = porosity / wash_ratio * (initial[0] - previous_retained) + inlet  # [kg/m**3]
        np.testing.assert_allclose(retained, previous_retained, rtol=RTOL)
        np.testing.assert_allclose(effluent, previous_effluent, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_nonconservative_deliquoring_surfaces_invalid_removal(data_path):
    """Expose the unresolved species-advection defect in #29 during accounting.

    The synthetic 1 Pa wash creates a nonuniform field that makes the existing
    deliquoring equation increase a species inventory. Once
    https://github.com/PharmaPy-org/PharmaPy/issues/29 is repaired, replace this
    provisional invalid-diagnostic expectation with valid, nonnegative removal.
    """
    pytest.importorskip('assimulo')
    from PharmaPy.Phases import LiquidPhase, SolidPhase
    path = str(data_path['flowsheet'] / 'compound_database.json')
    liquid = LiquidPhase(path, mass_frac=[.1, .1, .1, .1, .6], vol=1e-3)  # [-], [m**3]
    liquid.diffusivity = np.full((5, 5), 1e-9)  # [m**2/s], synthetic diffusivity
    solid = SolidPhase(path, mass_frac=[1, 0, 0, 0, 0],
                       x_distrib=[100, 200, 300], distrib=[1e4, 2e4, 1e4])
    # Grid [um], population [#/um], chosen to retain a heterogeneous wash field.
    washer = DisplacementWashing(solvent_idx=0, num_nodes=4, diam_unit=.1)  # [m]
    washer.Phases = [liquid, solid]
    washer.solve_unit(deltaP=1., wash_ratio=.1, dynamic=False)  # [Pa], [-]
    unit = DeliquoringStep(num_nodes=4, diam_unit=.1)  # [m]
    unit.Phases = washer.Outlet
    with pytest.warns(RuntimeWarning, match='nonconservative.*#29'):
        unit.solve_unit(deltaP=5e4, runtime=.03, verbose=False)  # [Pa], [s]
    assert not unit.removal_diagnostics_valid
    assert np.any(unit.liquid_removed_species < 0)
    assert np.isnan(unit.liquid_removed_mass_frac).all()
    assert unit.liquid_removed == pytest.approx(unit.liquid_removed_species.sum(), rel=1e-12)
    assert unit.result.mass_liquid_removed[-1] == pytest.approx(unit.liquid_removed, rel=1e-12)
    assert unit.result.mass_liquid_removed[0] == 0
