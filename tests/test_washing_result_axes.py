"""Washing result species routing with real phases and an analytical transport probe."""

import json

import numpy as np
import pytest
from scipy.special import erfc

from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SolidLiquidSep import DisplacementWashing

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('num_nodes', [3, 8])
@pytest.mark.parametrize('reorder_species', [False, True])
def test_washing_result_routes_species_and_node_axes(
        data_path, tmp_path, dynamic, num_nodes, reorder_species):
    """Publish each named species as a time-by-node analytical concentration.

    Parameters
    ----------
    data_path : dict
        Paths to shipped physical-property data.
    tmp_path : pathlib.Path
        Directory for the content-preserving species-order mutation.
    dynamic : bool
        Return a time series or only the washing-completion profile.
    num_nodes : int
        Cake node count, either below or above the five-species count.
    reorder_species : bool
        Permute database entries and all species-valued inputs together.
    """
    database = json.loads((data_path['flowsheet'] / 'compound_database.json').read_text())
    original_names = list(database)
    order = [4, 2, 0, 3, 1] if reorder_species else list(range(5))
    names = [original_names[index] for index in order]
    path = tmp_path / 'washing_species.json'
    path.write_text(json.dumps({name: database[name] for name in names}))
    composition = np.array([.05, .1, .15, .2, .5])[order]  # [-], asymmetric solvent-rich liquid
    liquid = LiquidPhase(str(path), mass_frac=composition, vol=1e-3)  # [m**3], litre-scale feed
    solid_composition = np.array([1., 0., 0., 0., 0.])[order]  # [-], original first species
    solid = SolidPhase(str(path), mass_frac=solid_composition,
                       x_distrib=[100., 200., 300.], distrib=[1e4, 2e4, 1e4])  # [um], [#/um], coarse-crystal probe
    diffusivity = (np.arange(1, 6) * 1e-4)[order]  # [m**2/s], distinct low-Peclet species
    liquid.diffusivity = np.tile(diffusivity[:, None], (1, 5))  # [m**2/s]
    solvent_index = names.index(original_names[-1])
    washer = DisplacementWashing(solvent_idx=solvent_index, num_nodes=num_nodes,
                                 diam_unit=.1)  # [m], laboratory filter diameter
    washer.Phases = [liquid, solid]
    initial = liquid.mass_conc.copy()  # [kg/m**3]
    pressure = 1.  # [Pa], low flow keeps all bin Peclet numbers below one
    wash_ratio = .5  # [-], half a cake-volume displacement
    height = washer.CakePhase.cake_vol / washer.cross_area  # [m]
    porosity = washer.CakePhase.porosity  # [-]
    velocity = pressure / np.mean(liquid.getViscosity()) / (
        washer.CakePhase.alpha * solid.getDensity() * height * (1 - porosity)
        + washer.resist_medium)  # [m/s], Darcy law
    assert np.all(velocity * 250e-6 / diffusivity < 1)  # [-], largest midpoint diameter [m]
    effective = diffusivity / np.sqrt(2)  # [m**2/s], low-Peclet dispersion branch
    completion = wash_ratio * height / velocity  # [s]
    times = completion * np.array([.25, .5, 1.]) if dynamic else np.array([completion])  # [s], separated observation times
    positions = np.linspace(0, height, num_nodes)  # [m]
    inlet = np.zeros(5)  # [kg/m**3]
    inlet[solvent_index] = liquid.rho_liq[solvent_index]  # [kg/m**3], pure washing solvent
    expected = np.empty((len(times), num_nodes, 5))  # [kg/m**3], time/node/species
    for row, time in enumerate(times):
        for node, position in enumerate(positions):
            for species in range(5):
                root = 2 * np.sqrt(effective[species] * time)  # [m], analytical diffusion length
                fraction = .5 * (
                    erfc((velocity * time - position) / root)
                    - np.exp(velocity * position / effective[species])
                    * erfc((velocity * time + position) / root))  # [-], Lapidus-Amundson solution
                expected[row, node, species] = (
                    fraction * (initial[species] - inlet[species]) + inlet[species])  # [kg/m**3]
    washer.solve_unit(pressure, wash_ratio=wash_ratio, dynamic=dynamic,
                      time_vals=times if dynamic else None)
    assert list(washer.result.mass_conc) == names
    np.testing.assert_allclose(washer.result.time, times, rtol=1e-12)  # [-], float64 arithmetic allowance
    np.testing.assert_allclose(washer.result.z, positions, rtol=1e-12)
    for column, name in enumerate(names):
        np.testing.assert_allclose(washer.result.mass_conc[name], expected[:, :, column],
                                   rtol=1e-12, atol=1e-12)  # [-], [kg/m**3], short analytical calculation
