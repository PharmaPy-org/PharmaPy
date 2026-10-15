"""Cake pore liquid handed back on its own basis (issue #463).

``DeliquoringStep`` and ``DisplacementWashing`` integrate the cake's pore
liquid as mass concentrations [kg/m**3] on their spatial grid. A liquid whose
concentrations are authoritative (``LiquidPhase.concentration_basis``, #439)
must leave with the pores' liquid volume, ``porosity * cake_vol *
sum_i(width_i * S_i)`` [m**3], and the pores' mean concentrations; a liquid
given by fractions keeps the fraction hand-back on the ideal-mixing density.

Fixture: the shipped five-species database
(``tests/Flowsheet/data/compound_database.json``), 1 L of liquid of mass
fractions (0.1, 0.1, 0.1, 0.1, 0.6) given either by those fractions or by
0.8 times their ideal-mixing mass concentrations (a non-closing liquid), and
crystals of A on a 100, 200, 300 um grid with populations (1, 2, 1)e4 #/um,
as in the issue reproduction. Expected volumes and concentrations are
computed from the outlet cake's porosity, volume, saturation and the units'
cell widths (the deliquoring grid ``delta_z``; washing cells with faces
halfway between nodes and half cells at both ends), not from the liquid's
own conversions. Displacement washing runs its static solution without a
solver; deliquoring integrates with CVode and is marked ``assimulo``.

Refs: https://github.com/PharmaPy-org/PharmaPy/issues/463,
https://github.com/PharmaPy-org/PharmaPy/issues/439
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.MixedPhases import Cake
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SolidLiquidSep import DeliquoringStep, DisplacementWashing

DATABASE = str(Path(__file__).parent / 'Flowsheet/data/compound_database.json')
with open(DATABASE) as handle:
    _PROPERTIES = json.load(handle)
RHO_LIQ = np.array([value['rho_liq'] for value in _PROPERTIES.values()])  # [kg/m**3]

MASS_FRAC = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], A, B, C, D, solvent
IDEAL_DENSITY = 1 / np.sum(MASS_FRAC / RHO_LIQ)  # [kg/m**3], ideal mixing
CLOSURE = 0.8  # [-], concentrations at 0.8 of the ideal-mixing values
LIQUID_VOLUME = 1e-3  # [m**3]
GRID = np.array([100.0, 200.0, 300.0])  # [um]
POPULATION = np.array([1e4, 2e4, 1e4])  # [#/um]
DIFFUSIVITY = 1e-4  # [m**2/s], synthetic constant binary diffusivities
SOLVENT_INDEX = 4  # [-], wash solvent
UNIT_DIAMETER = 0.1  # [m]
WASH_PRESSURE_DROP = 1.0  # [Pa]
WASH_RATIO = 0.5  # [-]
DELIQUORING_PRESSURE_DROP = 5e4  # [Pa]
DELIQUORING_TIME = 5.0  # [s]
RTOL = 1e-12  # [-], float64 round-off of short sums and quotients


def _phases(liquid_basis):
    """Build the liquid and crystals of the cake.

    Parameters
    ----------
    liquid_basis : {'concentrations', 'fractions'}
        Give the liquid by 0.8 times its ideal-mixing mass concentrations
        [kg/m**3] (non-closing) or by its mass fractions [-].

    Returns
    -------
    list
        ``LiquidPhase`` of LIQUID_VOLUME [m**3] and ``SolidPhase`` of
        crystals of A.
    """
    if liquid_basis == 'concentrations':
        mass_conc = CLOSURE * MASS_FRAC * IDEAL_DENSITY  # [kg/m**3]
        liquid = LiquidPhase(DATABASE, mass_conc=mass_conc, vol=LIQUID_VOLUME)
    else:
        liquid = LiquidPhase(DATABASE, mass_frac=MASS_FRAC, vol=LIQUID_VOLUME)
    num_species = len(MASS_FRAC)
    liquid.diffusivity = np.full((num_species, num_species), DIFFUSIVITY)
    solid = SolidPhase(DATABASE, mass_frac=[1, 0, 0, 0, 0],
                       x_distrib=GRID.copy(), distrib=POPULATION.copy())
    return [liquid, solid]


def _assert_pore_liquid(outlet, widths):
    """Check the outlet liquid against the outlet cake's pore liquid.

    Parameters
    ----------
    outlet : Cake
        Outlet cake with ``porosity`` [-], ``cake_vol`` [m**3], per-node
        ``saturation`` [-] and ``mass_concentr`` [kg/m**3].
    widths : numpy.ndarray
        Cell widths over cake height [-], shape (num_nodes,).
    """
    saturation = np.atleast_1d(outlet.saturation)  # [-]
    conc = np.atleast_2d(outlet.mass_concentr)  # [kg/m**3], nodes x species
    pore_volume = (outlet.porosity * outlet.cake_vol
                   * np.sum(widths * saturation))  # [m**3]
    species_mass = (outlet.porosity * outlet.cake_vol
                    * (widths * saturation) @ conc)  # [kg]
    liquid = outlet.Liquid_1
    assert liquid.vol == pytest.approx(pore_volume, rel=RTOL)
    np.testing.assert_allclose(liquid.mass_conc, species_mass / pore_volume,
                               rtol=RTOL, atol=0)
    assert liquid.mass == pytest.approx(species_mass.sum(), rel=RTOL)
    assert liquid.concentration_basis is True


def _washing_widths(unit):
    """Return the washing cells' widths over the cake height.

    Parameters
    ----------
    unit : DisplacementWashing
        Solved unit with node coordinates ``zProf`` [m].

    Returns
    -------
    numpy.ndarray
        Widths [-]: faces halfway between nodes and at 0 and cake height.
    """
    nodes = np.asarray(unit.zProf, dtype=float)  # [m]
    height = unit.CakePhase.cake_height  # [m]
    faces = np.concatenate(([0.0], (nodes[1:] + nodes[:-1]) / 2, [height]))  # [m]
    return np.diff(faces) / height


@pytest.mark.unit
def test_washing_hands_back_concentration_pore_liquid():
    unit = DisplacementWashing(solvent_idx=SOLVENT_INDEX, num_nodes=4,
                               diam_unit=UNIT_DIAMETER)
    unit.Phases = _phases('concentrations')
    unit.solve_unit(deltaP=WASH_PRESSURE_DROP, wash_ratio=WASH_RATIO,
                    dynamic=False)
    _assert_pore_liquid(unit.Outlet, _washing_widths(unit))


@pytest.mark.unit
def test_washing_keeps_fraction_liquid_on_ideal_mixing():
    unit = DisplacementWashing(solvent_idx=SOLVENT_INDEX, num_nodes=4,
                               diam_unit=UNIT_DIAMETER)
    unit.Phases = _phases('fractions')
    unit.solve_unit(deltaP=WASH_PRESSURE_DROP, wash_ratio=WASH_RATIO,
                    dynamic=False)
    liquid = unit.Outlet.Liquid_1
    assert liquid.concentration_basis is False
    # Mass is handed back, so the ideal-mixing path computes vol = mass / rho.
    assert liquid.vol == liquid.mass / liquid.getDensity()


def _deliquored(liquid_basis):
    """Deliquor the fixture cake.

    Parameters
    ----------
    liquid_basis : {'concentrations', 'fractions'}
        Basis of the cake liquid; see :func:`_phases`.

    Returns
    -------
    DeliquoringStep
        Unit solved for DELIQUORING_TIME [s] at DELIQUORING_PRESSURE_DROP
        [Pa].
    """
    unit = DeliquoringStep(num_nodes=3, diam_unit=UNIT_DIAMETER)
    cake = Cake(num_discr=3)
    cake.Phases = _phases(liquid_basis)
    unit.Phases = cake
    cake.z_external = unit.z_centers * unit.cake_height  # [m]
    unit.solve_unit(deltaP=DELIQUORING_PRESSURE_DROP, runtime=DELIQUORING_TIME,
                    verbose=False)
    return unit


@pytest.mark.integration
@pytest.mark.assimulo
def test_deliquoring_hands_back_concentration_pore_liquid():
    pytest.importorskip('assimulo')
    unit = _deliquored('concentrations')
    _assert_pore_liquid(unit.Outlet, unit.delta_z)


@pytest.mark.integration
@pytest.mark.assimulo
def test_deliquoring_keeps_fraction_liquid_on_ideal_mixing():
    pytest.importorskip('assimulo')
    unit = _deliquored('fractions')
    liquid = unit.Outlet.Liquid_1
    assert liquid.concentration_basis is False
    # Mass is handed back, so the ideal-mixing path computes vol = mass / rho.
    assert liquid.vol == liquid.mass / liquid.getDensity()
