"""Authoritative liquid concentrations without a solvent (issue #439).

A ``LiquidPhase`` or ``LiquidStream`` built or updated from ``mole_conc``
[mol/L] or ``mass_conc`` [kg/m**3] without ``name_solv`` must hold the
inventory its concentrations describe: ``moles = 1000 * sum(C) * vol`` [mol]
and ``mass = sum(C * MW) * vol`` [kg], whatever amount is supplied, so that
``mole_conc * vol * 1000 == mole_frac * moles`` and
``sum(mass_conc) * vol == mass``. Consumers that convert the inventory to a
volume (slurries, including their crystal population and mixing
temperature, ``Filter``, raw-material rows, ``BatchToFlowConnector`` and
``NameAnalyzer``) must keep it, while ``name_solv`` and fraction inputs keep
the ideal-mixing contract. ``NameAnalyzer`` treats fractions as a liquid's
held state only when they equal its stored fractions.

Fixture: the shipped PFR property file (species A, B, C, solv) with the
issue's asymmetric, non-closing concentrations
``[0.15, 0.10, 0.02, 5.0]`` mol/L in 2 L, for which
``sum_i C_i * MW_i / rho_i = 0.435`` instead of 1. Expected values are
derived from the JSON ``mw`` and ``rho_liq`` entries (and checked against
hand-evaluated totals), never from PharmaPy conversions. Tests marked
``assimulo`` run real CVode solves of batch and continuous reactors through
``SimulationExec`` and of a ``Filter``; the others construct real phases,
streams, slurries, connectors, mixers and connections without a solver.
One case calls ``SimulationExec._account_dynamic_stream`` directly, because
no shipped unit consumes a concentration with a mass flow. A vapor case
uses the shared two-species synthetic database of ``conftest``.

Refs: https://github.com/PharmaPy-org/PharmaPy/issues/439
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy import Reactors
from PharmaPy.Commons import unpack_states
from PharmaPy.Connections import Connection
from PharmaPy.Containers import Mixer
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Evaporators import ContinuousEvaporator, Evaporator
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.NameAnalysis import NameAnalyzer
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import (BatchToFlowConnector, LiquidStream,
                              SolidStream, VaporStream)
from PharmaPy.ThreePhaseSettler import ThreePhaseSettler
from PharmaPy.Utilities import CoolingWater
from conftest import THERMO_TWO_SPECIES
from test_simexec_raw_materials import _collector_with_inlet

THERMO_PATH = str(Path(__file__).parent
                  / 'integration/data/pfr_test_pure_comp.json')
with open(THERMO_PATH) as handle:
    _PROPERTIES = json.load(handle)
MW = np.array([value['mw'] for value in _PROPERTIES.values()])  # [g/mol], A, B, C, solv
RHO_LIQ = np.array([value['rho_liq'] for value in _PROPERTIES.values()])  # [kg/m**3]
MOLAR_VOLUME = MW / RHO_LIQ  # [L/mol], pure-species molar volume

CONC = np.array([0.15, 0.10, 0.02, 5.0])  # [mol/L], issue #439 fixture, non-closing
VOLUME = 0.002  # [m**3], 2 L charge
VOL_FLOW = 1.0e-6  # [m**3/s], synthetic stream flow
TEMP = 320.0  # [K]

# Independent expectations from the JSON. By hand:
# sum C*MW = 0.15*105.52 + 0.10*99.1741 + 0.02*204.6941 + 5.0*72.107
#          = 390.374292 kg/m**3, so 2 L hold 0.780748584 kg and
# 1000 * sum C * V = 1000 * 5.27 * 0.002 = 10.54 mol.
MASS_DENSITY = CONC @ MW  # [kg/m**3]
MOLAR_DENSITY = CONC.sum()  # [mol/L]
SPECIES_MOLES = CONC * VOLUME * 1000  # [mol], 1000 L/m**3
TOTAL_MASS = MASS_DENSITY * VOLUME  # [kg]
HAND_MASS_DENSITY = 390.374292  # [kg/m**3], evaluated above
HAND_TOTAL_MOLES = 10.54  # [mol], evaluated above

# Float64 round-off of four-term sums and a few products or quotients.
RTOL = 1e-12  # [-]
# A closing input differs from ideal mixing only by a few ulps (#439 survey:
# at most 5.9e-16 relative over 2000 random closing inputs).
CLOSURE_RTOL = 1e-14  # [-]


def ideal_mixing_density(mass_frac):
    """Return the ideal-mixing liquid density of the JSON species by hand.

    Parameters
    ----------
    mass_frac : numpy.ndarray
        Species mass fractions [-], shape (num_species,).

    Returns
    -------
    float
        ``1 / sum_i(w_i / rho_i)`` [kg/m**3].
    """
    return 1 / np.sum(np.asarray(mass_frac) / RHO_LIQ)


def assert_concentration_inventory(liquid, amounts, rtol=RTOL):
    """Check a liquid's amounts against the fixture concentrations.

    Parameters
    ----------
    liquid : LiquidPhase or LiquidStream
        Liquid built from ``CONC``.
    amounts : tuple of float
        Expected volume [m**3], mass [kg] and moles [mol] (per second for a
        stream).
    rtol : float, optional
        Relative tolerance [-].
    """
    vol, mass, moles = amounts
    assert liquid.concentration_basis is True
    np.testing.assert_allclose(liquid.mole_conc, CONC, rtol=rtol, atol=0)
    np.testing.assert_allclose(liquid.mass_conc, CONC * MW, rtol=rtol, atol=0)
    np.testing.assert_allclose([liquid.vol, liquid.mass, liquid.moles],
                               [vol, mass, moles], rtol=rtol, atol=0)
    np.testing.assert_allclose(liquid.mole_frac * liquid.moles,
                               liquid.mole_conc * liquid.vol * 1000,
                               rtol=rtol, atol=0)
    assert np.sum(liquid.mass_conc) * liquid.vol == pytest.approx(
        liquid.mass, rel=rtol, abs=0)


@pytest.mark.unit
def test_fixture_totals_match_hand_evaluation():
    assert MASS_DENSITY == pytest.approx(HAND_MASS_DENSITY, rel=RTOL)
    assert SPECIES_MOLES.sum() == pytest.approx(HAND_TOTAL_MOLES, rel=RTOL)
    # The fixture must not close ideal mixing (sum C_i v_i = 1), or the cases
    # below are vacuous; issue #439 reports sum C_i v_i = 0.435 [-].
    assert CONC @ MOLAR_VOLUME < 0.5


@pytest.mark.unit
@pytest.mark.parametrize('comp_name', ['mole_conc', 'mass_conc'])
@pytest.mark.parametrize('amount', ['vol', 'mass', 'moles'])
def test_phase_amounts_follow_concentrations(amount, comp_name):
    value = {'vol': VOLUME, 'mass': TOTAL_MASS,
             'moles': SPECIES_MOLES.sum()}[amount]  # [m**3], [kg] or [mol]
    composition = CONC if comp_name == 'mole_conc' else CONC * MW  # [mol/L] or [kg/m**3]
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, **{amount: value},
                         **{comp_name: composition.copy()})
    assert_concentration_inventory(
        liquid, (VOLUME, TOTAL_MASS, SPECIES_MOLES.sum()))
    np.testing.assert_allclose(liquid.mole_frac * liquid.moles, SPECIES_MOLES,
                               rtol=RTOL, atol=0)
    assert liquid.getInventoryDensity() == pytest.approx(MASS_DENSITY, rel=RTOL)
    assert liquid.getInventoryDensity('mole') == pytest.approx(MOLAR_DENSITY,
                                                               rel=RTOL)
    # The property model stays ideal mixing of the same composition.
    assert liquid.getDensity() == pytest.approx(
        ideal_mixing_density(CONC * MW / MASS_DENSITY), rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('flow_name', ['vol_flow', 'mass_flow', 'mole_flow'])
def test_stream_flows_follow_concentrations(flow_name):
    value = {'vol_flow': VOL_FLOW, 'mass_flow': VOL_FLOW * MASS_DENSITY,
             'mole_flow': VOL_FLOW * MOLAR_DENSITY * 1000}[flow_name]
    # [m**3/s], [kg/s] or [mol/s]
    stream = LiquidStream(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy(),
                          **{flow_name: value})
    np.testing.assert_allclose(
        [stream.vol_flow, stream.mass_flow, stream.mole_flow],
        [VOL_FLOW, VOL_FLOW * MASS_DENSITY, VOL_FLOW * MOLAR_DENSITY * 1000],
        rtol=RTOL, atol=0)
    np.testing.assert_allclose(stream.mole_frac * stream.mole_flow,
                               CONC * VOL_FLOW * 1000, rtol=RTOL, atol=0)
    assert stream.concentration_basis is True


@pytest.mark.unit
def test_profile_amounts_keep_row_shape():
    rows = np.vstack([CONC, CONC[::-1]])  # [mol/L], asymmetric second row
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, mole_conc=rows)
    assert np.shape(liquid.mass) == (2,)
    np.testing.assert_allclose(liquid.mass, rows @ MW * VOLUME, rtol=RTOL, atol=0)
    np.testing.assert_allclose(liquid.moles, rows.sum(axis=1) * VOLUME * 1000,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(liquid.getInventoryDensity(), rows @ MW,
                               rtol=RTOL, atol=0)


PROFILE_CONC = np.array([[0.15, 0.10, 0.02, 5.0],
                         [0.10, 0.15, 0.03, 6.0]])  # [mol/L], two asymmetric points


def assert_profile_invariants(liquid):
    """Check the inventory invariants at every point of a profile.

    Parameters
    ----------
    liquid : LiquidPhase
        Phase with a ``(num_points, num_species)`` composition.
    """
    np.testing.assert_allclose(liquid.mole_frac * np.reshape(liquid.moles, (-1, 1)),
                               liquid.mole_conc * np.reshape(liquid.vol, (-1, 1)) * 1000,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(liquid.mass_conc.sum(axis=1) * liquid.vol,
                               liquid.mass, rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('amount', ['retained', 'vol', 'vol_points',
                                    'mass_points', 'moles_points'])
def test_profile_update_reconciles_every_point(amount):
    """Reconcile a concentration profile update point by point (#439).

    The profile is built with 2 L per point, so each point holds
    ``0.002 * sum(C MW)`` [kg]. Updating to twice the concentrations
    without an amount retains those masses and halves the volumes; an
    explicit scalar or per-point amount controls instead. Expected values
    come from the JSON molar masses: ``mass = vol * sum(C MW)`` [kg] and
    ``moles = 1000 * vol * sum(C)`` [mol].

    Parameters
    ----------
    amount : str
        ``'retained'`` for a composition-only update, else the explicit
        amount: scalar volume or per-point volume, mass or moles.
    """
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                         mole_conc=PROFILE_CONC.copy())
    updated = 2 * PROFILE_CONC  # [mol/L]
    point_mass = VOLUME * (PROFILE_CONC @ MW)  # [kg], as charged
    point_volumes = np.array([1e-3, 3e-3])  # [m**3], unequal on purpose
    if amount == 'retained':
        liquid.updatePhase(mole_conc=updated)
        expected_vol = point_mass / (updated @ MW)  # [m**3]
    elif amount == 'vol':
        liquid.updatePhase(mole_conc=updated, vol=VOLUME)
        expected_vol = np.full(2, VOLUME)  # [m**3]
    elif amount == 'vol_points':
        liquid.updatePhase(mole_conc=updated, vol=point_volumes)
        expected_vol = point_volumes  # [m**3]
    elif amount == 'mass_points':
        liquid.updatePhase(mole_conc=updated, mass=point_volumes * (updated @ MW))
        expected_vol = point_volumes  # [m**3]
    else:
        liquid.updatePhase(mole_conc=updated,
                           moles=1000 * point_volumes * updated.sum(axis=1))
        expected_vol = point_volumes  # [m**3]
    np.testing.assert_allclose(liquid.vol, expected_vol, rtol=RTOL, atol=0)
    np.testing.assert_allclose(liquid.mass, expected_vol * (updated @ MW),
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(liquid.moles, 1000 * expected_vol * updated.sum(axis=1),
                               rtol=RTOL, atol=0)
    assert_profile_invariants(liquid)
    if amount == 'retained':
        np.testing.assert_allclose(liquid.mass, point_mass, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_profile_amount_of_another_length_is_rejected():
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                         mole_conc=PROFILE_CONC.copy())
    with pytest.raises(PharmaPyValueError,
                       match=r"2 points but its 'vol' has shape \(3,\)"):
        liquid.updatePhase(mole_conc=PROFILE_CONC, vol=np.full(3, VOLUME))


@pytest.mark.unit
def test_profile_point_without_material_may_have_zero_concentrations():
    rows = np.vstack([CONC, np.zeros(len(MW))])  # [mol/L], second point empty
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                         mole_conc=PROFILE_CONC.copy())
    with np.errstate(invalid='ignore'):  # 0/0 fractions of the empty point
        liquid.updatePhase(mole_conc=rows, vol=np.array([VOLUME, 0.0]))
    np.testing.assert_allclose(liquid.mass, [TOTAL_MASS, 0.0], rtol=RTOL, atol=0)
    with pytest.raises(PharmaPyValueError, match="sum to zero"):
        with np.errstate(invalid='ignore'):
            liquid.updatePhase(mole_conc=rows, vol=np.array([VOLUME, VOLUME]))


EMPTY_POINT_ROWS = np.vstack([CONC, np.zeros(len(MW))])  # [mol/L], 2nd point empty
EMPTY_POINT_FRACTIONS = np.vstack([[0.4, 0.1, 0.2, 0.3],
                                   np.zeros(len(MW))])  # [-], 2nd point empty


@pytest.mark.unit
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('amount', ['retained', 'mass', 'moles', 'fractions_mass',
                                    'fractions_vol', 'fractions_moles'])
def test_empty_profile_point_keeps_zero_inventory(amount, stream):
    """A point without material keeps zero mass, volume and moles.

    The empty point's composition is all zero, so its fractions are
    undefined (0/0), but its inventory must be exactly zero, not NaN, on
    every reconciliation path: a retained mass, an explicit per-point mass
    or moles on the concentration basis, and an explicit per-point mass,
    volume or moles with a zero fraction row. The filled point holds the
    fixture charge, ``0.002 * sum(C MW)`` [kg] and ``1000 * 0.002 * sum(C)``
    [mol]; with fractions w its mass m, volume ``m sum(w / rho)`` [m**3] and
    moles ``1000 m sum(w / MW)`` [mol] follow from the JSON by hand.

    Parameters
    ----------
    amount : str
        Reconciliation path of the second update.
    stream : bool
        Use a ``LiquidStream`` and check its flow aliases instead.
    """
    point_vol = np.array([VOLUME, 0.0])  # [m**3] or [m**3/s]
    if stream:
        liquid = LiquidStream(THERMO_PATH, temp=TEMP, vol_flow=VOLUME,
                              mole_conc=PROFILE_CONC.copy())
        names = {'mass': 'mass_flow', 'vol': 'vol_flow', 'moles': 'mole_flow'}
        conc_name = 'concentr'
    else:
        liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                             mole_conc=PROFILE_CONC.copy())
        names = {'mass': 'mass', 'vol': 'vol', 'moles': 'moles'}
        conc_name = 'mole_conc'
    fraction_row = EMPTY_POINT_FRACTIONS[0]  # [-]
    fraction_mass = TOTAL_MASS  # [kg], filled point of the fraction cases
    fraction_vol = fraction_mass * (fraction_row @ (1 / RHO_LIQ))  # [m**3]
    fraction_moles = 1000 * fraction_mass * (fraction_row @ (1 / MW))  # [mol]
    explicit = {'mass': ('mass', TOTAL_MASS),
                'moles': ('moles', SPECIES_MOLES.sum()),
                'fractions_mass': ('mass', fraction_mass),
                'fractions_vol': ('vol', fraction_vol),
                'fractions_moles': ('moles', fraction_moles)}
    # 0/0 fractions and 1/0 densities of the empty point are undefined
    with np.errstate(invalid='ignore', divide='ignore'):
        liquid.updatePhase(**{conc_name: EMPTY_POINT_ROWS,
                              names['vol']: point_vol})
        kwargs = {}
        if amount in explicit:
            name, value = explicit[amount]
            kwargs = {names[name]: np.array([value, 0.0])}  # [kg], [m**3] or [mol]
        if amount.startswith('fractions'):
            liquid.updatePhase(mass_frac=EMPTY_POINT_FRACTIONS, **kwargs)
            expected = [fraction_mass, fraction_vol, fraction_moles]
        else:
            liquid.updatePhase(**{conc_name: EMPTY_POINT_ROWS}, **kwargs)
            expected = [TOTAL_MASS, VOLUME, SPECIES_MOLES.sum()]  # [kg], [m**3], [mol]
    inventory = [np.asarray(getattr(liquid, names[name]))
                 for name in ('mass', 'vol', 'moles')]
    for values in inventory:
        assert np.all(np.isfinite(values))
        assert values[1] == 0
    np.testing.assert_allclose([values[0] for values in inventory], expected,
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('frac_name', ['mass_frac', 'mole_frac'])
def test_update_transitions_keep_or_switch_the_basis(frac_name):
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, mole_conc=CONC.copy())

    liquid.updatePhase(temp=TEMP + 10)  # [K], intensive-only update
    assert_concentration_inventory(liquid, (VOLUME, TOTAL_MASS, SPECIES_MOLES.sum()))

    liquid.updatePhase(vol=2 * VOLUME)  # [m**3], amount-only update keeps C
    assert_concentration_inventory(
        liquid, (2 * VOLUME, 2 * TOTAL_MASS, 2 * SPECIES_MOLES.sum()))

    fractions = np.array([0.4, 0.1, 0.2, 0.3])  # [-], asymmetric fractions
    if frac_name == 'mass_frac':
        mass_frac = fractions  # [-]
    else:
        mass_frac = fractions * MW / (fractions @ MW)  # [-], by hand
    liquid.updatePhase(**{frac_name: fractions})  # retains mass, ideal mixing
    assert liquid.concentration_basis is False
    assert liquid.mass == pytest.approx(2 * TOTAL_MASS, rel=RTOL)
    assert liquid.vol == pytest.approx(
        2 * TOTAL_MASS / ideal_mixing_density(mass_frac), rel=RTOL)
    assert liquid.getInventoryDensity() == liquid.getDensity()

    liquid.updatePhase(mole_conc=CONC.copy(), vol=VOLUME)  # back to C
    assert_concentration_inventory(liquid, (VOLUME, TOTAL_MASS, SPECIES_MOLES.sum()))

    liquid.updatePhase(mass_conc=2 * CONC * MW)  # [kg/m**3], retains mass
    assert liquid.concentration_basis is True
    assert liquid.mass == pytest.approx(TOTAL_MASS, rel=RTOL)
    assert liquid.vol == pytest.approx(VOLUME / 2, rel=RTOL)


@pytest.mark.unit
def test_stream_concentration_update_conserves_mass_flow():
    mass_frac = np.array([0.4, 0.1, 0.2, 0.3])  # [-]
    mass_flow = 2.0e-3  # [kg/s]
    stream = LiquidStream(THERMO_PATH, temp=TEMP, mass_frac=mass_frac,
                          mass_flow=mass_flow)
    stream.updatePhase(concentr=CONC.copy())
    assert stream.concentration_basis is True
    assert stream.mass_flow == pytest.approx(mass_flow, rel=RTOL)
    assert stream.vol_flow == pytest.approx(mass_flow / MASS_DENSITY, rel=RTOL)
    np.testing.assert_allclose(stream.mole_frac * stream.mole_flow,
                               CONC * stream.vol_flow * 1000, rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('comp_name', ['mole_conc', 'mass_conc'])
def test_solvent_completion_keeps_ideal_mixing(comp_name):
    solvent = 3  # [-], index of 'solv'
    # Solvent completed by hand from sum_i C_i v_i = 1.
    completed = CONC.copy()  # [mol/L]
    completed[solvent] = ((1 - CONC[:solvent] @ MOLAR_VOLUME[:solvent])
                          / MOLAR_VOLUME[solvent])
    composition = CONC if comp_name == 'mole_conc' else CONC * MW  # [mol/L] or [kg/m**3]
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, name_solv='solv',
                         **{comp_name: composition.copy()})
    assert liquid.concentration_basis is False
    np.testing.assert_allclose(liquid.mole_conc, completed, rtol=RTOL, atol=0)
    # The ideal-mixing amount path, bit for bit, and its closure by hand.
    assert liquid.mass == liquid.vol * liquid.getDensity()
    assert liquid.getInventoryDensity() == liquid.getDensity()
    assert liquid.mass == pytest.approx(completed @ MW * VOLUME, rel=RTOL)


@pytest.mark.unit
def test_solvent_updates_stay_on_ideal_mixing():
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, name_solv='solv',
                         mass_frac=[0.4, 0.1, 0.2, 0.3])  # [-]
    liquid.updatePhase(mole_conc=CONC.copy())  # [mol/L], retains mass
    assert liquid.concentration_basis is False
    # Mass is retained, so the ideal-mixing path computes vol = mass / rho.
    assert liquid.vol == liquid.mass / liquid.getDensity()
    liquid.updatePhase(mass_conc=CONC * MW, vol=VOLUME)  # [kg/m**3], [m**3]
    assert liquid.concentration_basis is False
    assert liquid.vol == VOLUME
    assert liquid.mass == VOLUME * liquid.getDensity()
    # The completed solvent closes ideal mixing, so the amounts agree with
    # the concentrations too (by hand: sum C_i v_i = 1).
    assert liquid.mole_conc @ MOLAR_VOLUME == pytest.approx(1, rel=RTOL)
    assert liquid.mass == pytest.approx(liquid.mass_conc.sum() * VOLUME, rel=RTOL)


@pytest.mark.unit
def test_inventory_density_rejects_unknown_basis():
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, mole_conc=CONC.copy())
    with pytest.raises(ValueError, match="basis must be 'mass' or 'mole'"):
        liquid.getInventoryDensity(basis='molar')


@pytest.mark.unit
def test_closing_concentrations_agree_with_ideal_mixing():
    closing = CONC.copy()  # [mol/L]
    closing[3] = (1 - CONC[:3] @ MOLAR_VOLUME[:3]) / MOLAR_VOLUME[3]
    assert closing @ MOLAR_VOLUME == pytest.approx(1, rel=CLOSURE_RTOL)
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, mole_conc=closing)
    mass_frac = closing * MW / (closing @ MW)  # [-]
    assert liquid.concentration_basis is True
    assert liquid.mass == pytest.approx(VOLUME * ideal_mixing_density(mass_frac),
                                        rel=CLOSURE_RTOL)
    assert liquid.getInventoryDensity() == pytest.approx(liquid.getDensity(),
                                                         rel=CLOSURE_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('amount', ['vol', 'mass', 'moles'])
def test_zero_total_concentration_with_amount_is_rejected(amount):
    with pytest.raises(PharmaPyValueError, match=f"sum to zero.*'{amount}'"):
        with np.errstate(invalid='ignore'):  # 0/0 fractions before the check
            LiquidPhase(THERMO_PATH, temp=TEMP, mole_conc=np.zeros(len(MW)),
                        **{amount: 1.0})  # [m**3], [kg] or [mol]


@pytest.mark.unit
def test_zero_concentration_update_of_held_mass_is_rejected():
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, mole_conc=CONC.copy())
    with pytest.raises(PharmaPyValueError, match="sum to zero.*'mass'"):
        with np.errstate(invalid='ignore'):
            liquid.updatePhase(mass_conc=np.zeros(len(MW)))


def _crystal_population():
    """Return a size grid and a volume-specific number distribution.

    Returns
    -------
    tuple of numpy.ndarray
        Grid [um] and distribution [#/m**3/um] whose third moment is
        0.2 m**3/m**3 by the trapezoidal rule on the 100 um grid:
        100 * (1e9 * 100**3 + 1.25e8 * 200**3) * 1e-18 = 0.2.
    """
    x_distrib = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
    distrib = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]
    return x_distrib, distrib


SHAPE_FACTOR = 0.5  # [-], volumetric shape factor, non-unit on purpose
SOLID_VOL_FRAC = SHAPE_FACTOR * 0.2  # [-], kv times the third moment above


@pytest.mark.unit
@pytest.mark.parametrize('route', ['distribution', 'moments'])
def test_slurry_stream_hands_over_liquid_volume_flow(route):
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        liquid = LiquidStream(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy())
        solid = SolidStream(THERMO_PATH, temp=TEMP, mass_frac=[1, 0, 0, 0],
                            kv=SHAPE_FACTOR)  # pure A crystals
    if route == 'distribution':
        x_distrib, distrib = _crystal_population()
        slurry = SlurryStream(vol_flow=VOL_FLOW, x_distrib=x_distrib,
                              distrib=distrib)
    else:
        # [#/m**3, m/m**3, m**2/m**3, m**3/m**3]; only mu_3 sets the split.
        moments = np.array([1.0e8, 1.0e4, 1.0, 0.2])
        slurry = SlurryStream(vol_flow=VOL_FLOW, moments=moments)
    slurry.Phases = (liquid, solid)
    liquid_vol_flow = (1 - SOLID_VOL_FRAC) * VOL_FLOW  # [m**3/s]
    assert slurry.Liquid_1.vol_flow == pytest.approx(liquid_vol_flow, rel=RTOL)
    assert slurry.Liquid_1.mass_flow == pytest.approx(
        liquid_vol_flow * MASS_DENSITY, rel=RTOL)
    np.testing.assert_allclose(slurry.Liquid_1.mole_conc, CONC, rtol=RTOL, atol=0)
    assert slurry.vol_flow == pytest.approx(VOL_FLOW, rel=RTOL)


@pytest.mark.unit
def test_slurry_hands_over_liquid_volume():
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        liquid = LiquidPhase(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy())
    solid = SolidPhase(THERMO_PATH, temp=TEMP, mass_frac=[1, 0, 0, 0],
                       kv=SHAPE_FACTOR)  # pure A crystals
    x_distrib, distrib = _crystal_population()
    slurry = Slurry(vol=VOLUME, x_distrib=x_distrib, distrib=distrib)
    slurry.Phases = (liquid, solid)
    liquid_volume = (1 - SOLID_VOL_FRAC) * VOLUME  # [m**3]
    assert slurry.Liquid_1.vol == pytest.approx(liquid_volume, rel=RTOL)
    # Handoff identity: the slurry's own liquid share [m**3] is handed over
    # as the liquid volume, not recovered from a mass by a density.
    assert slurry.Liquid_1.vol == slurry.getFractions()[0] * VOLUME
    assert slurry.Liquid_1.mass == pytest.approx(liquid_volume * MASS_DENSITY,
                                                 rel=RTOL)
    assert slurry.getTotalVol() == pytest.approx(VOLUME, rel=RTOL)
    assert slurry.mass_slurry == pytest.approx(
        slurry.Liquid_1.mass + slurry.Solid_1.mass, rel=RTOL)


RHO_SOLID_A = _PROPERTIES['A']['rho_solid']  # [kg/m**3], crystals of pure A
SLURRY_MASS = 1.0  # [kg] or [kg/s], known slurry amount for the mass route
UM3_TO_M3 = 1e-18  # [m**3/um**3], exact SI conversion


def _solid_volume(solid):
    """Return kv times the trapezoidal third moment of a total population.

    Parameters
    ----------
    solid : SolidPhase or SolidStream
        Solid with a total population ``distrib`` [#/um] or [#/um/s] on
        ``x_distrib`` [um].

    Returns
    -------
    float
        Crystal volume [m**3] or volume flow [m**3/s].
    """
    integrand = solid.distrib * solid.x_distrib**3  # [#/um * um**3]
    mu_3 = np.sum(np.diff(solid.x_distrib)
                  * (integrand[1:] + integrand[:-1]) / 2)  # [um**3]
    return SHAPE_FACTOR * mu_3 * UM3_TO_M3


def _hand_mass_split():
    """Return the slurry volume and liquid mass of ``SLURRY_MASS`` by hand.

    Returns
    -------
    tuple of float
        Slurry volume ``M / ((1 - phi) rho_l + phi rho_s)`` [m**3] and liquid
        mass ``(1 - phi) rho_l`` times that volume [kg], with the liquid's
        own density ``rho_l = MASS_DENSITY`` and ``phi = SOLID_VOL_FRAC``.
    """
    slurry_density = ((1 - SOLID_VOL_FRAC) * MASS_DENSITY
                      + SOLID_VOL_FRAC * RHO_SOLID_A)  # [kg/m**3]
    volume = SLURRY_MASS / slurry_density  # [m**3]
    return volume, (1 - SOLID_VOL_FRAC) * MASS_DENSITY * volume


@pytest.mark.unit
@pytest.mark.parametrize('route', ['vol', 'mass'])
def test_slurry_population_inventory_and_temperature_agree(route):
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        liquid = LiquidPhase(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy())
    solid = SolidPhase(THERMO_PATH, temp=TEMP, mass_frac=[1, 0, 0, 0],
                       kv=SHAPE_FACTOR)  # pure A crystals
    x_distrib, distrib = _crystal_population()
    if route == 'vol':
        slurry = Slurry(vol=VOLUME, x_distrib=x_distrib, distrib=distrib)
        volume, liquid_mass = VOLUME, (1 - SOLID_VOL_FRAC) * VOLUME * MASS_DENSITY
    else:
        slurry = Slurry(x_distrib=x_distrib, distrib=distrib)
        slurry.mass_slurry = SLURRY_MASS  # [kg]
        volume, liquid_mass = _hand_mass_split()  # [m**3], [kg]
    slurry.Phases = (liquid, solid)
    assert slurry.getTotalVol() == pytest.approx(volume, rel=RTOL)
    assert slurry.vol == pytest.approx(volume, rel=RTOL)
    assert slurry.Liquid_1.mass == pytest.approx(liquid_mass, rel=RTOL)
    assert slurry.Liquid_1.vol == pytest.approx(liquid_mass / MASS_DENSITY, rel=RTOL)
    # The solid mass agrees with its own population, and its volume share
    # with kv times the slurry's third moment.
    assert slurry.Solid_1.mass == pytest.approx(
        RHO_SOLID_A * _solid_volume(slurry.Solid_1), rel=RTOL)
    assert slurry.Solid_1.vol / slurry.getTotalVol() == pytest.approx(
        SOLID_VOL_FRAC, rel=RTOL)
    assert SHAPE_FACTOR * slurry.moments[3] == pytest.approx(SOLID_VOL_FRAC,
                                                            rel=RTOL)
    assert slurry.mass_slurry == pytest.approx(
        slurry.Liquid_1.mass + slurry.Solid_1.mass, rel=RTOL)
    # Both phases are at TEMP, so the mixing energy balance must return it.
    assert slurry.temp == pytest.approx(TEMP, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('route', ['vol', 'mass'])
def test_slurry_stream_population_inventory_and_temperature_agree(route):
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        liquid = LiquidStream(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy())
        solid = SolidStream(THERMO_PATH, temp=TEMP, mass_frac=[1, 0, 0, 0],
                            kv=SHAPE_FACTOR)  # pure A crystals
    x_distrib, distrib = _crystal_population()
    if route == 'vol':
        slurry = SlurryStream(vol_flow=VOL_FLOW, x_distrib=x_distrib,
                              distrib=distrib)
        vol_flow = VOL_FLOW  # [m**3/s]
        liquid_mass_flow = (1 - SOLID_VOL_FRAC) * VOL_FLOW * MASS_DENSITY  # [kg/s]
    else:
        slurry = SlurryStream(x_distrib=x_distrib, distrib=distrib)
        slurry.mass_slurry = SLURRY_MASS  # [kg/s]
        vol_flow, liquid_mass_flow = _hand_mass_split()  # [m**3/s], [kg/s]
    slurry.Phases = (liquid, solid)
    assert slurry.vol_flow == pytest.approx(vol_flow, rel=RTOL)
    assert slurry.Liquid_1.mass_flow == pytest.approx(liquid_mass_flow, rel=RTOL)
    np.testing.assert_allclose(slurry.distrib * slurry.vol_flow,
                               slurry.Solid_1.distrib, rtol=RTOL, atol=0)
    assert slurry.Solid_1.mass_flow == pytest.approx(
        RHO_SOLID_A * _solid_volume(slurry.Solid_1), rel=RTOL)
    assert slurry.Solid_1.vol / slurry.vol_flow == pytest.approx(
        SOLID_VOL_FRAC, rel=RTOL)
    # Solid per liquid mass [-] agrees with the phases' own mass flows.
    assert slurry.solid_conc == pytest.approx(
        slurry.Solid_1.mass_flow / slurry.Liquid_1.mass_flow, rel=RTOL)
    assert slurry.temp == pytest.approx(TEMP, rel=RTOL)


@pytest.mark.unit
def test_raw_inlet_volume_uses_inventory_density():
    mass_frac = np.array([0.1, 0.2, 0.3, 0.4])  # [-], ideal-mixing inlet
    other_volume = 0.003  # [m**3]
    mixer = Mixer()
    mixer.Inlets = [
        LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME, mole_conc=CONC.copy()),
        LiquidPhase(THERMO_PATH, temp=TEMP, vol=other_volume, mass_frac=mass_frac)]
    sim = SimulationExec(THERMO_PATH, flowsheet='M01')
    sim.M01 = mixer
    sim.SolveFlowsheet(verbose=False)
    for basis, amount, expected in (('mole', 'moles', SPECIES_MOLES.sum()),
                                    ('mass', 'mass', TOTAL_MASS)):
        raw = sim.get_raw_inlets(mixer, basis=basis)
        (record,) = raw['Inlet_0'].values()
        assert record[amount] == pytest.approx(expected, rel=RTOL)
        assert record['vol'] == pytest.approx(VOLUME, rel=RTOL)
        (ideal,) = raw['Inlet_1'].values()
        assert ideal['vol'] == pytest.approx(other_volume, rel=RTOL)
    # The mixer's mass balance receives the concentration inventory.
    expected_mixed_mass = TOTAL_MASS + other_volume * ideal_mixing_density(mass_frac)
    assert np.ravel(mixer.result.mass)[-1] == pytest.approx(expected_mixed_mass,
                                                            rel=RTOL)


CYCLE_TIME = 3600.0  # [s]
FLOW_MULT = 2.0  # [-], non-unit so a missing factor shows


def _discharged_connector():
    """Discharge the fixture charge through a solved connector.

    Returns
    -------
    BatchToFlowConnector
        Connector holding ``CONC`` in ``VOLUME``, solved once.
    """
    connector = BatchToFlowConnector(cycle_time=CYCLE_TIME, flow_mult=FLOW_MULT)
    connector.Phases = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                                   mole_conc=CONC.copy())
    connector.solve_unit()
    return connector


@pytest.mark.unit
def test_connector_outlet_keeps_concentrations():
    connector = _discharged_connector()
    vol_flow = VOLUME / CYCLE_TIME * FLOW_MULT  # [m**3/s]
    outlet = connector.Outlet
    np.testing.assert_allclose(outlet.mole_conc, CONC, rtol=RTOL, atol=0)
    assert outlet.vol_flow == pytest.approx(vol_flow, rel=RTOL)
    assert outlet.mass_flow == pytest.approx(vol_flow * MASS_DENSITY, rel=RTOL)
    np.testing.assert_allclose(outlet.mole_frac * outlet.mole_flow,
                               CONC * vol_flow * 1000, rtol=RTOL, atol=0)
    assert connector.outputs['mass_flow'][0] == pytest.approx(
        TOTAL_MASS / CYCLE_TIME * FLOW_MULT, rel=RTOL)


@pytest.mark.unit
def test_connection_hands_connector_concentrations_to_cstr():
    connector = _discharged_connector()
    cstr = Reactors.CSTR(isothermal=True)
    cstr.Phases = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                              mole_conc=CONC.copy())
    Connection(connector, cstr).transfer_data()
    converted = cstr.Inlet.y_inlet
    assert converted['mole_conc'].shape == (1, len(MW))
    np.testing.assert_allclose(converted['mole_conc'][0], CONC, rtol=RTOL, atol=0)
    np.testing.assert_allclose(converted['vol_flow'],
                               [VOLUME / CYCLE_TIME * FLOW_MULT], rtol=RTOL, atol=0)


def _stream_with_profile(names, values):
    """Attach an upstream profile to a concentration-basis stream.

    Parameters
    ----------
    names : list of str
        Upstream state names.
    values : list
        Upstream values, scalar/1-D (static) or with a time axis.

    Returns
    -------
    LiquidStream
        Stream built from ``CONC`` [mol/L] and ``VOL_FLOW`` [m**3/s].
    """
    stream = LiquidStream(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy(),
                          vol_flow=VOL_FLOW)
    stream.y_upstream = dict(zip(names, values))
    return stream


FIXTURE_MASS_FRAC = CONC * MW / MASS_DENSITY  # [-], fractions of the fixture


@pytest.mark.unit
@pytest.mark.parametrize('samples', [0, 1])
def test_held_state_fractions_use_inventory_density(samples):
    """Static or one-sample fraction states convert with the held density.

    Parameters
    ----------
    samples : int
        0 for a static source (no time axis), 1 for a one-sample profile.
    """
    mass_flow = VOL_FLOW * MASS_DENSITY  # [kg/s]
    composition, flow = FIXTURE_MASS_FRAC, mass_flow
    if samples:
        composition, flow = composition[np.newaxis], np.array([flow])
    stream = _stream_with_profile(['mass_frac', 'mass_flow', 'temp'],
                                  [composition, flow, TEMP])
    analyzer = NameAnalyzer(['mass_frac', 'mass_flow', 'temp'],
                            ['mole_conc', 'vol_flow', 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    np.testing.assert_allclose(np.ravel(converted['vol_flow']), VOL_FLOW,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(np.reshape(converted['mole_conc'], -1), CONC,
                               rtol=RTOL, atol=0)


@pytest.mark.unit
def test_multi_sample_fractions_keep_ideal_mixing():
    rows = np.vstack([FIXTURE_MASS_FRAC, [0.4, 0.1, 0.2, 0.3]])  # [-]
    mass_flow = np.array([2.0e-3, 1.0e-3])  # [kg/s]
    stream = _stream_with_profile(['mass_frac', 'mass_flow', 'temp'],
                                  [rows, mass_flow, np.full(2, TEMP)])
    analyzer = NameAnalyzer(['mass_frac', 'mass_flow', 'temp'],
                            ['mole_conc', 'vol_flow', 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    density = np.array([ideal_mixing_density(row) for row in rows])  # [kg/m**3]
    np.testing.assert_allclose(converted['vol_flow'], mass_flow / density,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(converted['mole_conc'],
                               rows * density[:, np.newaxis] / MW, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_one_sample_of_another_composition_keeps_ideal_mixing():
    """A one-sample profile unlike the carrier's fractions is not its state.

    Pure A [1, 0, 0, 0] carried by the fixture stream must convert with
    pure A's database density, 1230 kg/m**3, and exactly as the same sample
    duplicated into a two-sample profile, not with the carrier's
    ``sum(mass_conc)`` = 390.37 kg/m**3.
    """
    pure_a = np.array([1.0, 0.0, 0.0, 0.0])  # [-]
    mass_flow = 1.0e-3  # [kg/s]
    analyzer = NameAnalyzer(['mass_frac', 'mass_flow', 'temp'],
                            ['mole_conc', 'vol_flow', 'temp'], len(MW))
    converted = {}
    for samples in (1, 2):
        stream = _stream_with_profile(
            ['mass_frac', 'mass_flow', 'temp'],
            [np.tile(pure_a, (samples, 1)), np.full(samples, mass_flow),
             np.full(samples, TEMP)])
        converted[samples] = analyzer.convertUnits(stream)
    np.testing.assert_allclose(converted[1]['vol_flow'], [mass_flow / RHO_LIQ[0]],
                               rtol=RTOL, atol=0)
    np.testing.assert_array_equal(converted[1]['vol_flow'],
                                  converted[2]['vol_flow'][:1])
    np.testing.assert_allclose(converted[1]['mole_conc'][0],
                               pure_a * RHO_LIQ[0] / MW, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(converted[1]['mole_conc'],
                                  converted[2]['mole_conc'][:1])


@pytest.mark.unit
@pytest.mark.parametrize('carrier', ['concentrations', 'fractions'])
def test_static_fractions_of_another_composition_keep_ideal_mixing(carrier):
    """Static pure-A fractions convert with pure A's own density.

    The static conversion must use the ideal-mixing density and molar mass
    of the upstream composition, pure A's 1230 kg/m**3 and 105.52 g/mol,
    whatever the carrier stores: neither the fixture stream's
    ``sum(mass_conc)`` = 390.37 kg/m**3 nor the ideal density of a
    fraction-built carrier's own composition. So ``vol_flow = mass_flow /
    1230`` [m**3/s], ``mole_flow = 1000 mass_flow / 105.52`` [mol/s] and
    ``mole_conc = w * 1230 / MW`` [mol/L].

    Parameters
    ----------
    carrier : {'concentrations', 'fractions'}
        How the transferred stream was built.
    """
    pure_a = np.array([1.0, 0.0, 0.0, 0.0])  # [-]
    mass_flow = 1.0e-3  # [kg/s]
    if carrier == 'concentrations':
        stream = _stream_with_profile(['mass_frac', 'mass_flow', 'temp'],
                                      [pure_a, mass_flow, TEMP])
    else:
        stream = LiquidStream(THERMO_PATH, temp=TEMP,
                              mass_frac=[0.4, 0.1, 0.2, 0.3],
                              mass_flow=mass_flow)  # [-], [kg/s]
        stream.y_upstream = {'mass_frac': pure_a, 'mass_flow': mass_flow,
                             'temp': TEMP}
    converted = {}
    for down_flow, down_comp in (('vol_flow', 'mole_conc'),
                                 ('mole_flow', 'mole_frac')):
        analyzer = NameAnalyzer(['mass_frac', 'mass_flow', 'temp'],
                                [down_comp, down_flow, 'temp'], len(MW))
        converted.update(analyzer.convertUnits(stream))
    assert converted['vol_flow'] == pytest.approx(mass_flow / RHO_LIQ[0], rel=RTOL)
    assert converted['mole_flow'] == pytest.approx(mass_flow * 1000 / MW[0],
                                                   rel=RTOL)
    np.testing.assert_allclose(converted['mole_conc'], pure_a * RHO_LIQ[0] / MW,
                               rtol=RTOL, atol=0)


@pytest.mark.unit
def test_vapor_one_sample_of_its_own_fractions_uses_sample_temperature(tmp_path):
    """Only a liquid with authoritative concentrations has a held state.

    A vapor stream whose one-sample profile repeats its stored fractions at
    another temperature must still convert with the ideal-gas density of
    that sample, ``n = P V / (R T_sample)``.
    """
    gas_path = tmp_path / 'thermo.json'
    gas_path.write_text(json.dumps(THERMO_TWO_SPECIES))
    gas_constant = 8.314  # [J/mol/K], the model's documented rounded value
    pressure = 101325.0  # [Pa]
    stored_temp, sample_temp = 300.0, 350.0  # [K], distinct on purpose
    mole_frac = np.array([0.25, 0.75])  # [-]
    stream = VaporStream(str(gas_path), temp=stored_temp, pres=pressure,
                         mole_frac=mole_frac, mole_flow=1.0)  # [mol/s]
    stream.y_upstream = {'mole_frac': mole_frac[np.newaxis],
                         'vol_flow': np.array([VOL_FLOW]),
                         'temp': np.array([sample_temp])}
    analyzer = NameAnalyzer(['mole_frac', 'vol_flow', 'temp'],
                            ['mole_frac', 'mole_flow', 'temp'], len(mole_frac))
    converted = analyzer.convertUnits(stream)
    np.testing.assert_allclose(
        converted['mole_flow'],
        [VOL_FLOW * pressure / (gas_constant * sample_temp)], rtol=RTOL, atol=0)


@pytest.mark.unit
def test_ideal_stream_one_sample_profile_keeps_ideal_mixing():
    mass_frac = np.array([0.4, 0.1, 0.2, 0.3])  # [-], stored, unlike the sample
    mass_flow = 1.0e-3  # [kg/s]
    stream = LiquidStream(THERMO_PATH, temp=TEMP, mass_frac=mass_frac,
                          mass_flow=mass_flow)
    stream.y_upstream = {'mass_frac': FIXTURE_MASS_FRAC[np.newaxis],
                         'mass_flow': np.array([mass_flow]),
                         'temp': np.array([TEMP])}
    analyzer = NameAnalyzer(['mass_frac', 'mass_flow', 'temp'],
                            ['mole_conc', 'vol_flow', 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    density = ideal_mixing_density(FIXTURE_MASS_FRAC)  # [kg/m**3]
    np.testing.assert_allclose(converted['vol_flow'], [mass_flow / density],
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(converted['mole_conc'],
                               [FIXTURE_MASS_FRAC * density / MW], rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('comp_name', ['mole_conc', 'mass_conc'])
def test_concentration_flows_convert_to_volume_with_totals(comp_name):
    composition = CONC if comp_name == 'mole_conc' else CONC * MW  # [mol/L] or [kg/m**3]
    rows = np.vstack([composition, 2 * composition])  # two samples, double total
    mole_flow = np.array([3.0e-3, 1.0e-3])  # [mol/s]
    stream = _stream_with_profile([comp_name, 'mole_flow', 'temp'],
                                  [rows, mole_flow, np.full(2, TEMP)])
    analyzer = NameAnalyzer([comp_name, 'mole_flow', 'temp'],
                            [comp_name, 'vol_flow', 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    molar_density = np.array([1, 2]) * MOLAR_DENSITY * 1000  # [mol/m**3]
    np.testing.assert_allclose(converted['vol_flow'], mole_flow / molar_density,
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('comp_name', ['mole_conc', 'mass_conc'])
def test_dynamic_raw_volume_from_concentrations_and_mass_flow(comp_name):
    """Report the volume of a concentration feed given by mass flow.

    No shipped unit consumes a concentration together with a mass flow, so
    the accounting helper ``_account_dynamic_stream`` is called directly
    with a real stream and consumed fields. The volume must be the mass
    flow over each sample's total mass concentration, not over the
    ideal-mixing density of its fractions.
    """
    times = np.array([0.0, 10.0])  # [s]
    rows = np.vstack([CONC, 2 * CONC])  # [mol/L], second sample twice as dense
    mass_flow = np.array([2.0e-3, 1.0e-3])  # [kg/s]
    composition = rows if comp_name == 'mole_conc' else rows * MW  # [mol/L] or [kg/m**3]
    stream = LiquidStream(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy(),
                          vol_flow=VOL_FLOW)
    sim = SimulationExec(THERMO_PATH, flowsheet='M01')
    record = sim._account_dynamic_stream(
        stream, {comp_name: composition, 'mass_flow': mass_flow,
                 'temp': np.full(2, TEMP)}, ['mass_flow'], times, 'mass')
    vol_flow = mass_flow / (np.array([1, 2]) * MASS_DENSITY)  # [m**3/s]
    expected_vol = (vol_flow[0] + vol_flow[1]) / 2 * times[1]  # [m**3], trapezoid
    assert record['vol'] == pytest.approx(expected_vol, rel=RTOL)
    assert record['mass'] == pytest.approx(mass_flow.mean() * times[1], rel=RTOL)


RAW_TIMES = np.array([0.0, 4.0, 10.0])  # [s], non-uniform result grid


@pytest.mark.unit
@pytest.mark.parametrize('controlled', [False, True])
def test_raw_feed_volume_with_uncontrolled_composition(controlled):
    """Account a concentration feed whose composition is not controlled.

    A ``DynamicCollector`` consumes ``mass_frac``, ``mass_flow`` and
    ``temp``. With a temperature-only ``DynamicInput`` the consumed fractions
    are the stream's own stored ones, its held state, so the reported
    volume is the stream's ``vol_flow`` [m**3/s] times the duration, as for
    the static row, not the mass over the ideal-mixing density (0.435 of
    it for this fixture). Real stream and collector with a synthetic result
    grid, as in ``test_simexec_raw_materials``; no solver runs.

    Parameters
    ----------
    controlled : bool
        Attach a temperature-only DynamicInput (dynamic accounting) or
        none (static accounting).
    """
    stream = LiquidStream(THERMO_PATH, temp=TEMP, mole_conc=CONC.copy(),
                          vol_flow=VOL_FLOW)
    if controlled:
        control = DynamicInput()
        control.add_variable('temp', lambda time: TEMP + time)  # [K], ramp 1 K/s
        stream.DynamicInlet = control
    sim = SimulationExec(THERMO_PATH, {'U01': []})
    sim.uos_instances = {'U01': _collector_with_inlet(stream, time=RAW_TIMES)}
    record = sim.GetRawMaterials(basis='mass', totals=False).iloc[0]
    duration = RAW_TIMES[-1] - RAW_TIMES[0]  # [s]
    assert record['vol'] == pytest.approx(VOL_FLOW * duration, rel=RTOL)
    assert record['mass'] == pytest.approx(VOL_FLOW * MASS_DENSITY * duration,
                                           rel=RTOL)


EVAPORATOR_DRUM = 1.0  # [m**3]
EVAPORATOR_CHARGE = 0.2  # [m**3], charged liquid volume
EVAPORATOR_PRESSURE = 101325.0  # [Pa]
EVAPORATOR_UTILITY_TEMP = 400.0  # [K], heating utility
EVAPORATOR_FEED = 1.0  # [mol/s], continuous evaporator feed
GAS_CONSTANT = 8.314  # [J/mol/K], PharmaPy.Evaporators.gas_ct


def _evaporator(continuous, **composition):
    """Charge a batch or continuous evaporator with the fixture species.

    Parameters
    ----------
    continuous : bool
        Build a ``ContinuousEvaporator`` (level setpoint at the charge, fed
        by the same composition) instead of an ``Evaporator``.
    **composition
        One composition keyword for the charged ``LiquidPhase``.

    Returns
    -------
    Evaporator or ContinuousEvaporator
        Unsolved unit holding ``EVAPORATOR_CHARGE`` [m**3] at TEMP [K].
    """
    if continuous:
        unit = ContinuousEvaporator(EVAPORATOR_DRUM, pressure=EVAPORATOR_PRESSURE,
                                    frac_liq=EVAPORATOR_CHARGE / EVAPORATOR_DRUM)
    else:
        unit = Evaporator(EVAPORATOR_DRUM, pressure=EVAPORATOR_PRESSURE)
    unit.Phases = LiquidPhase(THERMO_PATH, temp=TEMP, pres=EVAPORATOR_PRESSURE,
                              vol=EVAPORATOR_CHARGE, **composition)
    unit.Utility = CoolingWater(mass_flow=1.0, temp_in=EVAPORATOR_UTILITY_TEMP)  # [kg/s]
    if continuous:
        unit.Inlet = LiquidStream(THERMO_PATH, temp=TEMP, pres=EVAPORATOR_PRESSURE,
                                  mole_flow=EVAPORATOR_FEED, **composition)
    return unit


@pytest.mark.unit
@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('charge', ['concentrations', 'fractions'])
def test_evaporator_seeds_the_ideal_mixing_liquid_volume(continuous, charge):
    """Seed the headspace from the model's own liquid volume.

    The evaporator states are moles and mole fractions with ideal mixing,
    ``V_liq = n sum_i x_i MW_i / rho_i / 1000`` [m**3]. For the non-closing
    fixture charge (n = 1000 * 5.27 * 0.2 = 1054 mol) that is
    0.435 * 0.2 m**3, not the stored 0.2 m**3; for a fraction charge it
    equals the stored volume. The seeded vapor is the ideal gas filling the
    rest of the drum at the seed temperature.

    Parameters
    ----------
    continuous : bool
        Continuous instead of batch evaporator.
    charge : {'concentrations', 'fractions'}
        Fixture concentrations, or asymmetric mass fractions [-].
    """
    composition = ({'mole_conc': CONC.copy()} if charge == 'concentrations'
                   else {'mass_frac': np.array([0.4, 0.1, 0.2, 0.3])})
    unit = _evaporator(continuous, **composition)
    states, _ = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    liquid = unit.Liquid_1
    ideal_volume = (values['mol_liq'] * (liquid.mole_frac @ MOLAR_VOLUME)
                    / 1000)  # [m**3], 1000 L/m**3
    if charge == 'fractions':
        assert ideal_volume == pytest.approx(EVAPORATOR_CHARGE, rel=RTOL)
    else:
        assert values['mol_liq'] == pytest.approx(SPECIES_MOLES.sum() * 100,
                                                  rel=RTOL)  # 0.2 m**3 / 2 L
    expected_vapor = (EVAPORATOR_PRESSURE * (EVAPORATOR_DRUM - ideal_volume)
                      / (GAS_CONSTANT * values['temp']))  # [mol]
    assert values['mol_vap'] == pytest.approx(expected_vapor, rel=RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('continuous', [False, True])
def test_evaporator_starts_consistently_from_a_concentration_charge(continuous):
    """IDA must not move the consistent initial pressure or liquid holdup.

    With the stored volume the initial volume closure was off by -0.113
    m**3 and IDA's consistent initialization moved the pressure from
    101325 Pa to 98779 Pa and the liquid from 1054 to 1050.7 mol.

    Parameters
    ----------
    continuous : bool
        Continuous instead of batch evaporator.
    """
    pytest.importorskip('assimulo')
    unit = _evaporator(continuous, mole_conc=CONC.copy())
    unit.solve_unit(runtime=1.0, verbose=False)
    # [-], IDA's default relative tolerance bounds its initialization change
    assert unit.result.pres[0] == pytest.approx(EVAPORATOR_PRESSURE, rel=1e-6)
    assert unit.result.mol_liq[0] == pytest.approx(SPECIES_MOLES.sum() * 100,
                                                   rel=1e-6)


FILTER_LIQUID_VOLUME = 1e-3  # [m**3]
CRYSTAL_GRID = np.array([100.0, 200.0, 300.0])  # [um]
CRYSTAL_POPULATION = np.array([1e4, 2e4, 1e4])  # [#/um], total crystals


@pytest.mark.unit
@pytest.mark.parametrize('bottom', ['concentrations', 'fractions'])
def test_settler_outlet_keeps_the_bottom_liquid(bottom):
    """Pass the bottom liquid through with its own mass flow.

    The settler's full solve depends on unrelated, untested setup, so
    ``retrieve_results`` is called on a real settler with real feed streams
    and a constant solid concentration [kg solid/kg liquid]; it is the
    method that builds the outlet. The outlet liquid must carry the bottom
    inlet's mass flow and, for a concentration inlet, its concentrations,
    and the outlet solid must be that concentration times the liquid mass
    flow.

    Parameters
    ----------
    bottom : {'concentrations', 'fractions'}
        Composition of the bottom-phase inlet.
    """
    solid_conc = 0.1  # [kg/kg], synthetic settled solid concentration
    settler = ThreePhaseSettler(d_impeller=0.05, d_vessel=0.2,
                                target_compound='A')  # [m], [m]
    settler.nomenclature()
    feed = SlurryStream()
    feed.Phases = [LiquidStream(THERMO_PATH, temp=TEMP, vol_flow=VOL_FLOW,
                                mass_frac=[0.1, 0.2, 0.3, 0.4]),
                   SolidStream(THERMO_PATH, temp=TEMP, mass_frac=[1, 0, 0, 0],
                               x_distrib=CRYSTAL_GRID.copy(),
                               distrib=CRYSTAL_POPULATION.copy(), kv=1.0)]
    settler.Inlet = feed
    composition = ({'mole_conc': CONC.copy()} if bottom == 'concentrations'
                   else {'mass_frac': np.array([0.4, 0.1, 0.2, 0.3])})
    settler.Inlet_bot_phase = LiquidStream(THERMO_PATH, temp=TEMP,
                                           vol_flow=VOL_FLOW, **composition)
    settler.retrieve_results(np.array([0.0, 1.0]),
                             np.full((2, 1), solid_conc))
    liquid_mass_flow = settler.Inlet_bot_phase.mass_flow  # [kg/s]
    if bottom == 'concentrations':
        assert liquid_mass_flow == pytest.approx(VOL_FLOW * MASS_DENSITY, rel=RTOL)
        np.testing.assert_allclose(settler.Outlet.Liquid_1.mole_conc, CONC,
                                   rtol=RTOL, atol=0)
    assert settler.Outlet.Liquid_1.mass_flow == pytest.approx(liquid_mass_flow,
                                                              rel=RTOL)
    assert settler.Outlet.Solid_1.mass_flow == pytest.approx(
        solid_conc * liquid_mass_flow, rel=RTOL)


RESIDENCE_TIME = 1800.0  # [s], shipped PFR case residence time
INTEGRATION_RTOL = 1e-10  # [-], two orders tighter than SOLVER_RTOL
INTEGRATION_ATOL = 1e-12  # [mol/L], far below the smallest concentration
SOLVER_RTOL = 1e-8  # [-], CVode error allowance on the balances


def _inert_kinetics(rate_constant=0.0):
    """Build A + B -> C kinetics on the fixture species.

    Parameters
    ----------
    rate_constant : float, optional
        Rate constant [L/mol/s]; zero makes the mixture inert.

    Returns
    -------
    PharmaPy.Kinetics.RxnKinetics
        Isothermal kinetics with zero activation energy and reaction heat.
    """
    return RxnKinetics(THERMO_PATH, k_params=[rate_constant], ea_params=[0.0],
                       rxn_list=['A + B --> C'], delta_hrxn=0.0)


def _reactor(kind, rate_constant=0.0, inlet_conc=None):
    """Configure a batch reactor or CSTR charged with the fixture.

    Parameters
    ----------
    kind : {'batch', 'cstr'}
        Reactor type.
    rate_constant : float, optional
        Rate constant [L/mol/s].
    inlet_conc : numpy.ndarray, optional
        CSTR feed concentrations [mol/L]; no feed when None.

    Returns
    -------
    object
        Configured, unsolved reactor at ``TEMP`` [K] with ``VOLUME`` [m**3].
    """
    reactor = (Reactors.BatchReactor(isothermal=True) if kind == 'batch'
               else Reactors.CSTR(isothermal=True))
    charge = CONC if inlet_conc is None else inlet_conc  # [mol/L]
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                                 mole_conc=charge.copy())
    reactor.Kinetics = _inert_kinetics(rate_constant)
    # Isothermal, so the utility [K], [kg/s] only completes the setup.
    reactor.Utility = CoolingWater(temp_in=300.0, mass_flow=0.1)
    if inlet_conc is not None:
        reactor.Inlet = LiquidStream(THERMO_PATH, temp=TEMP,
                                     mole_conc=inlet_conc.copy(),
                                     vol_flow=VOLUME / RESIDENCE_TIME)
    return reactor


@pytest.mark.integration
@pytest.mark.assimulo
def test_raw_materials_of_a_batch_charge_follow_concentrations():
    pytest.importorskip('assimulo')
    sim = SimulationExec(THERMO_PATH, flowsheet='R01')
    sim.R01 = _reactor('batch')
    sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': 100., 'verbose': False}},
                       verbose=False)
    moles = sim.GetRawMaterials(basis='mole')
    species = [f'moles_{name}' for name in _PROPERTIES]
    np.testing.assert_allclose(moles[species].values.ravel(), SPECIES_MOLES,
                               rtol=RTOL, atol=0)
    mass = sim.GetRawMaterials(basis='mass')
    np.testing.assert_allclose(mass['mass'].values, [TOTAL_MASS], rtol=RTOL)
    rows = sim.GetRawMaterials(basis='mass', totals=False)
    np.testing.assert_allclose(rows['vol'].values, [VOLUME], rtol=RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
def test_inert_batch_connector_cstr_route_keeps_concentrations():
    pytest.importorskip('assimulo')
    sim = SimulationExec(THERMO_PATH, flowsheet='R01 --> B01 --> R02')
    sim.R01 = _reactor('batch')
    sim.B01 = BatchToFlowConnector(cycle_time=CYCLE_TIME)
    sim.R02 = _reactor('cstr')
    opts = {'rtol': INTEGRATION_RTOL, 'atol': INTEGRATION_ATOL}
    sim.SolveFlowsheet(kwargs_run={
        'R01': {'runtime': 100., 'verbose': False, 'sundials_opts': opts},
        'R02': {'runtime': CYCLE_TIME, 'verbose': False, 'sundials_opts': opts}},
        verbose=False)
    np.testing.assert_allclose(sim.R02.Inlet.y_inlet['mole_conc'][0], CONC,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(sim.R02.Inlet.y_inlet['vol_flow'],
                               [VOLUME / CYCLE_TIME], rtol=RTOL, atol=0)
    np.testing.assert_allclose(sim.R02.result.mole_conc[-1], CONC,
                               rtol=SOLVER_RTOL, atol=0)


@pytest.mark.integration
@pytest.mark.assimulo
def test_reacting_fraction_charged_batch_reactor_conserves_mass():
    """A fraction charge keeps its mass through A + B -> C (#439).

    The reactor updates its liquid from the integrated ``mole_conc`` at the
    charged volume, so after the solve the liquid is on the concentration
    basis and its mass is ``vol * sum(C MW)``. With
    ``sum nu MW = 204.6941 - 105.52 - 99.1741 = 0`` that total mass
    concentration is constant, so the final mass equals the charge, the
    charged volume times the ideal-mixing density of the charge fractions.
    Before #439 the final liquid was rebuilt on the ideal-mixing density of
    the reacted composition and lost 8.2 % of the charge here.
    """
    pytest.importorskip('assimulo')
    mass_frac = np.array([0.4, 0.3, 0.0, 0.3])  # [-], A, B, C, solv
    rate_constant = 1.0e-2  # [L/mol/s], synthetic, converts much of B
    reactor = Reactors.BatchReactor(isothermal=True)
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMP, vol=VOLUME,
                                 mass_frac=mass_frac)
    reactor.Kinetics = _inert_kinetics(rate_constant)
    reactor.Utility = CoolingWater(temp_in=300.0, mass_flow=0.1)  # [K], [kg/s]
    reactor.solve_unit(runtime=10 * RESIDENCE_TIME, verbose=False,
                       sundials_opts={'rtol': INTEGRATION_RTOL,
                                      'atol': INTEGRATION_ATOL})
    charge = VOLUME * ideal_mixing_density(mass_frac)  # [kg]
    liquid = reactor.Liquid_1
    assert liquid.mole_conc[2] > 0.1  # [mol/L], C formed, so not vacuous
    assert liquid.vol == pytest.approx(VOLUME, rel=RTOL)
    assert liquid.mass == pytest.approx(charge, rel=SOLVER_RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
def test_reacting_cstr_outlet_conserves_mass_flow():
    """A + B -> C conserves mass (MW_C = MW_A + MW_B), so must the CSTR.

    The total mass concentration obeys
    ``d(sum C MW)/dt = Q/V (sum C_in MW - sum C MW) + (sum nu MW) r`` with
    ``sum nu MW = 204.6941 - 105.52 - 99.1741 = 0``, and starts at the feed
    value, so the outlet mass flow equals the feed's at every time.
    """
    pytest.importorskip('assimulo')
    feed = np.array([0.5, 0.4, 0.0, 5.0])  # [mol/L], synthetic, non-closing
    rate_constant = 1.0e-2  # [L/mol/s], synthetic, converts most of B
    # [g/mol]; the JSON decimals cancel exactly, the bound absorbs round-off
    assert MW[2] - MW[0] - MW[1] == pytest.approx(0, abs=1e-12)
    reactor = _reactor('cstr', rate_constant, inlet_conc=feed)
    reactor.solve_unit(runtime=10 * RESIDENCE_TIME, verbose=False,
                       sundials_opts={'rtol': INTEGRATION_RTOL,
                                      'atol': INTEGRATION_ATOL})
    feed_mass_flow = VOLUME / RESIDENCE_TIME * (feed @ MW)  # [kg/s]
    assert reactor.Inlet.mass_flow == pytest.approx(feed_mass_flow, rel=RTOL)
    assert reactor.Outlet.mass_flow == pytest.approx(feed_mass_flow,
                                                     rel=SOLVER_RTOL)
    # The reaction formed C [mol/L], absent from the feed, so the check is
    # not vacuous.
    assert reactor.Outlet.mole_conc[2] > 0.1


@pytest.mark.integration
@pytest.mark.assimulo
def test_cstr_to_mixer_route_conserves_mass_flow():
    pytest.importorskip('assimulo')
    feed = np.array([0.5, 0.4, 0.0, 5.0])  # [mol/L], synthetic, non-closing
    sim = SimulationExec(THERMO_PATH, flowsheet='R01 --> M01')
    sim.R01 = _reactor('cstr', 1.0e-2, inlet_conc=feed)
    sim.M01 = Mixer()
    sim.SolveFlowsheet(kwargs_run={'R01': {
        'runtime': RESIDENCE_TIME, 'verbose': False,
        'sundials_opts': {'rtol': INTEGRATION_RTOL, 'atol': INTEGRATION_ATOL}}},
        verbose=False)
    vol_flow = VOLUME / RESIDENCE_TIME  # [m**3/s]
    expected = vol_flow * (sim.R01.result.mole_conc @ MW)  # [kg/s], per sample
    np.testing.assert_allclose(sim.M01.result.mass_flow, expected, rtol=RTOL, atol=0)
    np.testing.assert_allclose(expected, vol_flow * (feed @ MW), rtol=SOLVER_RTOL)


STATION_DIAMETER = 0.1  # [m], synthetic bench filter
CAKE_RESISTANCE = 1e11  # [m/kg], synthetic specific cake resistance
PRESSURE_DROP = 1e5  # [Pa]


@pytest.mark.integration
@pytest.mark.assimulo
def test_filter_keeps_a_concentration_charge():
    """Filter the fixture liquid with crystals of A without changing it.

    The charge holds ``FILTER_LIQUID_VOLUME * sum(C MW)`` [kg] of liquid.
    Filtrate plus remaining liquid must equal it at the start and at the
    end, and after complete filtration the cake liquid must fill the cake
    pores of the filter's packing, ``V_s eps / (1 - eps)`` [m**3] with
    ``V_s = kv mu_3`` (trapezoidal third moment, kv = 1) and ``eps`` the
    filter-wall-corrected porosity the Filter uses. Before #439 the filter
    converted volumes with the ideal-mixing density (897 kg/m**3) while the
    liquid held 390 kg/m**3, so it processed 2.30 times the charge.
    """
    pytest.importorskip('assimulo')
    from PharmaPy.SolidLiquidSep import Filter
    liquid = LiquidPhase(THERMO_PATH, temp=TEMP, vol=FILTER_LIQUID_VOLUME,
                         mole_conc=CONC.copy())
    solid = SolidPhase(THERMO_PATH, temp=TEMP, x_distrib=CRYSTAL_GRID.copy(),
                       distrib=CRYSTAL_POPULATION.copy(), kv=1.0,
                       mass_frac=[1, 0, 0, 0])  # kv [-], cube-like A crystals
    unit = Filter(station_diam=STATION_DIAMETER, alpha=CAKE_RESISTANCE)
    unit.Phases = [liquid, solid]
    charged = FILTER_LIQUID_VOLUME * MASS_DENSITY  # [kg]
    assert unit.Liquid_1.mass == pytest.approx(charged, rel=RTOL)
    _, states = unit.solve_unit(deltaP=PRESSURE_DROP, verbose=False)
    assert states[0].sum() == pytest.approx(charged, rel=RTOL)
    assert states[-1].sum() == pytest.approx(charged, rel=SOLVER_RTOL)
    integrand = CRYSTAL_POPULATION * CRYSTAL_GRID**3  # [#/um * um**3]
    solid_volume = np.sum(np.diff(CRYSTAL_GRID)
                          * (integrand[1:] + integrand[:-1]) / 2) * UM3_TO_M3  # [m**3]
    porosity = unit.Solid_1.getPorosity(diam_filter=STATION_DIAMETER)  # [-]
    pore_volume = solid_volume * porosity / (1 - porosity)  # [m**3]
    assert unit.Outlet.Liquid_1.vol == pytest.approx(pore_volume, rel=SOLVER_RTOL)
    assert unit.Outlet.Liquid_1.mass == pytest.approx(pore_volume * MASS_DENSITY,
                                                      rel=SOLVER_RTOL)
