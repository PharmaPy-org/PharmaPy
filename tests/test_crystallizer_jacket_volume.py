"""#113 jacket transients on real phase heat capacities and cooling water.

The 0.14 default is the existing design assumption, not calibrated data.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Utilities import CoolingWater
from test_mixedphase_thermo_contracts import make_slurry
from test_crystallizer_parameter_evaluations import InitializationCaptured, make_unit

pytestmark = pytest.mark.unit

TEMPERATURE = 310.0  # [K], tank temperature
JACKET_TEMPERATURE = 300.0  # [K], intermediate between tank and utility inlet
JACKET_VOLUME = 2e-4  # [m**3], synthetic explicit jacket inventory
RTOL = 1e-12  # [-], roundoff tolerance for direct algebra


@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR, SemibatchCryst])
@pytest.mark.parametrize('volume_factor', [0.5, 1.0, 2.0])  # [-], half to twice design volume
def test_jacket_volume_controls_only_transient(data_path, unit_type, volume_factor):
    """Compare explicit inventory against the established default volume basis.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    unit_type : type
        Crystallizer model.
    volume_factor : float
        Multiplier of the explicit jacket inventory [-].
    """
    slurry = make_slurry(str(data_path['flowsheet'] / 'compound_database.json'))
    volume = JACKET_VOLUME * volume_factor  # [m**3]
    unit = unit_type('A', method='moments', vol_ht=volume)
    unit.Phases = slurry
    unit.Utility = CoolingWater(vol_flow=1e-5, temp_in=290.0)  # [m**3/s], [K]
    unit.vol_tank = slurry.vol  # [m**3]
    unit.diam_tank = 0.1  # [m], synthetic cylindrical tank
    unit.area_base = np.pi * unit.diam_tank**2 / 4  # [m**2]
    densities = slurry.getDensity()  # [kg/m**3]
    kwargs = dict(time=0.0, params=None, cryst_rate=0.0,
                  u_inputs={'Inlet': {'vol_flow': 0.0}}, rhos=densities,
                  mu_n=slurry.Solid_1.moments, distrib=None, mass_conc=None,
                  temp=TEMPERATURE, temp_ht=JACKET_TEMPERATURE,
                  vol=slurry.Liquid_1.vol)
    if unit_type is not BatchCryst:
        kwargs['rhos'] = [densities, densities]
        kwargs['h_in'] = slurry.getEnthalpy(TEMPERATURE)  # [J/m**3]
    if unit_type is MSMPR:
        kwargs['mu_n'] = slurry.moments  # [m**n/m**3]
        kwargs['vol'] = slurry.vol  # [m**3]
    actual = unit.energy_balances(**kwargs)  # [K/s], tank then jacket
    # Derive the jacket balance independently: advective and transferred power
    # divided by the water inventory's heat capacity.
    wetted_volume = (slurry.Liquid_1.vol if unit_type is SemibatchCryst
                      else slurry.vol)  # [m**3], existing area convention
    area = 4 * wetted_volume / unit.diam_tank + unit.area_base  # [m**2]
    water = unit.Utility
    power = (water.vol_flow * water.rho * water.cp
             * (water.temp_in - JACKET_TEMPERATURE)
             + unit.u_ht * area * (TEMPERATURE - JACKET_TEMPERATURE))  # [W]
    expected = power / (water.rho * water.cp * volume)  # [K/s]
    assert actual[1] == pytest.approx(expected, rel=RTOL, abs=0)
    unit.vol_ht = None
    default = unit.energy_balances(**kwargs)  # [K/s]
    legacy_volume = slurry.vol * 0.14  # [m**3], historical Batch slurry basis
    if unit_type is not BatchCryst:
        legacy_volume /= unit.vol_offset  # [m**3], MSMPR/Semibatch vessel basis
    legacy_rate = (water.vol_flow / legacy_volume
                   * (water.temp_in - JACKET_TEMPERATURE)
                   - unit.u_ht * area * (JACKET_TEMPERATURE - TEMPERATURE)
                   / water.rho / legacy_volume / water.cp)  # [K/s]
    # [-], near-machine precision verifies reduction to the historical value.
    assert default[1] == pytest.approx(legacy_rate, rel=1e-15, abs=0)
    # Jacket inventory does not enter the tank RHS.
    assert actual[0] == pytest.approx(default[0], rel=1e-15, abs=0)



@pytest.mark.parametrize('volume', [0.0, -JACKET_VOLUME, np.nan, np.inf])  # [m**3]
def test_constructor_rejects_invalid_jacket_volume(volume):
    with pytest.raises(ValueError, match=r'vol_ht.*finite.*positive.*m\*\*3'):
        BatchCryst('A', vol_ht=volume)


def test_constructor_keeps_unspecified_jacket_volume():
    assert BatchCryst('A', vol_ht=None).vol_ht is None


@pytest.mark.parametrize('unit_type', [MSMPR, SemibatchCryst])
def test_default_jacket_inventory_repeats_across_initializations(
        data_path, monkeypatch, unit_type):
    """Infer jacket inventory from its balance after two real initializations.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Stop at optional solver construction after geometry initialization.
    unit_type : type
        MSMPR or SemibatchCryst.
    """
    from PharmaPy.Kinetics import CrystKinetics

    source, _ = make_unit(data_path)
    slurry = source.Slurry
    unit = unit_type('A', method='1D-FVM', vol_tank=slurry.vol)
    unit.Phases = slurry
    unit.Kinetics = CrystKinetics()
    unit.Utility = CoolingWater(vol_flow=1e-5, temp_in=290.0)  # [m**3/s], [K]
    captures = []

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Stop after geometry is initialized and before the solver is built.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver options.
        states_init : ndarray
            Initial state vector in model units.
        params_mergd : ndarray
            Active vector in native kinetic units.

        Raises
        ------
        InitializationCaptured
            Always, after recording working volume [m**3].
        """
        captures.append(unit.vol_tank)
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    inventories = []  # [m**3], jacket inventories inferred from the balance
    for _ in range(2):
        with pytest.raises(InitializationCaptured):
            unit.solve_unit(runtime=1.0, verbose=False)  # [s], initialization only
        densities = slurry.getDensity()  # [kg/m**3], liquid/solid
        volume = slurry.vol if unit_type is MSMPR else slurry.Liquid_1.vol  # [m**3]
        moments = slurry.moments if unit_type is MSMPR else slurry.Solid_1.moments
        # [m**n/m**3] MSMPR; [m**n] total Semibatch
        _, jacket_rate = unit.energy_balances(
            time=0.0, params=None, cryst_rate=0.0,
            u_inputs={'Inlet': {'vol_flow': 0.0}}, rhos=[densities, densities],
            mu_n=moments, distrib=None, mass_conc=None, temp=TEMPERATURE,
            temp_ht=JACKET_TEMPERATURE, vol=volume,
            h_in=slurry.getEnthalpy(TEMPERATURE))
        # [K/s]; invert C_j*dT_j/dt = inlet heat + vessel heat independently.
        area = 4 * volume / unit.diam_tank + unit.area_base  # [m**2]
        water = unit.Utility
        power = (water.vol_flow * water.rho * water.cp
                 * (water.temp_in - JACKET_TEMPERATURE)
                 + unit.u_ht * area * (TEMPERATURE - JACKET_TEMPERATURE))  # [W]
        inventories.append(power / (water.rho * water.cp * jacket_rate))
    expected = 0.14 * slurry.vol / unit.vol_offset  # [m**3], historical vessel basis
    np.testing.assert_allclose(captures, [slurry.vol, slurry.vol], rtol=RTOL, atol=0)
    np.testing.assert_allclose(inventories, [expected, expected], rtol=RTOL, atol=0)
