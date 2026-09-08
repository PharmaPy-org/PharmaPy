"""#113 jacket transients on real phase heat capacities and cooling water.

The 0.14 default is the existing design assumption, not calibrated data.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Utilities import CoolingWater
from test_mixedphase_thermo_contracts import make_slurry

pytestmark = pytest.mark.unit

TEMPERATURE = 310.0  # [K], tank temperature
JACKET_TEMPERATURE = 300.0  # [K], intermediate between tank and utility inlet
JACKET_VOLUME = 2e-4  # [m**3], synthetic explicit jacket inventory
RTOL = 1e-12  # [-], roundoff tolerance for direct algebra


@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR, SemibatchCryst])
@pytest.mark.parametrize('volume_factor', [0.5, 1.0, 2.0])  # [-], half to twice design volume
def test_jacket_volume_controls_only_transient(data_path, unit_type, volume_factor):
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
    legacy_volume = slurry.vol * 0.14  # [m**3], exact historical design ratio
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
