"""#157 seed inventory through the real collector and crystallizer setup.

Native CVode results expose the actual seed volume and distribution at the
initial time. The inlet has an asymmetric nonzero CSD and frozen kinetics.
"""

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Streams import LiquidStream, SolidStream

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
RTOL = 1e-12  # [-], float64 roundoff allowance for short inventory calculations


@pytest.mark.parametrize('kv', [0.5, 1.0])  # [-], non-unit shape and legacy case
@pytest.mark.parametrize('quantity', ['liquid_volume', 'solid_shape'])
def test_collector_seed_preserves_inlet_shape(data_path, kv, quantity):
    """Check seed inventory from the real collector's first reported state.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic fixture paths.
    kv : float
        Crystal volumetric shape factor [-].
    quantity : str
        Liquid volume or solid shape contract to check.
    """
    pytest.importorskip('assimulo')
    path = str(data_path['flowsheet'] / 'compound_database.json')
    grid = np.array([100.0, 200.0, 400.0])  # [um], asymmetric crystal bins
    distribution = np.array([1e8, 2e8, 1e8])  # [#/m**3/um], finite seed inventory
    temperature = 310.0  # [K], shared inlet phase temperature
    flow = 1e-4  # [m**3/s], synthetic feed for initialization
    liquid = LiquidStream(path, temp=temperature, vol_flow=flow,
                          mass_frac=[0.1, 0.1, 0.1, 0.1, 0.6])  # [-]
    solid = SolidStream(path, temp=temperature, kv=kv, x_distrib=grid,
                        distrib=distribution, mass_frac=[1, 0, 0, 0, 0])  # [-]
    inlet = SlurryStream(vol_flow=flow, x_distrib=grid, distrib=distribution)
    inlet.Phases = (liquid, solid)
    inlet.y_inlet = dict(mass_conc=liquid.mass_conc, vol_flow=flow,
                         temp=temperature, distrib=distribution)
    inlet.y_upstream = inlet.y_inlet
    inlet.time_upstream = None
    collector = DynamicCollector()
    collector.Inlet = inlet
    collector.KinCryst = CrystKinetics(coeff_solub=[2000.0])
    # [kg/m**3], above feed concentration; default kinetic rates are zero.
    collector.kwargs_cryst = dict(target_ind=0, target_comp='A')
    collector.solve_unit(runtime=0.01, verbose=False,
                         sundials_opts={'rtol': 1e-9, 'atol': 1e-10})
    # Short accumulation [s] and error limits [-, state units] match the
    # established collector moment-inventory regression's small seed.
    result = collector.result
    seed_solid = collector.CrystInst.Solid_1
    # Independent trapezoids of x**3*n(x), with exact cubic um -> m conversion.
    weighted = grid**3 * distribution  # [um**2/m**3]
    moment_three = np.sum(np.diff(grid) * (weighted[1:] + weighted[:-1]) / 2) * 1e-18
    # [m**3/m**3], volume-specific third moment
    seed_volume = result.distrib[0, 0] / distribution[0]  # [m**3]
    if quantity == 'liquid_volume':
        expected = seed_volume * (1 - kv * moment_three)  # [m**3]
        assert result.vol[0] == pytest.approx(expected, rel=RTOL, abs=0)
        if kv == 1:
            assert result.vol[0] == pytest.approx(
                seed_volume * (1 - inlet.moments[3]), rel=RTOL, abs=0)
    else:
        assert seed_solid.kv == kv
