"""#157 seed inventory through the real collector and crystallizer setup.

Only the ODE creation boundary is intercepted; no integration is needed to
verify the physical seed phases. The inlet has an asymmetric nonzero CSD.
"""

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers import SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Streams import LiquidStream, SolidStream

pytestmark = pytest.mark.unit
RTOL = 1e-12  # [-], float64 roundoff allowance for short inventory calculations


class SeedCaptured(Exception):
    """Stop at ODE creation after real seed-phase initialization."""


@pytest.mark.parametrize('kv', [0.5, 1.0])  # [-], non-unit shape and legacy case
@pytest.mark.parametrize('quantity', ['liquid_volume', 'solid_shape'])
def test_collector_seed_preserves_inlet_shape(data_path, monkeypatch, kv, quantity):
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
    collector.KinCryst = CrystKinetics()
    collector.kwargs_cryst = dict(target_ind=0, target_comp='A')
    captured = []

    def capture_problem(self, eval_sens, states_init, params, jac_v_prod):
        """Capture actual initialized phases immediately before the solver.

        Parameters
        ----------
        self : SemibatchCryst
            Real delegated crystallizer.
        eval_sens, jac_v_prod : bool
            Solver options.
        states_init : ndarray
            CSD [#/um], concentrations [kg/m**3], volume [m**3], temperature [K].
        params : ndarray
            Native kinetic parameters.

        Raises
        ------
        SeedCaptured
            Always, after saving the real phases.
        """
        captured.append((self.Liquid_1, self.Solid_1))
        raise SeedCaptured

    monkeypatch.setattr(SemibatchCryst, 'set_ode_problem', capture_problem)
    with pytest.raises(SeedCaptured):
        collector.solve_unit(runtime=1.0, verbose=False)  # [s], setup only
    assert len(captured) == 1
    seed_liquid, seed_solid = captured[0]
    # Independent trapezoids of x**3*n(x), with exact cubic um -> m conversion.
    weighted = grid**3 * distribution  # [um**2/m**3]
    moment_three = np.sum(np.diff(grid) * (weighted[1:] + weighted[:-1]) / 2) * 1e-18
    # [m**3/m**3], volume-specific third moment
    seed_volume = seed_solid.distrib[0] / distribution[0]  # [m**3]
    if quantity == 'liquid_volume':
        expected = seed_volume * (1 - kv * moment_three)  # [m**3]
        assert seed_liquid.vol == pytest.approx(expected, rel=RTOL, abs=0)
        if kv == 1:
            assert seed_liquid.vol == seed_volume * (1 - inlet.moments[3])
    else:
        assert seed_solid.kv == kv
