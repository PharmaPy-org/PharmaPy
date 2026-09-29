# -*- coding: utf-8 -*-
"""Assimulo reactor tests whose imports also guard the lazy boundary.

Originally created by dcasasor on April 18, 2022.
"""

import unittest
import json
import importlib.util
from glob import glob
from pathlib import Path
from  numpy import genfromtxt, savetxt, allclose, vstack
import pytest

from PharmaPy.Reactors import PlugFlowReactor
from PharmaPy.Streams import LiquidStream
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Utilities import CoolingWater

HAS_ASSIMULO = importlib.util.find_spec("assimulo") is not None
pytestmark = [
    pytest.mark.assimulo,
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_ASSIMULO,
        reason="assimulo is not installed; solver-backed integration tests skipped",
    ),
]


class PlugFlowReactorTests(unittest.TestCase):
    """ Class containing tests for the PlugFlowReactor class in PharmaPy
    """

    def setUp(self):
        # Data
        data_dir = Path(__file__).resolve().parent / 'data'
        datapath = str(data_dir / 'pfr_test_pure_comp.json')

        with open(data_dir / 'pfr_test_constructor_kwargs.json') as f:
            data_objects = json.load(f)

        tau = data_objects['inlet'].pop('tau')
        data_objects['inlet']['vol_flow'] = data_objects['phase']['vol'] / tau

        data_objects['kinetics']['k_params'] *= 1/60
        data_objects['kinetics']['path'] = datapath

        time_integration = genfromtxt(data_dir / 'pfr_test_expected_times.csv',
                                      delimiter=',')

        data_objects['solve_unit']['time_grid'] = time_integration

        # PharmaPy objects
        inlet = LiquidStream(datapath, **data_objects['inlet'])
        phase = LiquidPhase(datapath, **data_objects['phase'])

        kinetics = RxnKinetics(**data_objects['kinetics'])
        utility = CoolingWater(**data_objects['utility'])

        self.m = reactor = PlugFlowReactor(**data_objects['reactor'])

        reactor.Inlet = inlet
        reactor.Phases = phase
        reactor.Kinetics = kinetics
        reactor.Utility = utility

        reactor.solve_unit(**data_objects['solve_unit'])

    def test_mole_conc(self):
        data_dir = Path(__file__).resolve().parent / 'data'
        filenames = glob(str(data_dir / 'pfr_test_expected_conc*'))
        filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        flags = []
        molefracs = self.m.result.mole_conc
        zipped = list(zip(*[ar.T for ar in molefracs.values()]))

        for ind, name in enumerate(filenames):
            expected_conc = genfromtxt(name, delimiter=',')

            # I'm excluding the solvent in this comparison

            abstol = 1e-4  # TODO: is this determined by the tolerances of assimulo?

            per_vol_elem = vstack(zipped[ind]).T[:, :-1]
            flag = allclose(expected_conc, per_vol_elem, atol=abstol)

            flags.append(flag)

        # TODO: maybe print largest deviation

        self.assertTrue(all(flags))

    def test_temperature(self):
        pass


if __name__ == '__main__':
    unittest.main()
