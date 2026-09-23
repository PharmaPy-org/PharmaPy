"""#227 physical FVM mass generation is independent of numerical CSD scale.

Synthetic primary nucleation and growth have unit relative supersaturation;
the three-point seed grid has an independently integrated second moment.
"""

import numpy as np
import pytest

from test_crystallizer_parameter_evaluations import (
    make_unit, GROWTH, KV, LIQUID_VOLUME, RTOL)

pytestmark = pytest.mark.unit

NUCLEATION = 1e8  # [#/m**3/s], synthetic rate at unit relative supersaturation


@pytest.mark.parametrize('scale', [1e-9, 1e-6, 1e-3, 1.0])  # [-], representation sweep
@pytest.mark.parametrize('radius', [0.0, 10.0])  # [um], point/finite nuclei
def test_physical_mass_source_is_scale_invariant(data_path, scale, radius):
    """Check total mass and the public volume balance at fixed physical rates.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    scale : float
        Numerical distribution multiplier [-].
    radius : float
        Nucleation size [um].
    """
    unit, states = make_unit(data_path, scale=scale, radius=radius)
    unit.Kinetics.params['nucl_prim'] = [NUCLEATION, 0.0, 1.0]
    # [#/m**3/s], [J/mol], [-]; linear nucleation, no Arrhenius attenuation.
    # Trapezoidal integration on [10, 20, 40] um with [1, 2, 1]*1e7 #/um:
    # mu_2 = 2.85e11 um**2, mu_3 = 8.85e12 um**3.
    second_moment = 0.285  # [m**2], exact total seed moment
    third_moment = 8.85e-6  # [m**3], exact total seed moment
    volume = LIQUID_VOLUME + KV * third_moment  # [m**3], total slurry
    density = unit.Solid_1.getDensity()  # [kg/m**3]
    nucleus_volume = KV * (radius / 1e6)**3  # [m**3/#], exact um to m conversion
    expected = density * (3 * KV * GROWTH * 1e-6 * second_moment
                           + NUCLEATION * volume * nucleus_volume)  # [kg/s]
    concentrations = states[unit.num_distr:-1]  # [kg/m**3]
    derivative, mass = unit.fvm_method(
        states[:unit.num_distr], unit.Solid_1.moments, concentrations,
        unit.Liquid_1.temp, None, density, vol=volume)
    assert mass == pytest.approx(expected, rel=RTOL, abs=0)
    reference, reference_states = make_unit(data_path, scale=1.0, radius=radius)
    reference.Kinetics.params['nucl_prim'] = [NUCLEATION, 0.0, 1.0]
    # [#/m**3/s], [J/mol], [-]; identical physical conditions at unit scale
    reference_derivative, _ = reference.fvm_method(
        reference_states[:reference.num_distr], reference.Solid_1.moments,
        concentrations, reference.Liquid_1.temp, None, density, vol=volume)
    np.testing.assert_allclose(derivative / scale, reference_derivative,
                               rtol=RTOL, atol=0)
    rhs = unit.unit_model(0.0, states)  # [state unit/s]
    liquid_density = unit.Liquid_1.getDensity()  # [kg/m**3]
    assert -rhs[-1] * liquid_density == pytest.approx(expected, rel=RTOL, abs=0)


def test_finite_radius_mass_uses_cubic_length_conversion(data_path):
    """Use scale=1 to isolate the physical size conversion from CSD scaling.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    radius = 10.0  # [um], a nucleus has kv*1e-15 m**3 volume
    unit, states = make_unit(data_path, radius=radius)
    unit.Kinetics.params['growth'][0] = 0.0  # [um/s], isolate birth mass
    unit.Kinetics.params['nucl_prim'] = [NUCLEATION, 0.0, 1.0]
    # [#/m**3/s], [J/mol], [-]; unit relative force.
    volume = LIQUID_VOLUME + KV * 8.85e-6  # [m**3], analytic third moment above
    expected = (unit.Solid_1.getDensity() * NUCLEATION * volume
                * KV * 1e-15)  # [kg/s], (10 um)**3 = 1e-15 m**3
    rhs = unit.unit_model(0.0, states)  # [state unit/s]
    assert -rhs[-1] * unit.Liquid_1.getDensity() == pytest.approx(
        expected, rel=RTOL, abs=0)
