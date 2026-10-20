"""Exact pathway-LP scaling regressions; real HiGHS/SLSQP, no reactor fixtures.

Positive objective rescaling cannot change the optimum. Flux coordinate changes
must preserve physical sources under both existing secondary-selection policies.
"""

import numpy as np
import pytest

from PharmaPy.Metabolic.pathways import ConfiguredPathwayModel, PathwayModelDefinition

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('coefficient', [1e-12, 1e-10, 1., 1e12])
@pytest.mark.parametrize('coordinate_scale', [1e-3, 1., 1e3])
@pytest.mark.parametrize('minimum_norm', [False, True])
def test_primary_optimum_survives_secondary_scaling(coefficient, coordinate_scale, minimum_norm):
    """Maximize c*v/s subject to 0 <= v/s <= 1; the physical optimum is one.

    Parameters
    ----------
    coefficient, coordinate_scale : float
        Dimensionless objective and coordinate conversions [-], spanning
        absolute solver tolerance and large-coefficient regimes.
    minimum_norm : bool
        Select the existing scaled minimum-norm policy instead of lexicographic.
    """
    definition = PathwayModelDefinition(
        state_ids=('substrate',), pathway_ids=('path',),
        exchange_matrix=[[-1./coordinate_scale]],  # [source/flux]
        growth_coefficients=[1./coordinate_scale],  # [growth/flux]
        objective_coefficients=[coefficient/coordinate_scale],  # [objective/flux]
        parameters={'capacity': 1.},  # [source/time]; exact unit-capacity fixture
        uptake_constraints=({'identifier': 'uptake', 'type': 'constant',
            'species': 'substrate', 'maximum_parameter': 'capacity'},),
        secondary_optimization=({'policy': 'scaled-minimum-norm',
            'reference_scales': {'path': coordinate_scale}} if minimum_norm else None))
    result = ConfiguredPathwayModel(definition).solve_fluxes([1.], biomass=1.)
    # Match the existing 1e-8 audit budget, without an absolute objective floor.
    np.testing.assert_allclose(result.fluxes/coordinate_scale, [1.], rtol=1e-8, atol=0.)
    assert result.primary_objective/coefficient == pytest.approx(1., abs=1e-8)


@pytest.mark.parametrize('coefficient', [0., -1.])
def test_zero_and_negative_objectives_keep_their_meaning(coefficient):
    """The existing lexicographic policy selects zero for these objectives.

    Parameters
    ----------
    coefficient : float
        Objective coefficient [-]; zero or negative on a nonnegative pathway.
    """
    definition = PathwayModelDefinition(
        ('substrate',), ('path',), [[-1.]], [1.], [coefficient],
        parameters={'capacity': 1.},  # [source/time]
        uptake_constraints=({'identifier': 'uptake', 'type': 'constant',
            'species': 'substrate', 'maximum_parameter': 'capacity'},))
    result = ConfiguredPathwayModel(definition).solve_fluxes([1.], biomass=1.)
    np.testing.assert_allclose(result.fluxes, [0.], atol=1e-10)


@pytest.mark.parametrize('multiplier', [1e-12, 1., 1e12])
def test_unequal_coordinates_preserve_all_physical_objective_terms(multiplier):
    """Independent capacities imply fluxes 0.5/1e10 and 1, with growth 1.5.

    Parameters
    ----------
    multiplier : float
        Common positive objective conversion [-]; not a biological adjustment.
    """
    conversion = 1e10  # [-]; coordinate change with unequal pathway scales
    definition = PathwayModelDefinition(
        ('A', 'B'), ('one', 'two'), np.diag([-conversion, -1.]),
        np.array([conversion, 1.]), multiplier*np.array([conversion, 1.]),
        parameters={'capA': .5, 'capB': 1.},  # [source/time]
        uptake_constraints=tuple({'identifier': name, 'type': 'constant', 'species': name,
            'maximum_parameter': 'cap'+name} for name in ('A', 'B')))
    result = ConfiguredPathwayModel(definition).solve_fluxes([1., 1.], biomass=1.)
    np.testing.assert_allclose(result.extracellular_rates, [-.5, -1.], atol=1e-8)
    assert result.primary_objective/multiplier == pytest.approx(1.5, abs=1e-8)
