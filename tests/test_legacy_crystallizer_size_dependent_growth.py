"""Legacy crystallizers refuse size-dependent growth laws.

``CrystKinetics`` accepts a five-entry growth law ``[k_g, E_g, g, alpha,
beta]`` whose size factor ``(1 + beta*L)**alpha`` only the refactored
``OneDFVMMechanism`` applies. The legacy ``BatchCryst``, ``MSMPR`` and
``SemibatchCryst`` read ``k_g, E_g, g`` alone, so they must refuse such a law
at the ``Kinetics`` setter instead of running size-independent growth without
warning. The shared predicate ``has_size_dependent_growth`` is checked here
too. Nothing is solved and no Assimulo is needed.
"""

import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Kinetics import CrystKinetics


pytestmark = pytest.mark.unit


SOLUB = [50.0]  # [kg/m**3]
GROWTH = [1.0, 0.0, 1.0]  # k [um/s], E [J/mol], g [-]
GROWTH_SIZE_DEPENDENT = [1.0, 0.0, 1.0, 0.5, 2.0]  # ..., alpha [-], beta [1/um]
# Either entry at zero makes the size factor exactly one: (1 + 2 L)**0 and
# (1 + 0 L)**0.5, so these laws are size independent despite their length.
GROWTH_ZERO_ALPHA = [1.0, 0.0, 1.0, 0.0, 2.0]
GROWTH_ZERO_BETA = [1.0, 0.0, 1.0, 0.5, 0.0]
STANDARD_NAME_COUNT = 3 + 4 + 3 + 3  # prim, sec, growth, dissolution [-]
SIZE_DEPENDENT_NAME_COUNT = STANDARD_NAME_COUNT + 2  # plus alpha, beta [-]
TANK_VOLUME = 1.0  # [m**3], required by the continuous units, unused here

UNIT_FACTORIES = [
    lambda: BatchCryst('C', method='1D-FVM'),
    lambda: MSMPR('C', method='1D-FVM', vol_tank=TANK_VOLUME),
    lambda: SemibatchCryst('C', method='1D-FVM', vol_tank=TANK_VOLUME),
]
UNIT_IDS = ['BatchCryst', 'MSMPR', 'SemibatchCryst']


@pytest.mark.parametrize('make_unit', UNIT_FACTORIES, ids=UNIT_IDS)
@pytest.mark.parametrize('reformulate', [False, True])
def test_legacy_crystallizer_refuses_size_dependent_growth(make_unit,
                                                            reformulate):
    """A nonzero alpha and beta is refused and leaves the unit untouched."""
    unit = make_unit()
    kinetics = CrystKinetics(SOLUB, growth=GROWTH_SIZE_DEPENDENT,
                             reformulate_kin=reformulate)

    with pytest.raises(PharmaPyNonImplementedError,
                       match='size-dependent growth'):
        unit.Kinetics = kinetics

    assert unit.Kinetics is None
    assert unit.mask_params is None


@pytest.mark.parametrize('make_unit', UNIT_FACTORIES, ids=UNIT_IDS)
@pytest.mark.parametrize(
    'growth, expected_count',
    [(GROWTH, STANDARD_NAME_COUNT),
     (GROWTH_ZERO_ALPHA, SIZE_DEPENDENT_NAME_COUNT),
     (GROWTH_ZERO_BETA, SIZE_DEPENDENT_NAME_COUNT)],
    ids=['three_entry', 'zero_alpha', 'zero_beta'])
def test_legacy_crystallizer_accepts_size_independent_growth(make_unit,
                                                              growth,
                                                              expected_count):
    """Three-entry laws and unit size factors attach as before."""
    unit = make_unit()
    kinetics = CrystKinetics(SOLUB, growth=growth)

    unit.Kinetics = kinetics

    assert unit.Kinetics is kinetics
    assert len(unit.mask_params) == expected_count
    assert tuple(unit.name_params) == tuple(kinetics.name_params)


@pytest.mark.parametrize(
    'growth, expected',
    [(None, False), (GROWTH, False), (GROWTH_SIZE_DEPENDENT, True),
     (GROWTH_ZERO_ALPHA, False), (GROWTH_ZERO_BETA, False)],
    ids=['omitted', 'three_entry', 'size_dependent', 'zero_alpha',
         'zero_beta'])
@pytest.mark.parametrize('reformulate', [False, True])
def test_has_size_dependent_growth(growth, expected, reformulate):
    """The predicate is True only when alpha and beta are both nonzero."""
    kinetics = CrystKinetics(SOLUB, growth=growth, reformulate_kin=reformulate)

    assert kinetics.has_size_dependent_growth() is expected
