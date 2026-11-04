"""Legacy crystallizers refuse growth laws longer than three entries.

``CrystKinetics`` accepts a five-entry growth law ``[k_g, E_g, g, alpha,
beta]`` whose size factor ``(1 + beta*L)**alpha`` only the refactored
``OneDFVMMechanism`` applies. The legacy ``BatchCryst``, ``MSMPR`` and
``SemibatchCryst`` evaluate ``k_g, E_g, g`` alone and ``deriv_cryst`` returns
three growth partials, so a longer law, even one whose size factor is one,
would list alpha and beta as estimable in ``name_params`` and ``mask_params``
while the analytic ``jac_params`` builds 13 columns under a 15-entry mask.
They refuse such a law at the ``Kinetics`` setter. The predicate
``has_size_dependent_growth`` that the moment closure uses is checked here
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
# (1 + 0 L)**0.5. The legacy units still refuse these laws, which carry two
# parameters they never apply; only the predicate treats them as size
# independent.
GROWTH_ZERO_ALPHA = [1.0, 0.0, 1.0, 0.0, 2.0]
GROWTH_ZERO_BETA = [1.0, 0.0, 1.0, 0.5, 0.0]
STANDARD_NAME_COUNT = 3 + 4 + 3 + 3  # prim, sec, growth, dissolution [-]
TANK_VOLUME = 1.0  # [m**3], required by the continuous units, unused here

UNIT_FACTORIES = [
    lambda: BatchCryst('C', method='1D-FVM'),
    lambda: MSMPR('C', method='1D-FVM', vol_tank=TANK_VOLUME),
    lambda: SemibatchCryst('C', method='1D-FVM', vol_tank=TANK_VOLUME),
]
UNIT_IDS = ['BatchCryst', 'MSMPR', 'SemibatchCryst']
FIVE_ENTRY_LAWS = [GROWTH_SIZE_DEPENDENT, GROWTH_ZERO_ALPHA, GROWTH_ZERO_BETA]
FIVE_ENTRY_IDS = ['size_dependent', 'zero_alpha', 'zero_beta']


@pytest.mark.parametrize('make_unit', UNIT_FACTORIES, ids=UNIT_IDS)
@pytest.mark.parametrize('growth', FIVE_ENTRY_LAWS, ids=FIVE_ENTRY_IDS)
@pytest.mark.parametrize('reformulate', [False, True])
def test_legacy_crystallizer_refuses_growth_beyond_three_entries(
        make_unit, growth, reformulate):
    """A five-entry law is refused, unit size factor or not, untouched."""
    unit = make_unit()
    kinetics = CrystKinetics(SOLUB, growth=growth,
                             reformulate_kin=reformulate)

    with pytest.raises(PharmaPyNonImplementedError,
                       match='growth law of 3 entries'):
        unit.Kinetics = kinetics

    assert unit.Kinetics is None
    assert unit.mask_params is None


@pytest.mark.parametrize('make_unit', UNIT_FACTORIES, ids=UNIT_IDS)
def test_legacy_crystallizer_accepts_three_entry_growth(make_unit):
    """A three-entry law attaches with one mask entry per parameter."""
    unit = make_unit()
    kinetics = CrystKinetics(SOLUB, growth=GROWTH)

    unit.Kinetics = kinetics

    assert unit.Kinetics is kinetics
    assert len(unit.mask_params) == STANDARD_NAME_COUNT
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
