"""Flat parameter-vector round trips for CrystKinetics.

``concat_params`` flattens the four parameter sets and ``set_params`` takes
that vector back; the crystallizers hand fitted parameters to the kinetics
this way. The vector must come back as the same sets whatever their lengths,
including the five-entry size-dependent growth law, and a vector of another
layout must be refused rather than silently reassigned.

Fixture: normalized parameters with zero activation energies, evaluated at
the reference temperature, so each built-in rate reduces to
``k * |S|**n`` with ``S = (c - c_sat) / c_sat`` [-]. Units: solubility and
concentration [kg/m**3], temperature [K], growth and dissolution [um/s],
nucleation [#/m**3/s].
"""

import numpy as np
import pytest

from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Kinetics import CrystKinetics


pytestmark = pytest.mark.unit


SOLUB = [50.0]  # [kg/m**3], constant solubility
NUCL_PRIM = [1.0, 0.0, 1.0]  # k [#/m**3/s], E [J/mol], b [-]
NUCL_SEC = [1.0, 0.0, 1.0, 1.0]  # k [#/m**3/s], E [J/mol], s_1 [-], s_2 [-]
GROWTH = [1.0, 0.0, 1.0]  # k [um/s], E [J/mol], g [-]
GROWTH_SIZE_DEPENDENT = [1.0, 0.0, 1.0, 0.5, 2.0]  # ..., alpha [-], beta [1/um]
DISSOLUTION = [3.0, 0.0, 1.0]  # k [um/s], E [J/mol], d [-]

TEMP = 298.15  # [K], the default reference temperature
DISSOLVING_CONC = 25.0  # [kg/m**3], S = -0.5
GROWING_CONC = 75.0  # [kg/m**3], S = +0.5
SHAPE_FACTOR = 1.0  # [-]
NO_CRYSTALS = np.zeros(4)  # [m**k/m**3], keeps secondary nucleation at zero


def _kinetics(growth, reformulate):
    """Build kinetics with every mechanism active.

    Parameters
    ----------
    growth : list of float
        Growth parameters, the three-entry or five-entry form.
    reformulate : bool
        Whether k and E are stored as the logarithmic phi parameters.

    Returns
    -------
    CrystKinetics
        Kinetics targeting the single species of a scalar concentration.
    """
    kinetics = CrystKinetics(
        coeff_solub=SOLUB,
        nucl_prim=NUCL_PRIM,
        nucl_sec=NUCL_SEC,
        growth=growth,
        dissolution=DISSOLUTION,
        reformulate_kin=reformulate,
    )
    kinetics.target_idx = 0
    return kinetics


def _rates(kinetics, conc):
    """Evaluate (nucleation, growth, dissolution) at one concentration.

    Parameters
    ----------
    kinetics : CrystKinetics
        Kinetics under test.
    conc : float
        Target concentration [kg/m**3].

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)``: nucleation [#/m**3/s], growth [um/s], dissolution
        [um/s].
    """
    return np.array(kinetics.get_kinetics(conc, TEMP, SHAPE_FACTOR, NO_CRYSTALS),
                    dtype=float)


def _power_law(params, conc):
    """Evaluate ``k * |S|**n`` for one mechanism, independently of the class.

    Parameters
    ----------
    params : list of float
        ``[k, E, n, ...]`` with E = 0 so the Arrhenius factor is unity.
    conc : float
        Target concentration [kg/m**3].

    Returns
    -------
    float
        Rate magnitude in the mechanism's units.
    """
    supersat = (conc - SOLUB[0]) / SOLUB[0]  # [-]
    return params[0] * abs(supersat) ** params[2]


@pytest.mark.parametrize("reformulate", [False, True], ids=["physical", "reformulated"])
@pytest.mark.parametrize(
    "growth", [GROWTH, GROWTH_SIZE_DEPENDENT], ids=["standard", "size_dependent"]
)
def test_identity_update_keeps_every_parameter_set(growth, reformulate):
    """Passing the object's own flat vector back leaves each set unchanged.

    The flat vector used to be split with a fixed [3, 4, 3, 3] layout, so a
    five-entry growth law lost alpha and beta to the dissolution set, which
    grew to five entries.
    """
    kinetics = _kinetics(growth, reformulate)
    before = {name: np.array(values, dtype=float)
              for name, values in kinetics.params.items()}

    kinetics.set_params(kinetics.concat_params())

    assert list(kinetics.params) == list(before)
    for name, values in before.items():
        assert len(kinetics.params[name]) == len(values)
        np.testing.assert_array_equal(kinetics.params[name], values)


@pytest.mark.parametrize("reformulate", [False, True], ids=["physical", "reformulated"])
def test_identity_update_keeps_the_rates(reformulate):
    """The rates survive the round trip, in both driving-force regimes.

    With the corrupted split the dissolution rate at ``DISSOLVING_CONC``
    moved from -1.5 um/s to about -0.062 um/s, because the dissolution set
    received growth's alpha and beta as its prefactor and energy.
    """
    kinetics = _kinetics(GROWTH_SIZE_DEPENDENT, reformulate)
    dissolving_before = _rates(kinetics, DISSOLVING_CONC)
    growing_before = _rates(kinetics, GROWING_CONC)

    kinetics.set_params(kinetics.concat_params())

    # Dissolution is reported negative; growth and primary nucleation positive.
    assert dissolving_before[2] == pytest.approx(
        -_power_law(DISSOLUTION, DISSOLVING_CONC))
    assert growing_before[1] == pytest.approx(
        _power_law(GROWTH_SIZE_DEPENDENT, GROWING_CONC))
    np.testing.assert_allclose(_rates(kinetics, DISSOLVING_CONC), dissolving_before)
    np.testing.assert_allclose(_rates(kinetics, GROWING_CONC), growing_before)


@pytest.mark.parametrize(
    "growth, other_growth",
    [(GROWTH, GROWTH_SIZE_DEPENDENT), (GROWTH_SIZE_DEPENDENT, GROWTH)],
    ids=["five_onto_three", "three_onto_five"],
)
def test_vector_of_another_layout_is_refused(growth, other_growth):
    """A vector for the other growth layout is rejected, not reassigned.

    The layout comes from the stored sets, so the only vectors that fit are
    the ones ``concat_params`` produces for that object.
    """
    kinetics = _kinetics(growth, reformulate=False)
    foreign = _kinetics(other_growth, reformulate=False).concat_params()
    expected_length = len(kinetics.concat_params())

    with pytest.raises(PharmaPyValueError, match=f"{expected_length} entries"):
        kinetics.set_params(foreign)

    assert len(kinetics.params["growth"]) == len(growth)
