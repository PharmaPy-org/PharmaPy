"""Parameter-name table regressions for CrystKinetics."""

import pytest

from PharmaPy.Kinetics import CrystKinetics


pytestmark = pytest.mark.unit


# Normalized fixture parameters. Magnitudes are irrelevant here: only the shape
# of the parameter-name table is under test, and no rate is evaluated.
SOLUB = [2000.0]  # [kg/m**3]
NUCL_PRIM = [1.0, 0.0, 1.0]  # k [#/m**3/s], E [J/mol], b [-]
NUCL_SEC = [1.0, 0.0, 1.0, 1.0]  # k [#/m**3/s], E [J/mol], s_1 [-], s_2 [-]
GROWTH = [1.0, 0.0, 1.0]  # k [um/s], E [J/mol], g [-]
# Size-dependent growth carries two further entries, read by the finite-volume
# mechanism as params['growth'][3] and [4].
GROWTH_SIZE_DEPENDENT = [1.0, 0.0, 1.0, 0.5, 2.0]  # ..., alpha [-], beta [1/um]
DISSOLUTION = [1.0, 0.0, 1.0]  # k [um/s], E [J/mol], d [-]

STANDARD_NAME_COUNT = 3 + 4 + 3 + 3  # prim, sec, growth, dissolution [-]
SIZE_DEPENDENT_NAME_COUNT = STANDARD_NAME_COUNT + 2  # plus alpha and beta [-]


def _kinetics(growth, reformulate):
    """Build kinetics differing only in the growth term and the formulation.

    Parameters
    ----------
    growth : list of float
        Growth parameters, either the three-entry form or the five-entry
        size-dependent form.
    reformulate : bool
        Whether to transform k and E into the logarithmic phi parameters.

    Returns
    -------
    CrystKinetics
        Kinetics object whose name table is under test.
    """
    return CrystKinetics(
        coeff_solub=SOLUB,
        nucl_prim=NUCL_PRIM,
        nucl_sec=NUCL_SEC,
        growth=growth,
        dissolution=DISSOLUTION,
        reformulate_kin=reformulate,
    )


@pytest.mark.parametrize("reformulate", [False, True])
def test_kinetics_without_a_growth_term_can_be_built(reformulate):
    """Omitting a mechanism must still produce the standard name table.

    Parameters are optional, so a solubility-only object is valid and is how
    callers describe a system whose rates are supplied by custom mechanisms.
    The name table describes all four mechanisms regardless of which were
    given, so it keeps its standard length.
    """
    kinetics = CrystKinetics(coeff_solub=SOLUB, reformulate_kin=reformulate)

    assert kinetics.num_params == STANDARD_NAME_COUNT
    assert len(kinetics.name_params) == kinetics.num_params


@pytest.mark.parametrize("reformulate", [False, True])
def test_size_dependent_growth_names_alpha_and_beta_separately(reformulate):
    """The two size-dependent growth parameters are named individually.

    Each entry of the table names exactly one parameter, because the
    crystallizer builds its estimation mask as one flag per name. Two names
    fused into a single entry would leave the mask one flag short.
    """
    kinetics = _kinetics(GROWTH_SIZE_DEPENDENT, reformulate)

    assert kinetics.num_params == SIZE_DEPENDENT_NAME_COUNT
    assert "alpha" in kinetics.name_params
    assert "beta" in kinetics.name_params


@pytest.mark.parametrize(
    "growth", [GROWTH, GROWTH_SIZE_DEPENDENT], ids=["standard", "size_dependent"]
)
def test_both_formulations_name_the_same_parameter_count(growth):
    """Reformulating changes parameter meanings, not how many there are.

    The logarithmic formulation replaces k and E with phi_1 and phi_2, a
    one-for-one substitution, so both tables describe the same parameter
    vector and must agree in length.
    """
    physical = _kinetics(growth, reformulate=False)
    reformulated = _kinetics(growth, reformulate=True)

    assert reformulated.num_params == physical.num_params
