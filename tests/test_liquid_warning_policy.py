"""Real liquid constructors preserve caller policy and downstream warnings.

The four-species fixture uses equal mass fractions. Crystallization uses a
synthetic constant solubility solely to exercise its public deprecation notice;
no solver or crystallization rate evaluation is required.
"""

import warnings

import pytest

from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit
ZERO_AMOUNT_MESSAGE = "'mass', 'moles' and 'vol' are all set to zero."
EQUAL_MASS_FRACTIONS = [0.25] * 4  # [-], equal shares of four species
CONSTANT_SOLUBILITY = [1.0]  # [kg/m**3], synthetic constant polynomial


@pytest.fixture(params=[LiquidPhase, LiquidStream])
def liquid_constructor(request, data_path):
    """Return a real zero-amount liquid constructor with valid composition.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Selects the phase or inherited stream constructor.
    data_path : dict
        Paths to the repository's thermophysical fixtures.

    Returns
    -------
    callable
        Constructor accepting keyword overrides; defaults to zero amount
        [kg, m**3, mol] or zero flow [kg/s, m**3/s, mol/s].
    """
    from functools import partial

    return partial(
        request.param,
        str(data_path["integration"] / "pfr_test_pure_comp.json"),
        mass_frac=EQUAL_MASS_FRACTIONS,
    )


def test_zero_amount_preserves_downstream_warning_visibility(liquid_constructor):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        original_filters = warnings.filters.copy()
        liquid_constructor()
        warnings.warn("application probe", UserWarning)
        warnings.warn("numerical probe", RuntimeWarning)
        CrystKinetics(coeff_solub=CONSTANT_SOLUBILITY, sup_sat_type="ratio")

        assert [item.category for item in seen] == [
            RuntimeWarning, UserWarning, RuntimeWarning, FutureWarning,
        ]
        assert str(seen[0].message).startswith(ZERO_AMOUNT_MESSAGE)
        assert str(seen[1].message) == "application probe"
        assert str(seen[2].message) == "numerical probe"
        assert "it now means S - 1" in str(seen[3].message)
        assert warnings.filters == original_filters


def test_zero_amount_respects_caller_ignore_policy(liquid_constructor):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", RuntimeWarning)
        original_filters = warnings.filters.copy()
        liquid_constructor()
        warnings.warn("application probe", UserWarning)

        assert [(item.category, str(item.message)) for item in seen] == [
            (UserWarning, "application probe"),
        ]
        assert warnings.filters == original_filters


def test_zero_amount_respects_caller_error_policy(liquid_constructor):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        original_filters = warnings.filters.copy()
        with pytest.raises(RuntimeWarning, match="all set to zero"):
            liquid_constructor()
        assert warnings.filters == original_filters


def test_disabled_input_check_preserves_warning_policy(liquid_constructor):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("always", UserWarning)
        original_filters = warnings.filters.copy()
        liquid_constructor(check_input=False)
        warnings.warn("application probe", UserWarning)

        assert [(item.category, str(item.message)) for item in seen] == [
            (UserWarning, "application probe"),
        ]
        assert warnings.filters == original_filters
