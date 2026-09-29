import pytest


pytestmark = pytest.mark.unit


def test_data_path_fixture_resolves_existing_test_data(data_path):
    assert (data_path["integration"] / "pfr_test_pure_comp.json").is_file()
    assert (data_path["flowsheet"] / "compound_database.json").is_file()
