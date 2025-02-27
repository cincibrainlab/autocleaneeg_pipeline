"""Test configuration and fixtures."""

import pytest

from tests.utils import get_test_file

@pytest.fixture
def test_raw_file():
    """Fixture to provide test raw file."""
    return get_test_file("0199_rest.raw") 