import pytest
from fixture_factory import market_fixture
from etf_option.pipeline import analyze
from etf_option.cleaning import Filters


@pytest.fixture(scope="session")
def raw():
    return market_fixture()


@pytest.fixture(scope="session")
def analyzed(raw):
    return analyze(raw, Filters())

