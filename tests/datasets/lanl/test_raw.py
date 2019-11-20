from unittest.mock import patch

import pytest

from eai_graph_tools.datasets.lanl import DatasetLANL
from eai_graph_tools.datasets.lanl.raw import DatasetLANLRaw
from tests.datasets.lanl import verify_all_elements


def test_element_list():
    with patch("eai_graph_tools.datasets.lanl.os.path.exists", return_value=True), \
            patch("eai_graph_tools.datasets.lanl.os.path.isdir", return_value=True):
        assert set(DatasetLANLRaw("immaterial").name_elements()) == set(DatasetLANL.ELEMENT.keys())


def test_name_elements(lanl_sample):
    assert set(lanl_sample.name_elements()) == {"auth", "dns", "flows", "proc", "redteam"}


@pytest.fixture
def lanl_sample():
    return DatasetLANLRaw("tests/test_data/lanl")


def test_element_normal(lanl_sample):
    verify_all_elements(lanl_sample)


def test_element_indexed(lanl_sample):
    verify_all_elements(lanl_sample, indexed=True)


def test_element_labeled(lanl_sample):
    verify_all_elements(lanl_sample, labeled=True)
