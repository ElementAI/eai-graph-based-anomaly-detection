import os.path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from eai_graph_tools.datasets.lanl import DatasetLANL, DirNotFound, ElementMissing, ElementDoesNotExist
from eai_graph_tools.datasets.lanl.raw import DatasetLANLRaw
from tests.datasets.lanl import verify_all_elements


def test_constructor_all_good():
    with patch("eai_graph_tools.datasets.lanl.os.path.exists", return_value=True), \
            patch("eai_graph_tools.datasets.lanl.os.path.isdir", return_value=True):
        ds = DatasetLANL("immaterial")
        assert os.path.basename(ds.path) == "immaterial"
        assert os.path.isabs(ds.path)


def test_constructor_dir_not_found():
    with pytest.raises(DirNotFound):
        DatasetLANL("/not/found")


def test_constructor_missing_element():
    def exists(p):
        return "dns" not in p
    with patch("eai_graph_tools.datasets.lanl.os.path.exists", side_effect=exists), \
            patch("eai_graph_tools.datasets.lanl.os.path.isdir", return_value=True):
        try:
            DatasetLANL("immaterial")
            pytest.fail()
        except ElementMissing as err:
            assert err.element == "dns"


def test_element_inexistent():
    def exists(p):
        return any(n in p for n in DatasetLANL.ELEMENT.keys())
    with patch("eai_graph_tools.datasets.lanl.os.path.exists", side_effect=exists), \
            patch("eai_graph_tools.datasets.lanl.os.path.isdir", return_value=True):
        with pytest.raises(ElementDoesNotExist):
            DatasetLANL("immaterial").element("nope")


def test_name_elements():
    with patch("eai_graph_tools.datasets.lanl.os.path.exists", return_value=True), \
            patch("eai_graph_tools.datasets.lanl.os.path.isdir", return_value=True):
        assert set(DatasetLANL("immaterial").name_elements()) == set(DatasetLANL.ELEMENT.keys()) - {"redteam"}


def test_element_contents():
    with TemporaryDirectory() as dir_temp:
        ds_raw = DatasetLANLRaw("tests/test_data/lanl")
        for name in ds_raw.name_elements():
            ds_raw.element(name, indexed=True, labeled=True).to_parquet(os.path.join(dir_temp, name + ".parquet"))

        verify_all_elements(DatasetLANL(dir_temp))


def test_concat():
    ds = DatasetLANLRaw("tests/test_data/lanl")
    df_concat = ds.concat(indexed=True, labeled=True)
    assert len(df_concat) == sum(len(ds.element(name)) for name in ds.name_elements())
    assert df_concat.index.min() == min(ds.element(name).index.min() for name in ds.name_elements())
    assert df_concat.index.max() == max(ds.element(name).index.max() for name in ds.name_elements())
