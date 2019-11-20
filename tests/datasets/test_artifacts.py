import pandas as pd
import pytest

from eai_graph_tools.datasets.artifacts import extractor_map, as_is, ignore_variation, drop


@pytest.fixture
def row():
    return pd.Series(
        {
            "a": "asdf",
            "b": "qwer",
            "c": "zxcv",
            "a_x": 1234,
            "b_y": 100
        },
        name=5
    )


def check_row_extraction(map_artifacts, variations, expected, row):
    assert expected == extractor_map(map_artifacts, variations)(row)


def test_extractor_map_empty(row):
    check_row_extraction({}, None, {}, row)


def test_extractor_map_no_variation(row):
    check_row_extraction(
        {"a": as_is, "b": as_is, "a_x": as_is},
        [],
        {"a": ["asdf"], "b": ["qwer"], "a_x": [1234]},
        row
    )


def test_extractor_map_as_is_with_variation(row):
    check_row_extraction(
        {"a": as_is, "b": as_is, "c": as_is},
        ["_x", "_y"],
        {"a": ["asdf"], "b": ["qwer"], "c": ["zxcv"], "a_x": [1234], "b_y": [100]},
        row
    )


def test_extractor_map_ignore_variation(row):
    check_row_extraction(
        {"a": ignore_variation, "b": as_is, "c": as_is},
        ["_x", "_y"],
        {"a": ["asdf", 1234], "b": ["qwer"], "c": ["zxcv"], "b_y": [100]},
        row
    )


def test_extractor_map_drop(row):
    check_row_extraction(
        {"a": ignore_variation, "b": ignore_variation, "c": drop},
        ["_x", "_y"],
        {"a": ["asdf", 1234], "b": ["qwer", 100]},  # b_y is b+_y, and we keep variations of b.
        row
    )


def test_extractor_map_drop_name_collision(row):
    check_row_extraction(
        {"a": ignore_variation, "a_x": drop},
        ["_x"],
        {"a": ["asdf", 1234]},  # a_x is a+_x, and we keep variations of a; keep wins over drop.
        row
    )
