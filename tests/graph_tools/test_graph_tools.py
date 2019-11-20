from copy import deepcopy
from typing import List

import dask.dataframe as dd
import pandas as pd
import pytest

from eai_graph_tools.datasets.artifacts import MapArtifacts, Artifact
from eai_graph_tools.graph_tools import GraphSpace, events_to_graphs


class SpaceMapsAsGraph(GraphSpace[dict]):

    def make(self, d: MapArtifacts) -> dict:
        return deepcopy(d)

    def merge(self, left: dict, right: dict) -> dict:
        assert all(isinstance(d, dict) for d in [left, right])
        d = deepcopy(left)
        for k, v in right.items():
            d.setdefault(k, [])
            d[k] += v
        return d

    def list_nodes(self, d: dict) -> List[Artifact]:
        return list(d.keys())


def extract_columns(row):
    return dict((name, [row[name]]) for name in row.index)


@pytest.fixture
def df():
    return dd.from_pandas(
        pd.DataFrame(
            data={
                "a": ["asdf", "qwer", "zxcv", "ghgh", None, "poiu", "fdsa", "asdf"],
                "b": ["ghgh", "qwer", None, "qwer", "qwer", "asdf", "zxcv", "qwer"],
                "c": ["1234", None, "asdf", "qwer", "6767", "hjkl", "poiu", None]
            },
            index=[1, 4, 8, 9, 13, 14, 16, 17]
        ),
        chunksize=4
    )


def run_test_events_to_graph(intervals, graphs, df, interval):
    df_obtained = events_to_graphs(
        df,
        interval,
        extract_columns,
        SpaceMapsAsGraph()
    ).compute(scheduler="single-threaded")
    assert df_obtained.equals(pd.DataFrame(data={"graph": graphs}, index=intervals))


def test_single_group_single_record(df):
    run_test_events_to_graph(
        [0],
        [{"a": ["asdf"], "b": ["ghgh"], "c": ["1234"]}],
        df.loc[0:2],
        5
    )


def test_groups_subdivide_partitions_neatly(df):
    run_test_events_to_graph(
        [0, 1, 2, 3],
        [
            {"a": ["asdf", "qwer"], "b": ["ghgh", "qwer"], "c": ["1234"]},
            {"a": ["zxcv", "ghgh"], "b": ["qwer"], "c": ["asdf", "qwer"]},
            {"a": ["poiu"], "b": ["qwer", "asdf"], "c": ["6767", "hjkl"]},
            {"a": ["fdsa", "asdf"], "b": ["zxcv", "qwer"], "c": ["poiu"]}
        ],
        df,
        5
    )


def test_groups_overlap_partitions(df):
    run_test_events_to_graph(
        [0, 1, 2],
        [
            {"a": ["asdf", "qwer"], "b": ["ghgh", "qwer"], "c": ["1234"]},
            {"a": ["zxcv", "ghgh"], "b": ["qwer", "qwer"], "c": ["asdf", "qwer", "6767"]},
            {"a": ["poiu", "fdsa", "asdf"], "b": ["asdf", "zxcv", "qwer"], "c": ["hjkl", "poiu"]}
        ],
        df,
        7
    )
