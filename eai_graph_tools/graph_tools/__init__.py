from abc import ABC, abstractmethod
from functools import reduce
from typing import TypeVar, Generic, List

import dask.dataframe as dd
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

from eai_graph_tools.datasets.artifacts import Artifact, MapArtifacts, ExtractorArtifacts


T = TypeVar("T")


class GraphSpace(ABC, Generic[T]):
    """
    Expresses an object that can instantiate graph-like objects that embody relationships between artifacts, and can
    merge together such objects.
    """

    @abstractmethod
    def make(self, event: MapArtifacts) -> T:
        "Makes a graph-like object from a single event, represented as a map of named artifacts."
        raise NotImplementedError()

    @abstractmethod
    def merge(self, left: T, right: T) -> T:
        "Merges two graph-like objects into a single one."
        raise NotImplementedError()

    @abstractmethod
    def list_nodes(self, g: T) -> List[Artifact]:
        "Returns a list of all values associated to vertices of the given graph-like object (in no particular order)."
        raise NotImplementedError()


class _MergerGraphs(dd.Aggregation):
    """
    Instantiate graphs (or graph-like objects) for the artifacts of events, for the hierarchical aggregation of
    single-event graphs into multi-event graphs.
    Dask aggregator to merge graphs from a group-by series.
    """
    def __init__(self, graph_space: GraphSpace) -> None:
        super().__init__("graph-merge", self._merge, self._merge)
        self._graph_space = graph_space

    def _merge(self, gby: SeriesGroupBy) -> pd.Series:
        """
        Merges together the graph-like objects associated to the events of one progressive chunk.
        """
        index = []
        graphs = []
        for i in gby.groups.keys():
            index.append(i)
            df_group = gby.get_group(i)
            if isinstance(df_group.iloc[0], str):
                graphs.append(df_group.iloc[0])
            else:
                graphs.append(reduce(self._graph_space.merge, df_group))
        return pd.Series(graphs, index=index)


def events_to_graphs(
    df: dd.DataFrame,
    interval: pd.Timedelta,
    extract_artifacts: ExtractorArtifacts,
    graph_space: GraphSpace
) -> dd.DataFrame:
    """
    Builds graphs for described time intervals, using events stored into the given dataframe.
    """
    def row_to_graph(row: pd.Series):
        row = row.dropna()
        return pd.Series({"interval": int(row.name / interval), "graph": graph_space.make(extract_artifacts(row))})

    return df.apply(row_to_graph, axis=1, meta={"interval": "int64", "graph": "object"}) \
        .groupby("interval") \
        .agg(_MergerGraphs(graph_space))
