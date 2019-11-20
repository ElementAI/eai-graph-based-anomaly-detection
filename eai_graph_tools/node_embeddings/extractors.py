import networkx as nx
import numpy as np

from abc import ABC, abstractmethod, abstractproperty
from typing import Generic, List, MutableMapping, TypeVar

NODE_T = TypeVar("NODE_T")


class Extractor(ABC, Generic[NODE_T]):

    @abstractmethod
    def __call__(self, x: nx.Graph) -> np.array:
        """
        Method to extract the preliminary set of features from the input graph
        """
        pass

    @abstractproperty
    # NB for genericity typing doesn't check node types, which can cause runtime issues if the dataset is inconsistent
    def nodes_to_indices(self) -> MutableMapping[NODE_T, int]:
        """
        Stores a mapping from the node objects to integer indices in the array output from __call__
        This allows quick and easy access that translates comfortably into the recursive setup that will be applied
        """
        pass


class BasicStatExtractor(Extractor[NODE_T]):

    def __call__(self, graph: nx.Graph) -> np.array:
        base_features: List = []
        nodes = nx.nodes(graph)
        self._nodes_to_indices: MutableMapping[NODE_T, int] = {}
        for node in nodes:
            # This field is updated on graph construction, so it allows O(1) access
            # https://github.com/networkx/networkx/blob/master/networkx/classes/graph.py
            sub_nodes = list(graph.adj[node]) + [node]
            sub_g = nx.subgraph(graph, sub_nodes)
            overall_counts = sum([len(graph.adj[x]) for x in sub_nodes])
            in_counts = len(nx.edges(sub_g))
            deg = nx.degree(sub_g, node)
            trans = nx.clustering(sub_g, node)
            self._nodes_to_indices[node] = len(base_features)
            base_features.append([in_counts,
                                  overall_counts,
                                  float(in_counts) / float(overall_counts),
                                  float(overall_counts - in_counts) / float(overall_counts),
                                  deg,
                                  trans])

        return np.array(base_features)

    @property
    def nodes_to_indices(self) -> MutableMapping[NODE_T, int]:
        return self._nodes_to_indices


class BostonExtractor(Extractor[NODE_T]):

    def __call__(self, graph: nx.Graph) -> np.array:
        base_features: List = []
        nodes = nx.nodes(graph)
        self._nodes_to_indices: MutableMapping[NODE_T, int] = {}
        for node in nodes:
            # This field is updated on graph construction, so it allows O(1) access
            # https://github.com/networkx/networkx/blob/master/networkx/classes/graph.py
            sub_nodes = list(graph.adj[node]) + [node]
            sub_g = nx.subgraph(graph, sub_nodes)
            overall_counts = sum([len(graph.adj[x]) for x in sub_nodes])
            in_counts = len(nx.edges(sub_g))
            deg = nx.degree(sub_g, node)
            trans = nx.clustering(sub_g, node)
            self._nodes_to_indices[node] = len(base_features)
            base_features.append([deg,
                                  in_counts - deg,  # Inter-neighbor edges
                                  overall_counts - (in_counts + deg),  # Extra-neighborhood edges
                                  trans])

        if np.array(base_features).shape[1] != 4:
            raise RuntimeError()

        return np.array(base_features)

    @property
    def nodes_to_indices(self) -> MutableMapping[NODE_T, int]:
        return self._nodes_to_indices


class DegreeExtractor(Extractor[NODE_T]):

    def __call__(self, graph: nx.Graph) -> np.array:
        base_features: List = []
        nodes = nx.nodes(graph)
        self._nodes_to_indices: MutableMapping[NODE_T, int] = {}
        for node in nodes:
            self._nodes_to_indices[node] = len(base_features)
            base_features.append([nx.degree(graph, node, weight='weight')])

        return np.array(base_features)

    @property
    def nodes_to_indices(self) -> MutableMapping[NODE_T, int]:
        return self._nodes_to_indices
