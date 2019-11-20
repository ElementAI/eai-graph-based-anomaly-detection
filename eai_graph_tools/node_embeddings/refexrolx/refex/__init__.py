# Adapted from https://github.com/benedekrozemberczki/RolX/blob/master/src/refex.py
import numpy as np
import torch
from eai_graph_tools.node_embeddings.extractors import Extractor
from eai_graph_tools.node_embeddings.refexrolx.refex.aggregators import Aggregator, SimpleAggregator
from typing import List, MutableMapping


class Refex():

    def __init__(self,
                 extractor: Extractor,
                 recursive_iterations: int = 3,
                 bins: int = 4,
                 pruning_threshold: float = 0.9,
                 aggregator: Aggregator = SimpleAggregator()) -> None:
        self.aggregator = aggregator
        self.extractor = extractor
        self.pruning_threshold = pruning_threshold
        self.recursive_iterations = recursive_iterations
        self.bins = bins
        self._features: MutableMapping[int, List] = {}

    def single_recursion(self, i, tens):
        print("Calculating adjacency")
        nodes = tens.edge_index[0].unique()
        adj = {x.item(): set() for x in nodes}
        for k in range(len(tens.edge_index[0])):
            adj[tens.edge_index[0][k].item()] |= set([tens.edge_index[1][k].item()])
        print("Iteration %i" % i)
        features_from_previous_round = self._features[i].shape[1]
        new_features = np.zeros((max(nodes) + 1, features_from_previous_round * self.aggregator.entries()))
        for k in nodes:
            main_features = self._features[i][list(adj[k.item()]), :]
            new_features[k, :] = np.ndarray.flatten(np.array(
                [self.aggregator(main_features[:, j]) for j in range(0, features_from_previous_round)]
            ))
        return new_features

    def sub_selector(self, new_features):
        """
        Pruning of features judged to be redundant based on their correlation. This method is currently unused since it
        can alter the dimensionality of the output, which breaks an assumption of ROLX
        """
        indices = set()
        for i in range(new_features.shape[1]):
            for j in range(new_features.shape[1]):
                if i != j:
                    c = np.corrcoef(new_features[:, i], new_features[:, j])
                    if abs(c[0, 1]) > self.pruning_threshold:
                        indices = indices.union(set([j]))

        keep = list(set(range(0, new_features.shape[1])).difference(indices))
        new_features = new_features[:, keep]
        indices = set()

        return new_features

    def do_recursions(self, tens):
        for recursion in range(0, self.recursive_iterations):
            new_features = self.single_recursion(recursion, tens)
            new_features = self.bin_features(new_features)

            self._features[recursion + 1] = new_features

    def bin_features(self, unbinned_features: np.array) -> np.array:
        return np.digitize(unbinned_features, np.logspace(0, np.log10(np.amax(unbinned_features))))

    def create_features(self, tens: torch.Tensor) -> np.ndarray:
        self._features[0] = tens.x.to(torch.float)
        self.do_recursions(tens)
        return self._features[self.recursive_iterations]
