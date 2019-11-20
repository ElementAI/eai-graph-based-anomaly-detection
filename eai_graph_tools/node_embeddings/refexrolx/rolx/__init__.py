import numpy as np

from eai_graph_tools.node_embeddings.refexrolx.refex import Refex
from .dim_reductors import nmf, DimReductor
from sklearn.preprocessing import normalize
from torch import Tensor
from typing import Mapping, Optional


class ROLX():
    """
    Simplistic implementation of the ROLX algorithm. This boils everything down to a matrix factorization,
    without some of the additional features described in the paper such as making the matrix coding cost
    a part of the loss function and using Lloyd-Max quantization and Huffman codes to reduce space.

    Regarding the factorization algoritm, the original ROLX paper cites another paper as the source of its algorithm
    (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.2286&rep=rep1&type=pdf) and uses custom
    implementations from Matlab (files NMF*.m in https://github.com/LLNL/refex-rolx). For simplicity, this
    implementation depends on Scikit-Learn, using Kullback-Leibler loss minimization and multiplicative update as
    described in the paper via the available parameters in Scikit-Learn
    (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

    For now Refex uses Refex with all parameters as default
    """
    def __init__(self,
                 graph: Tensor,
                 n_roles: int,
                 feature_extractor: Refex,
                 dim_reductor: DimReductor) -> None:
        self.graph = graph
        self.feature_extractor = feature_extractor
        self.model = (dim_reductor or nmf)(n_roles)

    def fit_roles(self) -> None:
        """
        Uses the graph specified at initialization to define roles. This occurs in two steps

        1) Feature extraction via Refex from the graph
        2) Matrix factorization to define the translation between role membership and feature significance

        There is no return value, but this method must be called before `extract_roles`. It is kept separate to
        manage the excess CPU load of the computations and keep initialization fast
        """
        print("Generating")
        features = self.feature_extractor.create_features(self.graph)
        print("Fitting")
        self.model.fit(features.tolist())

    def extract_roles(self, graph: Tensor, metric_normalization: Optional[str]) -> np.ndarray:
        """
        This method will throw sklearn.exceptions.NotFittedError unless `fit_roles` is called first

        This uses the trained translation between role membership and feature significance to calculate
        role membership probabilities for every node in the input graph based on training from the
        original graph. The output is the matrix factor normalized by row so that the values can be treated
        as probabilities of membership in the role defined by the value's index within the row.
        E.g., [0.1, 0.9] suggests a 10% probability of membership in role 0 and 90% in role 1
        """
        print("Generating features")
        new_features: Mapping = self.feature_extractor.create_features(graph)
        print("Inferring")
        features_reduced = self.model.transform(new_features)
        # L1 normalization makes it possible to treat the values as role membership propabilities
        if metric_normalization is None:
            features_normalized = features_reduced
        else:
            features_normalized = normalize(features_reduced, norm=metric_normalization)
        return features_normalized
