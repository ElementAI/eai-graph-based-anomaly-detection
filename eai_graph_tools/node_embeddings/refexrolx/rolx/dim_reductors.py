from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, Callable, Optional, Generic

import numpy as np
from sklearn.decomposition import NMF, PCA


NODE_T = TypeVar("NODE_T")


class _DimReductor(ABC, Generic[NODE_T]):

    @abstractmethod
    def fit(embs: Sequence[np.ndarray]) -> None:
        pass

    @abstractmethod
    def transform(embs: np.ndarray) -> np.ndarray:
        pass


DimReductor = Optional[Callable[[int], _DimReductor]]


def nmf(n_roles: int) -> DimReductor:
    return NMF(
        n_components=n_roles,
        beta_loss='kullback-leibler',
        solver='mu',
        max_iter=500
    )


def pca(n_roles: int) -> DimReductor:
    return PCA(n_components=n_roles)
