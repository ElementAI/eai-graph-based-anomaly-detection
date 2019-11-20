import numpy as np

from eai_graph_tools.node_embeddings.extractors import Extractor
from eai_graph_tools.node_embeddings.refexrolx.refex import Refex
from eai_graph_tools.node_embeddings.refexrolx.rolx.dim_reductors import nmf
from eai_graph_tools.node_embeddings.refexrolx.rolx import ROLX
from .rolx.dim_reductors import DimReductor
from torch import Tensor
from typing import Mapping, TypeVar, Optional

NODE_T = TypeVar("NODE_T")


class RefexRolx():

    def __init__(self,
                 n_roles: int,
                 initial_extractor: Extractor,
                 dim_reductor: DimReductor = nmf) -> None:
        super().__init__()
        self.n_roles = n_roles
        self.initial_extractor = initial_extractor
        self.dim_reductor = dim_reductor

    def initialize_embeddings(self, data: Tensor) -> None:
        # This will delay reporting an error until embed_roles is called
        # But it allows retraining to be gracefully passed over in empty cases
        if len(data.x) == 0:
            return
        self.rolx: ROLX[NODE_T] = ROLX(data, self.n_roles, Refex(self.initial_extractor), self.dim_reductor)
        self.rolx.fit_roles()

    def embed_roles(self, data: Tensor, metric_normalization: Optional[str]) -> Mapping[NODE_T, np.array]:
        return self.rolx.extract_roles(data, metric_normalization)
