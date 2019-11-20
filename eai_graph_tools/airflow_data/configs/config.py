import attr
import os.path as osp
from copy import deepcopy
from typing import Optional, Union, Sequence, Iterator, MutableMapping, Any
from eai_graph_tools.node_embeddings.refexrolx.rolx.dim_reductors import DimReductor

cfgs: MutableMapping[str, Any] = {}


@attr.s(auto_attribs=True)
class ModelDefaults():
    name: str
    graph_representation: str
    feature_extractor: str


PathSetting = Union[str, Sequence[str]]


def iter_path(path: PathSetting) -> Iterator[str]:
    if isinstance(path, str):
        return iter_path(path.split("/"))
    return iter(path)


"""
    WARNING! 'Config' needs to be updated to match the previous section (some fields changed with the Airflow port)
"""


class Config():

    MODELS = {
        'randr': ModelDefaults(
            name='refex-rolx',
            graph_representation='boston',
            feature_extractor='boston'
        ),
        'infomax': ModelDefaults(
            name='graphsage-gcn',
            graph_representation='shallow_simplified_edges',
            feature_extractor='degree'
        )
    }

    def __init__(self, name, **fields) -> None:
        self.name = name
        self._dict = deepcopy(fields)

    def clone(self, name: str) -> "Config":
        return Config(name, **self._dict)

    def setting(self, path: PathSetting, value: Any) -> "Config":
        components = list(iter_path(path))
        last = components.pop()
        d = self._dict
        for component in components:
            d = d.setdefault(component, {})
        d[last] = value
        return self

    def with_graph_representation(self, rep: str) -> "Config":
        return self.setting('model/graph_representation', rep)

    def with_feature_extractor(self, extractor: str) -> "Config":
        return self.setting('model/feature_extractor', extractor)

    def with_method_embedding(self, model_type: str, extractor: Optional[str] = None) -> "Config":
        if model_type not in Config.MODELS:
            raise RuntimeError(f"Bad model type: {model_type}")
        self.setting('model/model_type', model_type) \
            .setting('model/name', Config.MODELS[model_type].name)
        for path in ['model/graph_representation', 'model/feature_extractor']:
            if not self.is_defined(path):
                self.setting(path, getattr(Config.MODELS[model_type], osp.basename(path)))
        return self

    def with_dim_embedding(self, dim: int) -> "Config":
        assert dim >= 2
        return self.setting('model/hidden_dim', dim)

    def with_metric_normalization(self, metric: Optional[str]) -> "Config":
        return self.setting('inference/metric_normalization', metric)

    def with_dim_reductor(self, dim_reductor: DimReductor) -> "Config":
        return self.setting('inference/dim_reductor', dim_reductor)

    def is_defined(self, path: PathSetting) -> bool:
        d = self._dict
        for component in iter_path(path):
            if component not in d:
                return False
            d = d[component]
        return True

    def append_to(self, cfgs: MutableMapping[str, Any]) -> None:
        if not self.is_defined('model/model_type'):
            raise RuntimeError("Must have defined model type.")
        if not self.is_defined('model/hidden_dim'):
            raise RuntimeError("Must have defined embedding dimension.")
        if not self.is_defined('paths/output_path'):
            self.setting('paths/output_path', osp.join(osp.dirname(osp.realpath(__file__)), 'output_data', self.name))
        cfgs[self.name] = self._dict
