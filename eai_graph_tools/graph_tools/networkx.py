from typing import Callable, List
from uuid import uuid4
import networkx as nx
from eai_graph_tools.datasets.artifacts import Artifact, MapArtifacts, iter_map_artifacts
from eai_graph_tools.graph_tools import GraphSpace


ArchitectureGraph = Callable[[str, Artifact, str, Artifact], bool]


class NxGraphSpace(GraphSpace[nx.Graph]):

    def __init__(self, nx_class: Callable[[], nx.Graph], are_linked: ArchitectureGraph) -> None:
        # nx.MultiGraph is a subclass of nx.Graph, so either nx.Graph or nx.MultiGraph can be used as 1st parameter to
        # this constructor.
        super().__init__()
        self._nx_class = nx_class
        self._are_linked = are_linked

    def make(self, event: MapArtifacts) -> nx.Graph:
        g = self._nx_class()
        for name_left, artifact_left in iter_map_artifacts(event):
            for name_right, artifact_right in iter_map_artifacts(event):
                if self._are_linked(name_left, artifact_left, name_right, artifact_right):
                    g.add_edge((name_left, artifact_left), (name_right, artifact_right), key=uuid4())
        return g

    def merge(self, left: nx.Graph, right: nx.Graph) -> nx.Graph:
        return nx.algorithms.operators.binary.compose(left, right)

    def list_nodes(self, g: nx.Graph) -> List[Artifact]:
        return list(g.nodes)


def architecture_boston(name_left: str, artifact_left: Artifact, name_right: str, artifact_right: Artifact) -> bool:
    """
    Articulates the "Boston" graph architecture, by which vertices associated to artifacts present in an event form a
    clique within the graph of all events.
    """
    if name_left > name_right:
        return False
    if name_left < name_right:
        return True
    return hash(artifact_left) < hash(artifact_right)
