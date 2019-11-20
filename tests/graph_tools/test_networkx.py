import networkx as nx
import pytest

from eai_graph_tools.graph_tools.networkx import architecture_boston, NxGraphSpace


@pytest.fixture
def two_artifact_names():
    return sorted(["asdf", "qwer"], key=hash)


def test_architecture_boston_distinct_artifacts(two_artifact_names):
    name_a, name_b = two_artifact_names
    assert architecture_boston("a", name_a, "b", name_b)
    assert not architecture_boston("b", name_b, "a", name_a)  # Avoid link duplication.


def test_architecture_boston_same_name_diff_artifact(two_artifact_names):
    name_a, name_b = two_artifact_names
    assert architecture_boston("a", name_a, "a", name_b)
    assert not architecture_boston("a", name_b, "a", name_a)


def test_architecture_boston_diff_name_same_artifact():
    name_artifact = "asdf"
    assert architecture_boston("a", name_artifact, "b", name_artifact)
    assert not architecture_boston("b", name_artifact, "a", name_artifact)


def test_architecture_boston_same_name_same_artifact():
    name = "a"
    artifact = "asdf"
    assert not architecture_boston(name, artifact, name, artifact)


def test_architecture_edge_ambiguity(two_artifact_names):
    name_a, name_b = two_artifact_names
    assert architecture_boston("a", name_a, "b", name_b)
    assert not architecture_boston("b", name_a, "a", name_b)


@pytest.fixture
def event1():
    return {
        "a": ["asdf", "qwer"],
        "b": ["asdf"],
        "c": ["zxcv"]
    }


@pytest.fixture
def event2():
    return {
        "a": ["qwer"],
        "c": ["zxcv"],
        "d": ["uiop"]
    }


def check_nodes_edges(nodes, edges, g):
    assert nodes == sorted(g.nodes())
    assert edges == sorted([tuple(sorted(edge)) for edge in g.edges()])


def run_constructor_test(klass, event1):
    check_nodes_edges(
        [("a", "asdf"), ("a", "qwer"), ("b", "asdf"), ("c", "zxcv")],
        [
            (("a", "asdf"), ("a", "qwer")),
            (("a", "asdf"), ("b", "asdf")),
            (("a", "asdf"), ("c", "zxcv")),
            (("a", "qwer"), ("b", "asdf")),
            (("a", "qwer"), ("c", "zxcv")),
            (("b", "asdf"), ("c", "zxcv"))
        ],
        NxGraphSpace(klass, architecture_boston).make(event1)
    )


def test_constructor_graph(event1):
    run_constructor_test(nx.Graph, event1)


def test_merge_graph(event1, event2):
    gs = NxGraphSpace(nx.Graph, architecture_boston)
    check_nodes_edges(
        [("a", "asdf"), ("a", "qwer"), ("b", "asdf"), ("c", "zxcv"), ("d", "uiop")],
        [
            (("a", "asdf"), ("a", "qwer")),
            (("a", "asdf"), ("b", "asdf")),
            (("a", "asdf"), ("c", "zxcv")),
            (("a", "qwer"), ("b", "asdf")),
            (("a", "qwer"), ("c", "zxcv")),
            (("a", "qwer"), ("d", "uiop")),
            (("b", "asdf"), ("c", "zxcv")),
            (("c", "zxcv"), ("d", "uiop"))
        ],
        gs.merge(gs.make(event1), gs.make(event2))
    )


def test_constructor_multigraph(event1):
    run_constructor_test(nx.MultiGraph, event1)


def test_merge_multigraphs(event1, event2):
    gs = NxGraphSpace(nx.MultiGraph, architecture_boston)
    check_nodes_edges(
        [("a", "asdf"), ("a", "qwer"), ("b", "asdf"), ("c", "zxcv"), ("d", "uiop")],
        [
            (("a", "asdf"), ("a", "qwer")),
            (("a", "asdf"), ("b", "asdf")),
            (("a", "asdf"), ("c", "zxcv")),
            (("a", "qwer"), ("b", "asdf")),
            (("a", "qwer"), ("c", "zxcv")),
            (("a", "qwer"), ("c", "zxcv")),
            (("a", "qwer"), ("d", "uiop")),
            (("b", "asdf"), ("c", "zxcv")),
            (("c", "zxcv"), ("d", "uiop"))
        ],
        gs.merge(gs.make(event1), gs.make(event2))
    )
