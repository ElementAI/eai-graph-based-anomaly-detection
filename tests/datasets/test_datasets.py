from contextlib import contextmanager
from copy import copy, deepcopy
import io
import os.path
import pickle
from unittest.mock import patch, MagicMock, call

import pandas as pd
import pytest

from eai_graph_tools.datasets import use_prerendered_graphs, write_graph_and_info, merge_graphs
from eai_graph_tools.graph_tools import GraphSpace


@contextmanager
def exercising_write_graph_and_info(obj_expected, interval_info):
    path_wb = "path_output"
    file_wb = MagicMock(spec=io.RawIOBase)

    def my_open(path, mode, **kwargs):
        if mode == "rb":
            # This happens when testing merge_graph.
            return io.BytesIO(pickle.dumps({path}))
        elif mode == "wb":
            # This would be the output of write_graph_and_info.
            assert path == path_wb
            return file_wb
        else:
            assert False, f"Unexpected open mode: {mode}"

    with patch("eai_graph_tools.datasets.open", create=True, side_effect=my_open) as mock_open:
        yield path_wb
        mock_open.assert_called_with(path_wb, "wb")
        file = file_wb.__enter__.return_value
        assert file.write.call_args == call(pickle.dumps((obj_expected, interval_info)))


@pytest.fixture
def interval_info():
    ii = {"timestamp_original": pd.Timestamp.now()}
    ii["timestamp_adjusted"] = copy(ii['timestamp_original'])
    return ii


def test_write_graph_and_info(interval_info):
    with exercising_write_graph_and_info("asdf", interval_info) as path_output:
        write_graph_and_info("asdf", interval_info, path_output)


class GraphSpaceSetPaths(GraphSpace):

    def make(self, event):
        return set(event.keys())

    def merge(self, left, right):
        return left | right

    def list_nodes(self, setpaths):
        return list(setpaths)


def test_merge_graphs_no_path(interval_info):
    with exercising_write_graph_and_info(set(), interval_info) as path_output:
        merge_graphs([], GraphSpaceSetPaths(), interval_info, path_output)


def test_merge_graphs_afew(interval_info):
    nodes = {"asdf", "qwer", "zxcv"}
    final_ii = deepcopy(interval_info)
    final_ii.update(dict((n, 0) for n in nodes))

    with exercising_write_graph_and_info(nodes, final_ii) as path_output:
        merge_graphs(list(nodes), GraphSpaceSetPaths(), deepcopy(interval_info), path_output)


def test_use_prerendered_graphs_bad_representation():
    with pytest.raises(AssertionError):
        use_prerendered_graphs(lambda ts: [], GraphSpaceSetPaths())({"graph_representation": "boston"}, {}, {})


def graph_files_regular_intervals(start, interval_width, num):
    layout = [start + k * interval_width for k in range(num)]

    def ts2files(dir_base, ts_start, ts_end):
        return [
            os.path.join(dir_base, str(n))
            for n, ts in enumerate(layout)
            if ts >= ts_start and ts < ts_end
        ]

    return ts2files


def run_test_use_prerendered_graphs(
    start_input,
    interval_width_input,
    num_input,
    tp,
    in_files,
    out_files,
    output_expected
):
    open_orig = open

    def my_open(path, mode, **kwargs):
        if mode == "rb":
            return io.BytesIO(pickle.dumps({path}))
        return open_orig(path, mode, **kwargs)

    output = {}

    def my_write_graph(G, interval_info, path_output):
        nonlocal output
        output[path_output] = (G, interval_info)

    with patch("eai_graph_tools.datasets.open", side_effect=my_open), \
            patch("eai_graph_tools.datasets.write_graph_and_info", side_effect=my_write_graph):
        assert use_prerendered_graphs(
            graph_files_regular_intervals(start_input, interval_width_input, num_input),
            GraphSpaceSetPaths()
        )(tp, in_files, out_files) == "succeeded"
        assert output == output_expected


def make_output(**kwargs):
    output = {}
    for name_output, (ts, nodes) in kwargs.items():
        ii = dict((f"timestamp_{suffix}", ts) for suffix in ["original", "adjusted"])
        ii.update(dict((n, 0) for n in nodes))
        output[name_output] = (set(nodes), ii)
    return output


def make_out_files(anchor, interval, d):
    return dict((anchor + k * interval, Path(p)) for (k, p) in d.items())


class Path(object):

    def __init__(self, p):
        self.path = p


def test_use_prerendered_graphs_1to1():
    anchor = pd.Timestamp(2019, 10, 1)
    interval = pd.Timedelta(1, unit="min")
    tp = {
        "graph_representation": "prerendered",
        "start": anchor,
        "interval_width": interval
    }
    in_files = {"raw_file": Path("input")}
    out_files = make_out_files(anchor, interval, {0: "A", 1: "B", 2: "C"})

    run_test_use_prerendered_graphs(
        anchor,
        interval,
        len(out_files),
        tp,
        in_files,
        out_files,
        make_output(
            A=(anchor, ["input/0"]),
            B=(anchor + interval, ["input/1"]),
            C=(anchor + 2 * interval, ["input/2"])
        )
    )


def test_use_prerendered_graphs_inexact_alignment():
    anchor_graphs = pd.Timestamp(2019, 10, 1, 0, 0, 0)
    interval_graphs = pd.Timedelta(1, unit="min")
    anchor_work = pd.Timestamp(2019, 10, 1, 0, 2, 1)
    interval_work = pd.Timedelta(2.75, unit="min")
    tp = {
        "graph_representation": "prerendered",
        "start": anchor_work,
        "interval_width": interval_work
    }
    in_files = {"raw_file": Path("input")}
    out_files = make_out_files(anchor_work, interval_work, {0: "A", 1: "B"})

    run_test_use_prerendered_graphs(
        anchor_graphs,
        interval_graphs,
        10,
        tp,
        in_files,
        out_files,
        make_output(
            A=(anchor_work, ["input/3", "input/4"]),
            B=(anchor_work + interval_work, ["input/5", "input/6", "input/7"])
        )
    )
