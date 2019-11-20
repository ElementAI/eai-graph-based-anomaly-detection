import os
import os.path as osp
import pickle
from typing import Callable, Sequence, Dict, TypeVar, cast
from dask import delayed
import pandas as pd
from eai_graph_tools.graph_tools import GraphSpace
from eai_graph_tools.airflow_tasks import BENIGN, MALICIOUS, UNLABELLED

T = TypeVar('T')
Timestamp2GraphFiles = Callable[[os.PathLike, pd.Timestamp, pd.Timestamp], Sequence[os.PathLike]]


def convert_label_to_numerical(label):
    if label == 'BENIGN'or label == 1:
        num = BENIGN
    elif label == 'n/a':
        num = UNLABELLED
    else:
        # 'MALICIOUS', -1, 'FTP-Patator', etc.
        # UNB should be reformatted to have a label[BENIGN,MALICIOUS] and an 'attack_type' column instead of encoding
        # the attack type into the malicious/benign label.
        num = MALICIOUS
    return num


def use_prerendered_graphs(
    ts2files: Timestamp2GraphFiles,
    graph_space: GraphSpace[T]
) -> Callable[[Dict, Dict, Dict], str]:

    def _prepare_graphs(tp: Dict, in_files: Dict, out_files: Dict) -> str:
        assert tp['graph_representation'] == "prerendered"

        dir_graphs = in_files["raw_file"].path
        start = tp['start']
        interval_width = tp['interval_width']

        jobs_done = []
        for index_interval, rp in enumerate(out_files.values()):
            if osp.isfile(rp.path):
                continue

            ts_start = start + index_interval * interval_width
            ts_next = ts_start + interval_width
            interval_info = dict((f"timestamp_{suffix}", ts_start) for suffix in ["original", "adjusted"])

            paths_graphs_relevant = ts2files(dir_graphs, ts_start, ts_next)
            if len(paths_graphs_relevant) > 0:
                jobs_done.append(delayed(merge_graphs)(paths_graphs_relevant, graph_space, interval_info, rp.path))
            else:
                write_graph_and_info(graph_space.make({}), interval_info, rp.path)

        num_jobs_done = delayed(sum)(jobs_done).compute()
        assert num_jobs_done == len(jobs_done)
        return "succeeded"

    return _prepare_graphs


def write_graph_and_info(G: T, interval_info: Dict, path_output: os.PathLike) -> None:
    with open(path_output, "wb") as f:
        pickle.dump((G, interval_info), f)


def merge_graphs(
    paths: Sequence[os.PathLike],
    graph_space: GraphSpace[T],
    interval_info: Dict,
    path_output: os.PathLike
) -> int:
    G = graph_space.make({})
    for path in paths:
        with open(path, "rb") as f:
            G = graph_space.merge(G, cast(T, pickle.load(f)))

    interval_info.update(dict((node, 0) for node in graph_space.list_nodes(G)))
    write_graph_and_info(G, interval_info, path_output)
    return 1
