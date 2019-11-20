import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Data
from eai_graph_tools.airflow_tasks.resources import ResourcePathStatic, ResourcePathDynamic, ResourcePath
from typing import Optional, List, Dict, Tuple


def generate_intervals(start: pd.Timestamp,
                       end: pd.Timestamp,
                       interval_width: pd.Timedelta,
                       static_path_prefix: str = None,
                       dynamic_path_prefix: Optional[List[Tuple[str, str]]] = None):

    assert (static_path_prefix is not None and dynamic_path_prefix is None) or\
           (static_path_prefix is None and dynamic_path_prefix is not None), "Ill defined prefix"
    assert start < end and (end - interval_width) >= start, \
        f"Invalid start/end/interval_width config: {start}/{end}/{interval_width}"

    span = pd.date_range(start=start, end=end - interval_width, freq=interval_width)
    data_files = ['%s_%s' % (str(i), str(i + interval_width)) for i in span]

    filenames: Dict[str, ResourcePath] = {}
    for i, f in enumerate(data_files):
        if static_path_prefix is not None:
            filenames[f'data_files_{i}'] = ResourcePathStatic(path=osp.join(static_path_prefix, f))
        elif dynamic_path_prefix is not None:
            dynamic_path = dynamic_path_prefix.copy()
            dynamic_path.append(('const', f))
            filenames[f'data_files_{i}'] = ResourcePathDynamic(path=dynamic_path)
        else:
            filenames[f'data_files_{i}'] = ResourcePathStatic(path=f)

    return filenames


def load_dataset_vendor(filenames,
                        start: pd.Timestamp,
                        interval_width: pd.Timedelta):

    data_files = [f.path for k, f in filenames.items()]
    vendor = DatasetVendor(data_files, start, interval_width)
    return vendor


class FullData(Data):
    r"""Subclass of pytorch geometric's 'Data' class, which includes an array for storing the node_ids

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    """

    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 pos=None,
                 node_indexes_in_tensors: Optional[Dict] = None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.node_indexes_in_tensors = node_indexes_in_tensors
        super().__init__(x, edge_index, edge_attr, y, pos)

    def get_x(self, node_id):
        if node_id in self.node_indexes_in_tensors:
            return self.x[self.node_indexes_in_tensors[node_id]]
        else:
            assert False, f"Can't retrieve x, node_id={node_id} hasn't been found."
            return None

    def get_y(self, node_id):
        if node_id in self.node_indexes_in_tensors:
            return self.y[self.node_indexes_in_tensors[node_id]]
        else:
            assert False, f"Can't retrieve y, node_id={node_id} hasn't been found."
            return None


class DatasetVendor():
    def __init__(self,
                 files: List[str],
                 start: pd.Timestamp,
                 interval_width: pd.Timedelta):
        self.files = files
        self.start = start
        self.interval_width = interval_width
        self._index = 0

    def __len__(self):
        return len(self.files)

    def get(self, i):
        return torch.load(self.files[i])

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        self._index = self._index + 1
        return self.get(self._index - 1)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
