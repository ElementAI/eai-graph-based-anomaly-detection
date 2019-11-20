import torch
from torch_geometric.nn import SAGEConv
from scipy.sparse import coo_matrix
import numpy as np


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
def test_coo_matrix():
    row = np.array([0, 1])
    col = np.array([1, 0])
    data = np.array([1, 2])

    coo_array = coo_matrix((data, (row, col)), shape=(2, 2)).toarray()
    assert coo_array[0][1] == 1
    assert coo_array[1][0] == 2


def test_sage_conv():
    row = np.array([0, 1, 1, 2])
    col = np.array([1, 0, 2, 1])
    data = np.array([1, 3, 5])

    in_channels, out_channels = (1, 1)
    edge_index = torch.tensor([row, col])
    num_nodes = edge_index.max().item() + 1

    x = torch.tensor([data], dtype=torch.float).permute(1, 0)

    assert x.size() == (num_nodes, in_channels)

    conv = SAGEConv(in_channels, out_channels, normalize=False, bias=False)
    conv.weight[0][0] = 1.0  # Forcing the weight to 1.0 for testing convenience!

    assert conv.__repr__() == f'SAGEConv({in_channels}, {out_channels})'
    output = conv(x, edge_index)

    assert output[0][0] == 2  # (1+3)/2
    assert output[1][0] == 3  # (1+3+5)/3
    assert output[2][0] == 4  # (3+5)/2
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
