import dask
import networkx as nx
import pandas as pd
import pickle
import torch
from eai_graph_tools.datasets.intervals import load_dataset_vendor
from eai_graph_tools.datasets.lanl import DatasetLANL
from eai_graph_tools.node_embeddings.extractors import BostonExtractor, DegreeExtractor
from torch_geometric.data import Data


def create_lanl_dataset(tp, in_files, out_files, *op_args, **op_kwargs):
    print("create_lanl_dataset")

    dataset = load_dataset_vendor(out_files,
                                  tp['start'],
                                  tp['interval_width'])

    df = DatasetLANL(in_files["raw_file"].path).element("auth")

    if tp['feature_extractor'] == 'degree':
        extractor = DegreeExtractor()
    elif tp['feature_extractor'] == 'boston':
        extractor = BostonExtractor()

    process_intervals(df,
                      dataset,
                      tp['start'],
                      tp['end'],
                      tp['interval_width'],
                      tp['graph_representation'],
                      extractor)


def process_intervals(df,
                      dataset,
                      set_start,
                      set_end,
                      width,
                      graph_representation,
                      extractor):
    dask_intervals = ()
    span = pd.date_range(start=set_start,
                         end=set_end - width,
                         freq=width)
    for i, start in enumerate(span):
        end = start + width
        # Tuple append
        dask_intervals = dask_intervals + \
            (dask.delayed(process_interval)(df.loc[start.timestamp():end.timestamp()],
                                            start,
                                            end,
                                            graph_representation,
                                            extractor,
                                            dataset.data_files[i],
                                            dataset.interval_files[i],
                                            dataset.index_files[i],
                                            label_index="is_attack"),)

    # Unify the output with a passthrough task
    # This will likely be replaced with a persistence on disk that is compatible with Airflow
    return dask.delayed(lambda *results: results)(*dask_intervals).compute()


def process_interval(interval_df,
                     start,
                     end,
                     graph_representation,
                     extractor,
                     data_file,
                     interval_file,
                     index_file,
                     label_index):
    print("Interval %s to %s" % (str(start), str(end)))
    if graph_representation == "shallow_simplified_edges":
        G, interval_info = create_shallow_simplified_edges_graph(interval_df, label_index)
    elif graph_representation == "boston":
        G, interval_info = create_boston_graph(interval_df, label_index)

    nodes_occurence = {}
    for i, node_id in enumerate(G.nodes):
        nodes_occurence[node_id] = i

    # convert to tensors
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    x = torch.Tensor(extractor(G).tolist())
    data = Data(x=x, edge_index=edge_index)

    # This is wrong: labels are not used for benign/malicious here (to comply with gexf format)
    y = [0 if G.node[i].get('label', 'False') == 'False' else 1 for i in G.nodes]

    data.y = torch.tensor(y)

    torch.save(data, data_file)
    with open(interval_file, "wb") as interval_f:
        pickle.dump(interval_info, interval_f)
    with open(index_file, "wb") as index_f:
        pickle.dump(nodes_occurence, index_f)

    return data, nodes_occurence, interval_info


def create_shallow_simplified_edges_graph(df, label_index):
    fields = ['userdomain_source',
              'computer_source',
              'userdomain_destination',
              'computer_destination']
    edges = [('userdomain_source', 'userdomain_destination'),
             ('userdomain_source', 'computer_source'),
             ('computer_source', 'computer_destination'),
             ('userdomain_destination', 'computer_destination')]
    return create_graph(df, fields, edges, label_index=label_index)


def create_boston_graph(df, label_index):
    fields = ['userdomain_source',
              'computer_source',
              'userdomain_destination',
              'computer_destination']
    edges = []
    # Boston is fully connected
    for i in fields:
        for j in fields:
            # Prevent non-edges and duplicates
            if i != j and not (j, i) in edges:
                edges += [(i, j)]

    return create_graph(df, fields, edges, label_index=label_index)


def create_graph(df,
                 fields,
                 edges,
                 time_index="time",
                 label_index="label"):

    G = nx.Graph()
    interval_info = {}
    for index, row in df.iterrows():
        interval_info['timestamp'] = index
        # Jury rig for post_analysis_tasks while hard-coding of UNB is still present
        interval_info['timestamp_original'] = pd.Timestamp(int(index), unit='s')
        interval_info['timestamp_adjusted'] = pd.Timestamp(int(index), unit='s')
        for f in fields:
            G.add_node(row[f], label=row[label_index])
            interval_info[row[f]] = row[label_index]
            interval_info[row[f]] = row[label_index]

        for e in edges:
            # Connecting Dst Port -> Dst IP
            if G.has_edge(e[0], e[1]):
                G.edges[e[0], e[1]]['weight'] += 1
            else:
                G.add_edge(e[0], e[1], weight=1)

    return G, interval_info
