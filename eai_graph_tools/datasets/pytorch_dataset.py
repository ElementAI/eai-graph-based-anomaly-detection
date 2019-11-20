import torch
import networkx as nx
import os.path as osp
import pandas as pd

from eai_graph_tools.datasets.intervals import load_dataset_vendor
from eai_graph_tools.node_embeddings.extractors import BostonExtractor, DegreeExtractor
from eai_graph_tools.datasets.intervals import FullData
from eai_graph_tools.datasets import convert_label_to_numerical


def create_dataset(log,
                   in_files,
                   out_files,
                   start,
                   end,
                   interval_width,
                   interval_overlap,
                   graph_representation,
                   feature_extractor):

    log.info("create_dataset")
    dataset = load_dataset_vendor(out_files,
                                  start,
                                  interval_width)

    df = pd.read_hdf(in_files['raw_file'].path, mode='r')

    # TODO: fix this in raw unb df file.
    if 'unb' in in_files['raw_file'].path:
        print("UNB workaround: modifying the index")
        df['index'] = df['Adjusted Time']  # Set "Adjusted Time" as index
        df = df.sort_values('Adjusted Time')
        df.set_index('index', inplace=True)
    else:
        df = df.sort_values('Timestamp')
        df.reset_index(inplace=True)

    # To be modified if other graph representations are being used!
    assert graph_representation == "shallow_simplified_edges" or graph_representation == "boston", \
        f"Node creation needs to be adapted for the '{graph_representation}' representation"
    assert interval_overlap == 0, "Current implementation doesn't support window overlapping"

    for file_index, file_path in enumerate(dataset.files):
        log.info(f"Checking path: {file_path}...")
        if osp.exists(file_path):
            continue

        print(f"Preparing {file_path}...")

        data = FullData()
        interval_start = start + file_index * interval_width
        interval_end = start + (file_index + 1) * interval_width
        interval_end = end if interval_end > end else interval_end

        print(f"\nfile_index: {file_index} \n start: {interval_start} \n end: {interval_end}")

        if 'unb' in in_files['raw_file'].path:
            df_interval = df.loc[pd.Timestamp(int(interval_start), unit='s'):pd.Timestamp(int(interval_end), unit='s')]
        else:
            serie = df[df['Timestamp'] >= interval_start]
            start_idx = serie.index[0] if len(serie) > 0 else None
            serie = df[(df['Timestamp'] >= interval_start) & (df['Timestamp'] < interval_end)]
            end_idx = serie.index[-1] if len(serie) > 0 else None

            if start_idx is None or end_idx is None:
                print(f"Can't find matching df entries for interval delimiters (file_path: {file_path},"
                      f"start_idx:{start_idx}, end_idx: {end_idx}")
                torch.save(data, file_path)
                return "succeeded"
            df_interval = df.iloc[start_idx:end_idx]

        if len(df_interval) == 0:
            print(f"Empty interval, saving empty data file : {file_path}")
            torch.save(data, file_path)
            return "succeeded"

        if graph_representation == "shallow_simplified_edges":
            G, interval_info = create_shallow_simplified_edges_graph(df_interval, interval_start, interval_end)
        elif graph_representation == "boston":
            G, interval_info = create_boston_graph(df_interval, interval_start, interval_end)

        if len(list(G.nodes)) == 0:
            print(f"Empty graph, saving empty data file : {file_path}")
            torch.save(data, file_path)
            return "succeeded"

        # convert to tensors
        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        if feature_extractor == 'degree':
            extractor = DegreeExtractor()
        elif feature_extractor == 'boston':
            extractor = BostonExtractor()

        node_indexes_in_tensors = {}
        for i, node_id in enumerate(G.nodes):
            node_indexes_in_tensors[node_id] = i

        node_ground_truths = list(nx.get_node_attributes(G, 'interval_ground_truth').values())

        data = FullData(x=torch.Tensor(extractor(G).tolist()),
                        y=torch.tensor(node_ground_truths),
                        edge_index=edge_index,
                        node_indexes_in_tensors=node_indexes_in_tensors)

        torch.save(data, file_path)
    return "succeeded"


def create_shallow_simplified_edges_graph(df, start, end):
    """
        Shallow + ip-ip edges and using simple edge weights.
    """
    ports = pd.unique(df[['Source Port', 'Destination Port']].values.ravel('K'))
    ips = pd.unique(df[['Source IP', 'Destination IP']].values.ravel('K'))

    G = nx.Graph(mode="interval", start=start, end=end)
    G.add_nodes_from(ports, type="port")
    G.add_nodes_from(ips, type="ip")

    interval_info = {}
    for index, row in df.iterrows():
        src_ip = row['Source IP']
        src_port = row['Source Port']
        dst_ip = row['Destination IP']
        dst_port = row['Destination Port']
        label = row['Label']
        interval_info['timestamp'] = row['Timestamp']

        # -1: malicious, 1: benign, 0: unlabelled
        G.nodes[src_ip]['interval_ground_truth'] = convert_label_to_numerical(label)
        G.nodes[dst_ip]['interval_ground_truth'] = convert_label_to_numerical(label)
        G.nodes[src_port]['interval_ground_truth'] = convert_label_to_numerical(label)
        G.nodes[dst_port]['interval_ground_truth'] = convert_label_to_numerical(label)

        # Setting labels for nodes on the current window
        G.nodes[src_ip]['label'] = src_ip
        G.nodes[src_port]['label'] = src_port
        G.nodes[dst_ip]['label'] = dst_ip
        G.nodes[dst_port]['label'] = dst_port

        # Connecting Src IP -> Dst IP nodes together
        if G.has_edge(src_ip, dst_ip):
            G.edges[src_ip, dst_ip]['weight'] += 1
        else:
            G.add_edge(src_ip, dst_ip, weight=1, label_class=label, type="ip_to_ip")

        # Connecting Src IP -> Src Port
        if G.has_edge(src_ip, src_port):
            G.edges[src_ip, src_port]['weight'] += 1
        else:
            G.add_edge(src_ip, src_port, weight=1, label_class=label, type="ip_to_port")

        # Connecting Src Port -> Dst Port
        if G.has_edge(src_port, dst_port):
            G.edges[src_port, dst_port]['weight'] += 1
        else:
            G.add_edge(src_port, dst_port, weight=1, label_class=label, type="port_to_port")

        # Connecting Dst Port -> Dst IP
        if G.has_edge(dst_port, dst_ip):
            G.edges[dst_port, dst_ip]['weight'] += 1
        else:
            G.add_edge(dst_port, dst_ip, weight=1, label_class=label, type="ip_to_port")
    return G, interval_info


def create_boston_graph(df, start, end):
    """
        Fully connected graph architecture as described by Boston Fusion

    """
    ports = pd.unique(df[['Source Port', 'Destination Port']].values.ravel('K'))
    ips = pd.unique(df[['Source IP', 'Destination IP']].values.ravel('K'))

    G = nx.Graph(mode="interval", start=start, end=end)
    G.add_nodes_from(ports, type="port")
    G.add_nodes_from(ips, type="ip")

    interval_info = {}
    for index, row in df.iterrows():
        src_ip = row['Source IP']
        src_port = row['Source Port']
        dst_ip = row['Destination IP']
        dst_port = row['Destination Port']
        label = row['Label']
        interval_info['timestamp'] = row['Timestamp']

        # -1: malicious, 1: benign
        G.nodes[src_ip]['interval_ground_truth'] = convert_label_to_numerical(label)
        G.nodes[dst_ip]['interval_ground_truth'] = convert_label_to_numerical(label)
        G.nodes[src_port]['interval_ground_truth'] = convert_label_to_numerical(label)
        G.nodes[dst_port]['interval_ground_truth'] = convert_label_to_numerical(label)

        # Setting labels for nodes on the current window
        G.nodes[src_ip]['label'] = src_ip
        G.nodes[src_port]['label'] = src_port
        G.nodes[dst_ip]['label'] = dst_ip
        G.nodes[dst_port]['label'] = dst_port

        nodes = (src_ip, src_port, dst_ip, dst_port)

        for i in nodes:
            for j in nodes:
                if i != j:
                    if i in (src_ip, dst_ip):
                        frm = "ip"
                    else:
                        frm = "port"
                    if j in (src_ip, dst_ip):
                        to = "ip"
                    else:
                        to = "port"
                    G.add_edge(i, j,
                               label_class=label,
                               weight=1,
                               type="%s_to_%s" % (frm, to))
    return G, interval_info
