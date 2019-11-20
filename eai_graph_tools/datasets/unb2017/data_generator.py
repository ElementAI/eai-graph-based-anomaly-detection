import attr
import networkx as nx
import os
import pandas as pd
from eai_graph_tools.graph_tools.exporters.pandas.export_h5 import export_dataframe_to_h5
from eai_graph_tools.airflow_tasks.legacy_pipeline.step import Step
from numpy import dtype
from typing import Optional


class UNB2017ToH5(Step):
    @attr.s(auto_attribs=True)
    class Input():
        data_dir: str
        out_dir: Optional[str]
        blanks: bool

    Output = str

    def _run(self, params):
        out_dir = params.out_dir
        if out_dir is None:
            out_dir = params.data_dir
        out_file = os.path.join(out_dir, "unb2017.h5")
        convert_unb_to_hdf5(params.data_dir,
                            out_file,
                            params.blanks,
                            timestamp_from=None,
                            timestamp_to=None)
        return out_file


adjusted_dataset_start = pd.Timestamp(year=2017, month=7, day=3, hour=8, minute=55, second=0)
adjusted_dataset_end = pd.Timestamp(year=2017, month=7, day=5, hour=1, minute=47, second=0)


# COMPLETE DATASET (blanks are between those periods)
unb2017_time_intervals_reordered = [
    {'start': pd.Timestamp(year=2017, month=7, day=3, hour=8, minute=55, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=3, hour=13, minute=0, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=3, hour=1, minute=0, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=3, hour=5, minute=2, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=4, hour=8, minute=53, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=4, hour=13, minute=0, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=4, hour=1, minute=0, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=4, hour=5, minute=1, second=25)},
    {'start': pd.Timestamp(year=2017, month=7, day=5, hour=8, minute=42, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=5, hour=13, minute=0, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=5, hour=1, minute=0, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=5, hour=5, minute=10, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=6, hour=8, minute=59, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=6, hour=13, minute=0, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=6, hour=1, minute=0, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=6, hour=5, minute=4, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=7, hour=8, minute=59, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=7, hour=13, minute=0, second=0)},
    {'start': pd.Timestamp(year=2017, month=7, day=7, hour=1, minute=0, second=0),
     'end': pd.Timestamp(year=2017, month=7, day=7, hour=5, minute=2, second=0)},
]


# Importing UNB from original files
def convert_unb_to_hdf5(traffic_labelling_folder,
                        output_file,
                        keep_blanks=False,
                        timestamp_from=None,
                        timestamp_to=None):

    df = convert_csv_to_dataframe(traffic_labelling_folder)

    if timestamp_from and timestamp_to:
        df = df.loc[timestamp_from:timestamp_to]

    if keep_blanks:
        export_dataframe_to_h5(df, output_file)
    else:
        print("Removing blank periods")
        initial_value = unb2017_time_intervals_reordered[0]['start']

        previous_end = 0
        for i, ti in enumerate(unb2017_time_intervals_reordered):
            if i == 0:
                delta = ti['start'] - initial_value
            else:
                delta = delta + ti['start'] - previous_end

            previous_end = ti['end']
            df.loc[ti['start']:ti['end'], 'Adjusted Time'] = df.loc[ti['start']:ti['end'], 'Timestamp'] - delta

        df['index'] = df['Adjusted Time']
        df.set_index('index', inplace=True)
        export_dataframe_to_h5(df.sort_values(by='index'), output_file)


unb_2017_dtypes = {
    'Flow ID': dtype('O'),
    'Source IP': dtype('O'),
    'Source Port': dtype('int64'),
    'Protocol': dtype('int64'),
    'Destination IP': dtype('O'),
    'Destination Port': dtype('int64'),
    'Timestamp': dtype('O'),
    'Flow Duration': dtype('int64'),
    'Total Fwd Packets': dtype('int64'),
    'Total Backward Packets': dtype('int64'),
    'Label': dtype('O')}


use_cols = [
    'Flow ID',
    'Source IP',
    'Source Port',
    'Protocol',
    'Destination IP',
    'Destination Port',
    'Timestamp',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Label',
]


def convert_csv_to_dataframe(directory, drop_na=True):

    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file = os.path.join(directory, filename)
            print(f"Importing: {file}")
            data = pd.read_csv(file,
                               na_filter=False,
                               skipinitialspace=True,
                               dtype=unb_2017_dtypes,
                               usecols=use_cols,
                               encoding='cp1252',
                               # Technically a hack, this skips any line starting with a comma.
                               # These lines have no label and are invalid, so they should be skipped
                               # like comments, but strictly speaking they are not comments
                               comment=",")

            if "Monday-WorkingHours.pcap_ISCX.csv" in filename:
                data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%d/%m/%Y %H:%M:%S")
            else:
                data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%d/%m/%Y %H:%M")
            dataframes.append(data)

    df = pd.concat(dataframes, ignore_index=True, sort=True)
    df = df.sort_values(by=['Timestamp'])

    if drop_na:
        df = df.dropna()

    print(f"Shape: {df.shape}")
    print(df.head()["Timestamp"])
    print("---")
    print(df.tail()["Timestamp"])

    df['index'] = df['Timestamp']
    df.set_index('index', inplace=True)

    return df


def create_nx_graph_continuous(df,
                               timestamp_start=None,
                               timestamp_end=None,
                               timestamp_format='pandas_timestamp',  # use 'str' if graphml output is wanted!
                               filter_field="Source IP",
                               ip_addr_filter_list=None):

    if timestamp_start is not None or timestamp_end is not None:
        df = df.loc[timestamp_start:timestamp_end]

    if ip_addr_filter_list is not None:
        df = df[df[filter_field].isin(ip_addr_filter_list)]

    ports = pd.unique(df[['Source Port', 'Destination Port']].values.ravel('K'))
    ips = pd.unique(df[['Source IP', 'Destination IP']].values.ravel('K'))

    if timestamp_start is None:
        timestamp_start = df['Timestamp'].iloc[0]
    if timestamp_end is None:
        timestamp_end = df['Timestamp'].iloc[-1]

    from_t = timestamp_start.to_pydatetime().strftime('%Y-%m-%d_%H%M%S%f')
    to_t = timestamp_end.to_pydatetime().strftime('%Y-%m-%d_%H%M%S%f')

    print(f"from_t: {from_t} to_t:{to_t}")
    G = nx.MultiGraph(mode="dynamic", start=from_t, end=to_t)

    # TODO: add node types
    if timestamp_format == 'pandas_timestamp':
        start = timestamp_start
        end = timestamp_end
    elif timestamp_format == 'str':
        start = from_t
        end = to_t

    G.add_nodes_from(ports, start=start, end=end, type="port")
    G.add_nodes_from(ips, start=start, end=end, type="ip")

    for index, row in df.iterrows():
        timestamp = row['Timestamp']
        src_ip = row['Source IP']
        src_port = row['Source Port']
        dst_ip = row['Destination IP']
        dst_port = row['Destination Port']
        fwd_packets = row['Total Fwd Packets']
        backward_packets = row['Total Backward Packets']
        label_class = row['Label']
        protocol = row['Protocol']
        duration = row['Flow Duration']

        start = timestamp
        # Duration in us ... keep a 1ms granularity
        if duration < 1000:
            duration = 1
        else:
            duration = duration // 1000
        end = start + pd.Timedelta(milliseconds=duration)

        if timestamp_format == 'str':
            start = start.to_pydatetime().strftime('%Y-%m-%d_%H%M%S%f')
            end = end.to_pydatetime().strftime('%Y-%m-%d_%H%M%S%f')

        # Setting labels for nodes on the current window
        G.nodes[src_ip]['label'] = src_ip
        G.nodes[src_port]['label'] = src_port
        G.nodes[dst_ip]['label'] = dst_ip
        G.nodes[dst_port]['label'] = dst_port

        # Connecting Src IP -> Dst IP nodes together
        G.add_edge(src_ip, dst_ip, start=start, end=end, label_class=label_class, type="ip_to_ip")

        # Connecting Src IP -> Src Port
        G.add_edge(src_ip, src_port, start=start, end=end, label_class=label_class, fwd_packets=fwd_packets,
                   backward_packets=backward_packets, protocol=protocol, type="ip_to_port")

        # Connecting Src Port -> Dst Port
        G.add_edge(src_port, dst_port, start=start, end=end, label_class=label_class,
                   total_packets=fwd_packets + backward_packets, protocol=protocol, type="port_to_port")

        # Connecting Dst Port -> Dst IP
        # fwd and backward reversed for dst_ip
        G.add_edge(dst_port, dst_ip, start=start, end=end, label_class=label_class, fwd_packets=backward_packets,
                   backward_packets=fwd_packets, protocol=protocol, type="ip_to_port")

    print(nx.info(G))
    return G


def generate_labels_for_interval(node_list,
                                 hdf5_input_file):
    df = pd.read_hdf(hdf5_input_file, mode='r')

    df['index'] = df['Adjusted Time']
    df.set_index('index', inplace=True)

    for node in node_list:
        df.loc[(df['Source IP'] == node) | (df['Destination IP'] == node)]
