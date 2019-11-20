#!/usr/bin/env python3
from argparse import ArgumentParser
from graph_tools.exporters.export_gexf import gexf_exporter
import pandas as pd
from datasets.unb2017.data_generator import create_nx_graph_continuous


def create_animated_graph(hdf5_input,
                          gexf_output_file,
                          start,
                          end,
                          filter_field,
                          ip_addr_filter_list):

    # import as dataframe
    df = pd.read_hdf(hdf5_input, mode='r')

    print(pd.Timestamp(start))
    print(pd.Timestamp(end))

    # convert to networkx (multigraph)
    nx_multi = create_nx_graph_continuous(df,
                                          timestamp_start=pd.Timestamp(start),
                                          timestamp_end=pd.Timestamp(end),
                                          filter_field=filter_field,
                                          ip_addr_filter_list=ip_addr_filter_list)

    # export to gexf (to be opened with Gephi)
    gexf_exporter(nx_multi, gexf_output_file)


if __name__ == "__main__":
    """
        Converts a Pandas dataframe as a animated graph in Gephi (with filtering capabilities).
        Dataframe must have a timestamp as index.
        The graph building function assumes the fields of the UNB2017 dataframe.
        $ ...

    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--hdf5_input", help="'RAW' dataset file (Pandas HDF5 dataframe file")
    parser.add_argument("-g", "--gexf_output", help="Gexf output file to load in Gephi")
    parser.add_argument("-s", "--start", help="Start timestamp (such as '20170703 01:00:00')")
    parser.add_argument("-e", "--end", help="End timestamp (such as '20170703 01:00:00')")
    parser.add_argument("-f", "--filter_field", help="Field to use for filtering")
    parser.add_argument("-l", "--filter_item_list", nargs='+', type=str, help="IP list to include")

    args = parser.parse_args()

    create_animated_graph(args.hdf5_input,
                          args.gexf_output,
                          pd.Timestamp(args.start),
                          pd.Timestamp(args.end),
                          args.filter_field,
                          args.filter_item_list)
