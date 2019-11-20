#!/usr/bin/env python3
from argparse import ArgumentParser
from eai_graph_tools.datasets.unb2017.data_generator import UNB2017ToH5


if __name__ == "__main__":
    """
        Ex:
            $ export PYTHONPATH="${PYTHONPATH}:."
            $ source venv/bin/activate
            (venv)$ python3 datasets/raw_generators/unb2017/cli_tools/create_raw_hdf5.py
            -i ~/datasets/unb/2017/TrafficLabelling/ -o datasets//pytorch_geometric/unb2017/raw/unb2017.h5 -k False
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_data", help="UNB2017 CSV files folder.")
    parser.add_argument("-o", "--output_file", help="Output as Pandas dataframe file (hdf5).")
    parser.add_argument('--blanks', dest='blanks', action='store_true',
                        help="Keep or not the blank periods in the dataset (default to False).")
    parser.add_argument('--no-blanks', dest='blanks', action='store_false')
    parser.set_defaults(blanks=False)

    args = parser.parse_args()

    inp = UNB2017ToH5.Input(args.input_data,
                            args.output_file,
                            args.blanks)
    UNB2017ToH5().run(inp)
