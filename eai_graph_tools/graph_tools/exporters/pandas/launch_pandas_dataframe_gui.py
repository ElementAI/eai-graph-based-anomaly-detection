#!/usr/bin/env python3
import dfgui
import pandas as pd
from argparse import ArgumentParser


"""
To be run as a separate script to launch panda dataframe inspector
dfgui install:
    https://github.com/bluenote10/PandasDataFrameGUI
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="HDF5 Panda dataframe file")
    args = parser.parse_args()

    df = pd.read_hdf(args.file, mode='r')
    dfgui.show(df)
