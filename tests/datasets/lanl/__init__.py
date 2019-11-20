from typing import Any

import dask.dataframe as dd
import pandas as pd

from eai_graph_tools.datasets.lanl import DatasetLANL


def verify_dataframe(name: str, df: dd.DataFrame, labeled: bool):
    path_data = f"tests/test_data/lanl/{name}.txt"
    columns = list(DatasetLANL._element_columns(name))
    dtype = dict(DatasetLANL.ELEMENT[name]["columns"])

    if set(df.columns) - {'time'} > set(columns) - {'time'}:
        # More columns than expected? Assume the dataset was labeled.
        path_data += ".labeled"
        for _, col, dtyp in DatasetLANL.ELEMENT[name].get("labeled_by", []):
            columns.append(col)
            dtype[col] = dtyp

    index_col = 'time'
    if 'time' in df.columns:
        index_col = None

    df_expected = pd.read_csv(path_data, names=columns, dtype=dtype, index_col=index_col)
    assert df_expected.equals(df.compute())


def verify_all_elements(ds: DatasetLANL, **kwargs: Any) -> None:
    for name in ds.name_elements():
        verify_dataframe(name, ds.element(name, **kwargs), False)
