import logging
from typing import Any, Iterable

import dask.dataframe as dd

from eai_graph_tools.datasets.lanl import DatasetLANL


logger = logging.getLogger(__file__)


class DatasetLANLRaw(DatasetLANL):

    def _element(self, name: str, **kwargs: Any) -> dd.DataFrame:
        logger.info(f"Element {name}: reading CSV file")
        df = dd.read_csv(
            self._path_to_element(name),
            names=self._element_columns(name),
            dtype=DatasetLANL.ALL_COLUMNS
        )

        indexed = kwargs.get("indexed", False)
        if indexed:
            logger.info(f"Element {name}: setting time as index")
            df = df.set_index("time", sorted=True)

        if kwargs.get('labeled', False):
            for label, column_new, _ in DatasetLANL.ELEMENT[name].get("labeled_by", []):
                logger.info(f"Element {name}: merging with label data element {label}")
                df_label = self.element(label, labeled=False, indexed=indexed)
                df_merged = df.merge(df_label, how="left", on=self._element_columns(label), indicator=True)
                df_merged[column_new] = (df_merged["_merge"] != "left_only")
                df_post = df_merged.drop("_merge", axis="columns")

                # Merging between automatic categorical types implicitly converts these columns, in the merge result, as
                # object columns. We must explicitly reconvert these to categorical data post-merge.
                df = df_post.astype(
                    dict((col, dtyp) for col, dtyp in DatasetLANL.ELEMENT[label]['columns'] if col != 'time')
                )

        return df

    def name_elements(self) -> Iterable[str]:
        return list(DatasetLANL.ELEMENT.keys())

    def _path_to_element(self, name: str) -> str:
        return self._join(name + ".txt")
