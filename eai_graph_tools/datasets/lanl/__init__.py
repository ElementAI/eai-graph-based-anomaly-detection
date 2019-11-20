import os.path
from typing import Tuple, List, Optional, Any, Iterable, Mapping, cast, Dict

import dask.dataframe as dd

from eai_graph_tools.datasets.artifacts import extractor_map, ExtractorArtifacts, as_is, ignore_variation


Column = Tuple[str, str]
Columns = List[Column]
ColumnLabeling = Tuple[str, str, str]


class DirNotFound(Exception):

    def __init__(self, path: str) -> None:
        super().__init__(f"LANL dataset directory {path} not found or unreadable.")
        self.path = path


class ElementMissing(Exception):

    def __init__(self, path: str, element: str) -> None:
        super().__init__(f"Element {element} missing from LANL dataset in {path}.")
        self.path = path
        self.element = element


class ElementDoesNotExist(Exception):

    def __init__(self, element: str) -> None:
        super().__init__(f"Element {element} is not a part of the LANL dataset.")
        self.element = element


class DatasetLANL:

    ELEMENT: Mapping[str, Mapping[str, List]] = {
        "auth": {
            "columns": [
                ("time", "int64"),
                ("userdomain_source", "category"),
                ("userdomain_destination", "category"),
                ("computer_source", "category"),
                ("computer_destination", "category"),
                ("type_auth", "category"),
                ("type_logon", "category"),
                ("orientation_auth", "category"),
                ("outcome_auth", "category")
            ],
            "labeled_by": [("redteam", "is_attack", "bool")]
        },
        "proc": {
            "columns": [
                ("time", "int64"),
                ("userdomain_source", "category"),
                ("computer_destination", "category"),
                ("name_process", "category"),
                ("status_process", "category")
            ]
        },
        "flows": {
            "columns": [
                ("time", "int64"),
                ("duration", "int64"),
                ("computer_source", "category"),
                ("port_source", "category"),
                ("computer_destination", "category"),
                ("port_destination", "category"),
                ("protocol", "category"),
                ("num_packets", "int32"),
                ("num_bytes", "int64")
            ]
        },
        "dns": {
            "columns": [
                ("time", "int64"),
                ("computer_source", "category"),
                ("computer_destination", "category")
            ]
        },
        "redteam": {
            "columns": [
                ("time", "int64"),
                ("userdomain_source", "category"),
                ("computer_source", "category"),
                ("computer_destination", "category")
            ]
        }
    }
    ALL_COLUMNS = dict(
        sum([v['columns'] for v in ELEMENT.values()], cast(Columns, [])) + [
            (name, dtype) for _, name, dtype in sum([cast(Dict, v).get("labeled_by", []) for v in ELEMENT.values()], [])
        ]
    )
    ARTIFACTS = [
        "userdomain",
        "computer",
        "name",
        "port",
        "protocol"
    ]

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__()
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "raw")
        self._path = os.path.realpath(path)
        self._verify_coherent()

    @property
    def path(self) -> str:
        return self._path

    def name_elements(self) -> Iterable[str]:
        for n in DatasetLANL.ELEMENT:
            if n != "redteam":
                yield n

    def element(self, name: str, **kwargs: Any) -> dd.DataFrame:
        if name not in DatasetLANL.ELEMENT:
            raise ElementDoesNotExist(name)
        return self._element(name, **kwargs)

    def _element(self, name: str, **ignored: Any) -> dd.DataFrame:
        columns_dtype = DatasetLANL.ELEMENT[name]['columns'] + \
            [(col, dtyp) for _, col, dtyp in DatasetLANL.ELEMENT[name].get('labeled_by', [])]
        return dd.read_parquet(
            self._join(name + ".parquet"),
            categories=dict([(col, 0xffff) for col, dtyp in columns_dtype if dtyp == 'category'])
        )

    def concat(self, names: Optional[Iterable[str]] = None, **kwargs: Any) -> dd.DataFrame:
        return dd.concat([self.element(name, **kwargs) for name in names or self.name_elements()])

    @classmethod
    def _element_columns(cls, name: str) -> Columns:
        return [c for c, _ in DatasetLANL.ELEMENT[name]['columns']]

    def _join(self, suffix) -> str:
        return os.path.join(self.path, suffix)

    def _path_to_element(self, name: str) -> str:
        return self._join(name + ".parquet")

    def _verify_coherent(self) -> None:
        if not os.path.isdir(self.path):
            raise DirNotFound(self.path)
        for element in self.name_elements():
            if not os.path.exists(self._path_to_element(element)):
                raise ElementMissing(self.path, element)

    @staticmethod
    def artifact_extractor(mash_source_destination: bool = False) -> ExtractorArtifacts:
        select_source_destination = ignore_variation if mash_source_destination else as_is
        return extractor_map(
            {
                "userdomain": select_source_destination,
                "computer": select_source_destination,
                "port": select_source_destination,
                "type_auth": as_is,
                "name_process": as_is,
                "protocol": as_is
            },
            ["_source", "_destination"]
        )
