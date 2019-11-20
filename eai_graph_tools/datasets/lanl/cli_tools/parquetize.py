import os.path
import sys

from dask.distributed import Client

from eai_graph_tools.datasets.lanl.raw import DatasetLANLRaw


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path to Los Alamos dataset> [<Dask scheduler addr:port>]")
        sys.exit(1)
    path_dataset = sys.argv[1]
    if len(sys.argv) >= 3:
        print(f"Connecting to distributed Dask cluster scheduled from {sys.argv[1]}", file=sys.stderr)
        client = Client(sys.argv[2])
    else:
        print(f"Setting up local ad hoc Dask compute cluster, with default parameters", file=sys.stderr)
        client = Client()

    ds = DatasetLANLRaw(path_dataset)
    futures = []
    for name in ds.name_elements():
        path_parquet = os.path.join(ds.path, name + ".parquet")
        print(f"{name}: getting dataframe, labeled and indexed by time")
        df = ds.element(name, indexed=True, labeled=True)
        print(f"{name}: storing as Parquet to {path_parquet} (overwriting)", file=sys.stderr)
        futures.append(df.to_parquet(path_parquet, compression="gzip"))
    client.gather(futures)
