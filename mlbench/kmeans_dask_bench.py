from pathlib import Path

import dask.dataframe as dd
from dask_ml.cluster import KMeans

from mlbench.utils import parquet_as_known, timings, timed, load_config


def read(cfg):
    df = dd.read_parquet("/tmp/foo.parq/")
    X = parquet_as_known(df, "/tmp/foo.parq/_metadata")
    X = X.persist()
    return X


@timed
def fit(cfg, km, X):
    km.fit(X)
    return km


def main():
    cfg = Path(__file__).parent.joinpath("kmeans_config.yaml")
    cfg = load_config(str(cfg))
    kmeans = KMeans(n_clusters=3, random_state=0)
    X = read(cfg)
    fit(cfg, kmeans, X)
    print(timings)


if __name__ == '__main__':
    main()
