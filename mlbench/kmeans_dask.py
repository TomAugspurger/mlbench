import dask.dataframe as dd
from dask_ml.cluster import KMeans

from mlbench.utils import parquet_as_known, timings, timed


def init():
    pass


def read():
    df = dd.read_parquet("/tmp/foo.parq/")
    X = parquet_as_known(df, "/tmp/foo.parq/_metadata")
    X = X.persist()
    return X


@timed
def fit(km, X):
    km.fit(X)
    return km


def main():
    kmeans = KMeans(n_clusters=3, random_state=0)
    X = read()
    fit(kmeans, X)
    print(timings)


if __name__ == '__main__':
    main()
