import dask.dataframe as dd
from dask_ml.cluster import KMeans

from .utils import parquet_as_known


def init():
    pass


def read():
    df = dd.read_parquet("/tmp/foo.parq/")
    X = parquet_as_known(df, "/tmp/foo.parq/_metadata")
    X = X.persist()
    return X


def fit(km, X):
    km.fit(X)
    return km


def main():
    kmeans = KMeans(n_clusters=3, random_state=0)
    X = read()
    fit(kmeans, X)


if __name__ == '__main__':
    main()
