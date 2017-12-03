from dask_ml import datasets
import dask.dataframe as dd


def main(n_samples=10_000, n_features=500, chunks=1000):
    X, y = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        chunks=chunks)
    df = dd.from_array(X).rename(columns=str)
    df.to_parquet("/tmp/foo.parq")
    return df


if __name__ == '__main__':
    main()
