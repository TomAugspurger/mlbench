from pathlib import Path
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from mlbench.utils import timings, timed, init_spark, load_config


def read(cfg, spark):
    df = spark.read.load("/tmp/foo.parq/")
    assembler = VectorAssembler(outputCol="features")
    X = assembler.setInputCols(df.columns).transform(df).select("features")
    X = X.persist()
    return X


@timed
def fit(cfg, kmeans, X):
    return kmeans.fit(X)


def main():
    cfg = Path(__file__).parent.joinpath("kmeans_config.yaml")
    cfg = load_config(str(cfg))

    spark = init_spark(cfg)
    kmeans = KMeans(k=3, seed=0)

    X = read(cfg, spark)
    fit(cfg, kmeans, X)
    print(timings)


if __name__ == '__main__':
    main()
