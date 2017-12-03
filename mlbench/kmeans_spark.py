from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler


def init():
    spark = SparkSession.builder.appName("ml-bench").getOrCreate()
    return spark


def read(spark):
    df = spark.read.load("/tmp/foo.parq/")
    assembler = VectorAssembler(outputCol="features")
    X = assembler.setInputCols(df.columns).transform(df).select("features")
    X = X.persist()
    return X


def fit(kmeans, X):
    return kmeans.train(X)


def main():
    spark = init()
    kmeans = KMeans(k=3, seed=0)

    X = read(spark)
    kmeans.train(X)


if __name__ == '__main__':
    main()
