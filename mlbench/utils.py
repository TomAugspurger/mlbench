from collections import defaultdict
from functools import wraps
from timeit import default_timer as tic

import fastparquet as fp
import yaml
from pyspark.sql import SparkSession

timings = defaultdict(list)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: grab config.
        # TODO: structlog or something similar
        t0 = tic()
        result = func(*args, **kwargs)
        t1 = tic()
        timings[func.__name__].append(t1 - t0)
        return result
    return wrapper


def parquet_as_known(df, src=None):
    row_groups = fp.ParquetFile(src).row_groups
    X = df.values
    chunks = (tuple(x.num_rows for x in row_groups),
              X.chunks[1])
    X._chunks = chunks
    return X


def init_spark():
    """Initialize a Spark Session"""
    spark = SparkSession.builder.appName("ml-bench").getOrCreate()
    return spark


def load_config(src):
    with open(src) as f:
        cfg = f.read()

    return yaml.load(cfg)
