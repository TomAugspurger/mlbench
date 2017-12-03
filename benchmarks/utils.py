import fastparquet as fp


def parquet_as_known(df, src=None):
    row_groups = fp.ParquetFile(src).row_groups
    X = df.values
    chunks = (tuple(x.num_rows for x in row_groups),
              X.chunks[1])
    X._chunks = chunks
    return X
