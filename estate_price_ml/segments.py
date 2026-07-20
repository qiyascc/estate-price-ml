import pandas as pd

from estate_price_ml import config


def split_segment(df, name):
    spec = config.SEGMENTS[name]
    mask = pd.Series(True, index=df.index)
    for column, value in spec.items():
        mask = mask & (df[column] == value)
    return df[mask].reset_index(drop=True)


def all_segments(df):
    return {name: split_segment(df, name) for name in config.SEGMENTS}


def segment_sizes(df):
    return {name: int(len(split_segment(df, name))) for name in config.SEGMENTS}
