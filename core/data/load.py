import os
from typing import NamedTuple

import pandas as pd
import numpy as np


def load(input: 'data path', target: 'target name', problem_type: str) -> ("data", "meta"):
    """
    Load dataset and generate meta data.
    """
    data = pd.read_csv(input)
    meta = _summarize(data, Meta(target=target, problem_type=problem_type))

    return data, meta


class Meta(NamedTuple):
    target: str = None
    problem_type: str = None
    categorical: list = list()
    dropped_cols: list = list()
    encoder_path: str = None
    features_for_train: list = list()
    stats: dict = None


def _summarize(data: pd.DataFrame, meta: NamedTuple) -> 'meta':
    cats = _infer_categorical(data, meta)
    meta = meta._replace(categorical=cats)

    stats = dict()
    for c in data.drop(meta.target, axis=1).columns:
        if c in meta.categorical: # categorical
            value = data.loc[data[c].notnull(), c].value_counts().index[0]
            stats[c] = (value, 'MODE')
        else: # numeric
            value = data.loc[data[c].notnull(), c].median()
            stats[c] = (value, 'MEDIAN')
    meta = meta._replace(stats=stats)
    return meta


def _infer_categorical(
    data: pd.DataFrame, meta: NamedTuple
) -> "categorical cols list":
    if meta is None:
        return list(data.dtypes[data.dtypes == np.object].index)
    else:
        temp = data.drop(meta.target, axis=1)
        return list(temp.dtypes[temp.dtypes == np.object].index)