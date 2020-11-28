import os
import json
from typing import NamedTuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def auto_preprocess(
    data: pd.DataFrame, meta: NamedTuple, **kwargs
) -> ("data", "encoder"):
    """
    Preprocess automatically(really naive for now)
    - Drop ID or other unnecessary cols
    - Imputation
    - Encoding

    Return
    ------
    data: pandas.core.frame.DataFrame, preprocessed data
    encoder: meta dict for encoders 
    """
    # drop & imputate
    data, meta = drop_and_impute(data, meta)

    # encoding for target column if it is string type
    encoder = dict()
    if data[meta.target].dtype == np.object:
        data[meta.target], encoder[meta.target] = label_encode(data[meta.target])

    # encoding for categorical features
    for feature in meta.categorical:
        data[feature], encoder[feature] = label_encode(data[feature])

    return data, encoder


def make_data_fit(input: "path of data", meta: NamedTuple):
    """
    Make input data right depending on the meta data to test in proper condition
    """
    # load data
    data = pd.read_csv(input)

    # drop
    if meta.dropped_cols:
        data = data.drop(meta.dropped_cols, axis=1)

    # imputation
    to_impute = _get_feature_status(data, meta.target)['impute']
    for imp in to_impute:
        data.loc[data[imp].isnull(), imp] = meta.stats[imp][0]

    # feature selection phase
    if meta.features_for_train:
        if meta.target in data.columns:
            data = data.loc[:, meta.features_for_train + [meta.target]]
        else:
            data = data.loc[:, meta.features_for_train]

    # encoding phase
    encoder_path = meta.encoder_path
    with open(encoder_path, "r") as json_file:
        encoder_dict = json.load(json_file)

    if encoder_dict:
        for k in list(encoder_dict.keys()):
            encoder = LabelEncoder()
            encoder.classes_ = encoder_dict[k]
            try:
                data.loc[:, k] = encoder.transform(data.loc[:, k])
            except:
                pass

    # remove unzipped files
    os.remove(meta.encoder_path)

    return data


def drop_and_impute(data, meta):
    status = _get_feature_status(data, meta.target)
    for imp in status['impute']:
        data.loc[data[imp].isnull(), imp] = meta.stats[imp][0]
    if status['drop']:
        data = data.drop(status['drop'], axis=1)
        meta = meta._replace(dropped_cols=meta.dropped_cols+dropped_cols)
    return data, meta


def label_encode(column: pd.Series):
    encoder = LabelEncoder()
    transformed_col = encoder.fit_transform(column)
    return transformed_col, encoder.classes_.tolist()


def onehot_encode(column: pd.Series):
    raise NotImplementedError()


def _get_feature_status(data: pd.DataFrame, target: str) -> 'status dict':
    
    missing_rates = data.isnull().sum() / data.shape[0]

    to_impute = list(missing_rates[(0 < missing_rates) & (missing_rates <= 0.5)].index)
    to_drop = list(missing_rates[0.5 < missing_rates].index)
    status = dict(impute=to_impute, drop=to_drop)

    id_cols = _infer_id_cols(data, target)
    status['drop'].extend(id_cols)
    
    return status


def _infer_id_cols(data: pd.DataFrame, target: str) -> 'ID cols list':
    nrow = data.drop(target, axis=1).nunique()
    return list(nrow[nrow == data.shape[0]].index)