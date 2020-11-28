import pandas as pd
from typing import NamedTuple
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.utils.feature_select import select_feature
from core.utils.manage_report import export_train_report


class DecisionTree:
    def __init__(self, problem_type: str, random_state=42, **kwargs):
        """
        problem_type: str, problem type which is one of 'binary', 'multiclass', 'regression'
        random_state: int
        **kwargs: Parameters of the estimator, DecisionTreeRegressor, DecisionTreeClassifier
            DecisionTreeClassifier: https://bit.ly/3lzLWeK
            DecisionTreeRegressor: https://bit.ly/310Ds8M
        """
        if problem_type == "regression":
            self.estimator = DecisionTreeRegressor(random_state=random_state, **kwargs)
        else:
            self.estimator = DecisionTreeClassifier(random_state=random_state, **kwargs)


    def fit(self, data: pd.DataFrame, meta: NamedTuple, cv, save_path, file_id: str, encoder: 'dict of encoders', tuning=False):
        """
        data: pandas.core.frame.DataFrame, data which contains features and target
        meta: dict, meta data for construct model and for the prediction with new raw data in the future
        save_path: str, path to save the reports and meta files
        cv: int, 'K' value for the cross validation
        tuning: bool, decide if optimize model
        """
        if tuning:
            raise NotImplementedError()
        else:
            X = data.drop(meta.target, axis=1)
            y = data[meta.target]

            # feature selection phase
            features_for_train = select_feature(
                X=X, y=y, problem_type=meta.problem_type
            )
            meta = meta._replace(features_for_train=features_for_train)
            export_train_report(
                estimator=self.estimator,
                X=X.loc[:, features_for_train],
                y=y,
                meta=meta,
                cv=cv,
                file_id=file_id,
                save_path=save_path,
                encoder=encoder,
            )