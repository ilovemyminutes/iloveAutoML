import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import permutation_importance


def select_feature(
    X, y, problem_type, estimator="dt", method="permutation", random_state=42
):
    """
    Select features using several selection methods.

    Args
    ------
    X: pandas.core.frame.DataFrame
    y: pandas.core.series.Series
    problem_type: str, problem type which is one of 'binary', 'multiclass', 'regression'

    Return
    ------
    essence_feature: list, selected feature(s)
    """
    if method == "permutation":
        importances = get_permutation_importance(
            X,
            y,
            problem_type=problem_type,
            estimator="dt",
            n_repeats=5,
            n_jobs=-1,
            random_state=random_state,
        )
        threshold = np.percentile(importances, 50, interpolation="nearest")
        essence_feature = list(X.columns[np.where(importances > threshold)[0]])
    else:
        raise NotImplementedError()

    return essence_feature


def get_permutation_importance(
    X: pd.DataFrame, y: pd.Series, problem_type: str, estimator: 'sklearn estimator object', n_repeats: int=4, n_jobs=-1, random_state=42
):
    """
    Calculate feature importance using permutation importance.

    Args
    ------
    X: pandas.core.frame.DataFrame or 2D numpy.ndarray
    y: pandas.core.series.Series or 1D numpy.ndarray
    problem_type: str, problem type which is one of 'binary', 'multiclass', 'regression'
    estimator: str, estimator used for calculating feature importance
    normalize: bool, normalize for feature importance result or not
    n_repeats: int, number of permutation
    n_jobs: int, number of cores for parallel processing
    """
    if problem_type == "regression":
        calculator = DecisionTreeRegressor(random_state=random_state).fit(X, y)
    else:
        calculator = DecisionTreeClassifier(random_state=random_state).fit(X, y)
    importances = permutation_importance(estimator=calculator, X=X, y=y, n_jobs=-1)[
        "importances_mean"
    ]

    # normalize
    importances = (importances - importances.min()) / (
        importances.max() - importances.min()
    )

    return importances
