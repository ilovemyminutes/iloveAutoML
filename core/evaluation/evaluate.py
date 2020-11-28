import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


def evaluate_cv(
    estimator: "trained model object",
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    scoring: list=["accuracy", "precision", "recall", "f1"],
    return_train_score=True,
) -> "dict of CV result":
    result = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
    )
    return result


class MeasuringTool:
    """
    Evaluate several metrics.

    Args
    ------
    y_true: pandas.core.series.Series or 1D numpy.ndarray, true values of target
    y_pred: pandas.core.series.Series or 1D numpy.ndarray, predicted values of target
    """

    def __init__(self, y_true: np.array, y_pred: np.array) -> "evaluation result":
        self.y_true = y_true
        self.y_pred = np.around(y_pred)

    def get_scores(self, scoring='all'):
        """
        Get 4 evaluation results: Accuracy, Precision, Recall, and F1.

        Return
        ------
        result: pandas.core.frame.DataFrame, evaluation result
        """
        if scoring == 'all':
            result = (
                pd.Series(
                    dict(
                        Accuracy=self.Accuracy,
                        Precision=self.Precision,
                        Recall=self.Recall,
                        F1=self.F1_Score,
                    )
                )
                .to_frame()
                .T
            )
        else:
            raise NotImplementedError()

        return result

    @property
    def Precision(self):
        return precision_score(y_true=self.y_true, y_pred=self.y_pred)

    @property
    def Accuracy(self):
        return (self.y_true == self.y_pred).sum() / self.y_true.shape[0]

    @property
    def Recall(self):
        return recall_score(y_true=self.y_true, y_pred=self.y_pred)

    @property
    def F1_Score(self, problem_type="binary"):
        if problem_type == "binary":
            return f1_score(y_true=self.y_true, y_pred=self.y_pred, average="binary")
        else:
            raise NotImplementedError()
