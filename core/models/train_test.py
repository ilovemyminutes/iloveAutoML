from enum import Enum

from core.data.load import load
from core.preprocessing.preprocess import auto_preprocess, make_data_fit
from core.models.decisiontree import DecisionTree
from core.utils.manage_report import (
    read_report,
    export_test_report,
    generate_file_id,
)


def train(input: str, target: str, problem_type: str, save_path: str, estimator: str, cv: int, **kwargs):
    """
    Learn automatically depending on the inputs
    - Preprocessing
        - Dropping ID Columns
        - Dropping rows which contains missing values
        - Label encoding
    - Feature Selection using Permutation Importance

    Args
    ------
    input: str, path of dataset
    target: str, name of the target column
    problem_type: str, problem type which is one of 'binary', 'multiclass', 'regression'
    save_path: str, path to save reports
    estimator: str or Enum object, type of estimator to train
    cv: int, k value for cross-validation
    """

    if isinstance(problem_type, Enum):
        problem_type = problem_type.value

    if isinstance(estimator, Enum):
        estimator = estimator.value

    data, meta = load(input=input, target=target, problem_type=problem_type)
    data, encoder = auto_preprocess(data=data, meta=meta)

    if estimator == "dt":
        model = DecisionTree(problem_type=problem_type, **kwargs)
    else:
        raise NotImplementedError()

    file_id = generate_file_id()
    model.fit(data, meta, cv, save_path, file_id, encoder=encoder)


def test(
    input: "path of data",
    meta_path: "path of meta data",
    save_path: "path to save result",
):
    """
    test for the input data with trained model

    Args
    ------
    - input: str, path of data for test
    - meta_path: str, path of meta files
    - save_path: str, path to save result
    """
    estimator, meta = read_report(meta_path)

    # make dataset in the same condition as train
    data = make_data_fit(input=input, meta=meta)
    file_id = meta_path.split("/")[-1] + "(" + input.split("/")[-1] + ")"

    # CASE1: data has no target - just prediction
    export_test_report(estimator, data, file_id, save_path, meta.target)
