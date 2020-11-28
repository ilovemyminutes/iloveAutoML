import os
import glob
import json
import zipfile
from datetime import datetime
from typing import NamedTuple

import pandas as pd
import joblib

from core.evaluation.evaluate import MeasuringTool, evaluate_cv
from core.data.load import Meta


def export_train_report(
    X: pd.DataFrame,
    y: pd.Series,
    meta: NamedTuple,
    estimator: "trained model object",
    cv: int,
    file_id: "ID used to save the outputs",
    save_path: str,
    encoder: dict = None,
) -> "saved path":
    """
    Export report after train.

    Args
    ------
    X: pandas.core.frame.DataFrame, feature data
    y: pandas.core.series.Series, target data
    estimator: sklearn estimator object, estimator to train and export report
    cv: int, 'K' value for cross validation
    file_id: str, ID used to save the outputs
    save_path: str, path to save
    enoder: dict, encoders used to encode categorical features
    """
    model_report = pd.Series(estimator.get_params()).to_frame("Value")
    model_report.index.name = f"Params"

    report = evaluate_cv(estimator, X, y, cv)

    train_report = pd.DataFrame(
        dict(
            Accuracy=report["train_accuracy"],
            Precision=report["train_precision"],
            Recall=report["train_recall"],
            F1=report["train_f1"],
        )
    )
    train_report = pd.concat(
        [
            train_report,
            train_report.mean().to_frame("AVG").T,
            train_report.std().to_frame("STD").T,
        ],
        axis=0,
    )
    train_report.index.name = f"K={cv}"

    valid_report = pd.DataFrame(
        dict(
            Accuracy=report["test_accuracy"],
            Precision=report["test_precision"],
            Recall=report["test_recall"],
            F1=report["test_f1"],
        )
    )
    valid_report = pd.concat(
        [
            valid_report,
            valid_report.mean().to_frame("AVG").T,
            valid_report.std().to_frame("STD").T,
        ],
        axis=0,
    )
    valid_report.index.name = f"K={cv}"

    feature_report = pd.Series(X.columns).to_frame("Features")

    estimator_name = get_estimator_name(estimator)

    folder_path = os.path.join(save_path, f"{estimator_name}_{file_id}")
    create_folder(folder_path)
    writer = pd.ExcelWriter(
        os.path.join(folder_path, f"dt_report_{file_id}.xlsx")
    )  # pylint: disable=abstract-class-instantiated
    train_report.to_excel(writer, sheet_name="Train Result")
    valid_report.to_excel(writer, sheet_name="Validation Result")
    model_report.to_excel(writer, sheet_name="Model Setting")
    feature_report.to_excel(writer, sheet_name="Features for Train")

    writer.save()

    estimator.fit(X.values, y)
    joblib.dump(estimator, os.path.join(folder_path, f"dt_model_{file_id}.pkl"))
    print(f"Train report has been saved in '{folder_path}'.")

    _export_meta_data(
        save_path=folder_path,
        meta=meta,
        estimator=estimator,
        file_id=file_id,
        encoder=encoder,
    )


def export_test_report(
    estimator: "trained model object",
    data: pd.DataFrame,
    file_id: str,
    save_path: str,
    target: str,
):
    # data without target
    if target is None:
        pred_report = pd.DataFrame(
            dict(
                Prediction=estimator.predict(data),
                Probability=estimator.predict_proba(data)[:, 1],
            )
        )
        pred_report.index.name = "ID"

        folder_path = os.path.join(save_path, f"prediction_{file_id}")
        create_folder(folder_path)
        writer = pd.ExcelWriter(
            os.path.join(folder_path, f"prediction_{file_id}.xlsx")
        )  # pylint: disable=abstract-class-instantiated
        pred_report.to_excel(writer, sheet_name="Prediction Result")

        writer.save()
        print(f"Test report has been saved in '{folder_path}'.")

    # data with target
    else:
        X = data.drop(target, axis=1)
        y = data[target]
        pred_report = pd.DataFrame(
            dict(
                Observation=y.values.tolist(),
                Prediction=estimator.predict(X),
                Probability=estimator.predict_proba(X)[:, 1],
            )
        )
        pred_report.index.name = "ID"

        test_report = MeasuringTool(
            y_true=y, y_pred=pred_report["Probability"]
        ).get_scores()

        folder_path = os.path.join(save_path, f"prediction_{file_id}")
        create_folder(folder_path)
        writer = pd.ExcelWriter(
            os.path.join(folder_path, f"prediction_{file_id}.xlsx")
        )  # pylint: disable=abstract-class-instantiated
        test_report.to_excel(writer, sheet_name="Test Result", index=False)
        pred_report.to_excel(writer, sheet_name="Prediction Result")

        writer.save()
        print(f"Test report has been saved in '{folder_path}'.")


def read_report(path):
    """
    Read reports which contains trained model and meta data.

    Args
    ------
    path: str, path of meta file

    Return
    ------
    estimator: sklearn estimator object, trained model
    meta: dict, meta data
    """
    path = path.replace("\\", "/")

    # input path point out meta.zip file
    if path.endswith(".zip"):
        meta_path = "/".join(path.split("/")[:-1])
        meta_zip = zipfile.ZipFile(path)
        meta_zip.extractall(meta_path)
        meta_zip.close()
        paths = [
            f
            for f in glob.glob(os.path.join(path, "*"))
            if not os.path.basename(f).endswith(".zip")
        ]

    # input path point out the folder of meta.zip file
    else:
        meta_zip = zipfile.ZipFile(glob.glob(os.path.join(path, "*.zip"))[0])
        meta_zip.extractall(path)
        meta_zip.close()
        paths = [
            f
            for f in glob.glob(os.path.join(path, "*"))
            if not os.path.basename(f).endswith(".zip")
        ]

    for file_path in paths:
        if file_path.endswith(".json") and "meta" in file_path:
            with open(file_path, "r") as json_file:
                meta_raw = json.load(json_file)
        elif file_path.endswith(".pkl") and "model" in file_path:
            estimator = joblib.load(file_path)

    # remove extracted files
    for f in paths:
        try:
            if ("encoder" not in f.split("/")[-1]) and (
                not f.split("/")[-1].endswith(".xlsx")
            ):
                os.remove(f)
        except:
            pass

    # make meta NamedTuple
    meta = Meta(*map(lambda x: x[0], zip(meta_raw.values())))

    return estimator, meta


def generate_file_id():
    return (
        str(datetime.today())
        .replace(":", "")
        .split(".")[0]
        .replace(" ", "")
        .replace("-", "")
    )


def get_estimator_name(estimator):
    return "".join([i for i in list(str(estimator)) if i.isupper()])[:-1].lower()


def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Creating directory. " + path)


def _export_meta_data(
    save_path: str,
    meta: NamedTuple,
    estimator: "trained model object",
    file_id: str,
    encoder: dict = None,
):
    """
    Make zip file which contains meta files after train. It will be used for test phase as a meta data.

    Args
    ------
    path: str, save path
    estimator_name: str, estimator name used to name report file
    file_id: str, report file ID
    """
    # NamedTuple to dictionary to save as a json file
    meta = dict(meta._asdict())
    estimator_name = get_estimator_name(estimator)
    # make json file of encoder
    if encoder is not None:
        encoder_path = os.path.join(
            save_path, f"{estimator_name}_encoder_{file_id}.json"
        )
        meta["encoder_path"] = encoder_path
        with open(encoder_path, "w") as json_file:
            json.dump(encoder, json_file)

    # make json file of meta data
    meta_path = os.path.join(save_path, f"{estimator_name}_meta_{file_id}.json")
    with open(meta_path, "w") as json_file:
        json.dump(meta, json_file)

    # compress all meta files as a zip file
    meta_zip = zipfile.ZipFile(
        os.path.join(save_path, f"{estimator_name}_{file_id}.zip"), "w"
    )
    for folder, _, files in os.walk(save_path):
        for file in files:
            if file.endswith(".json") or file.endswith(".pkl"):
                meta_zip.write(
                    os.path.join(folder, file),
                    os.path.relpath(os.path.join(folder, file), save_path),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
    meta_zip.close()
    paths = [
        f
        for f in glob.glob(f"{save_path}/*")
        if (not os.path.basename(f).endswith(".zip"))
        and (not os.path.basename(f).endswith(".xlsx"))
    ]

    # remove all files except for the zip file
    for f in paths:
        os.remove(f)
