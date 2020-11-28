import os
import sys
import unittest
from typing import NamedTuple

from core.data.load import load
from core.preprocessing.preprocess import auto_preprocess
from core.models.decisiontree import DecisionTree
from core.utils.manage_report import generate_file_id
from core.utils.type_collection import EstimatorTypes, DataTypes
from core.models.train_test import train, test


class Test(unittest.TestCase):
    
    PROBLEM_TYPE = "binary"
    TRAIN = DataTypes.Marketing_NaNs_Train
    TEST = DataTypes.Marketing_NaNs_Test

    TARGET = "insurance_subscribe"
    META = "results/dt_20201028174759"
    SAVE_PATH = "results"
    CV = 3

    def test_load(self):
        data, meta = load(
            input=self.TRAIN.value, target=self.TARGET, problem_type=self.PROBLEM_TYPE
        )

        print(meta, data.shape)
        self.assertEqual(meta.target, self.TARGET)
        self.assertEqual(meta.problem_type, self.PROBLEM_TYPE)

    def test_preprocess(self):
        data, meta = load(
            input=self.TRAIN.value, target=self.TARGET, problem_type=self.PROBLEM_TYPE
        )

        _, encoder = auto_preprocess(data, meta)

        self.assertEqual(len(encoder), len(meta.categorical))

    def test_DecisionTree(self):
        data, meta = load(
            input=self.TRAIN.value, target=self.TARGET, problem_type=self.PROBLEM_TYPE
        )
        data, encoder = auto_preprocess(data=data, meta=meta)

        model = DecisionTree(problem_type=self.PROBLEM_TYPE)
        model.fit(
            data=data,
            meta=meta,
            cv=self.CV,
            save_path=self.SAVE_PATH,
            file_id=generate_file_id(),
            encoder=encoder
        )

    def test_train(self):
        train(
            input=self.TRAIN.value,
            target=self.TARGET,
            problem_type=self.PROBLEM_TYPE,
            save_path=self.SAVE_PATH,
            estimator=EstimatorTypes.DecisionTree,
            cv=self.CV,
        )

    def test_test(self):
        test(input=self.TEST.value, meta_path=self.META, save_path=self.SAVE_PATH)
