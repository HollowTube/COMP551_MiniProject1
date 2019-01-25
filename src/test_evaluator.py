from unittest import TestCase
from Evaluator import Evaluator
from preprocess import Preprocess
import json

with open("proj1_data.json") as fp:
    data = json.load(fp)


class TestEvaluator(TestCase):
    def setUp(self):
        self.test_size = 100
        self.data = data[:self.test_size]

    def test_split(self):
        preprocessor = Preprocess()
        preprocessor.preprocess(self.data)
        num_splits = 5
        x_set = preprocessor.matrixify(self.data)
        x_train, x_test = Evaluator.split(x_set, num_splits)
        self.assertEqual(x_test.shape[0], self.expected_split(num_splits))
        self.assertEqual(x_train.shape[0], self.test_size - (self.expected_split(num_splits)))

    def expected_split(self, num_splits):
        return self.test_size // num_splits + self.test_size % num_splits

    def test_cross_validator(self):
        self.fail()
