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

    def test_cross_validator(self):
        self.fail()
