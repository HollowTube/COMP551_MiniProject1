from unittest import TestCase
from preprocess import Preprocess

import json

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)


class TestMatrixify(TestCase):
    def setUp(self):
        self.test_size = 10
        self.data = data[:self.test_size]

    def test_splitter(self):
        preprocessor = Preprocess()
        preprocessor.preprocess(self.data)
        num_splits = 5
        x_set = preprocessor.matrixify(self.data)
        x_train, x_test = Preprocess.split(x_set, num_splits)
        self.assertEqual(x_test.shape[0], self.expected_split(num_splits))
        self.assertEqual(x_train.shape[0], self.test_size - (self.expected_split(num_splits)))

    def expected_split(self, num_splits):
        return self.test_size // num_splits + self.test_size % num_splits
    def test_matrixify(self):
        preprocessor = Preprocess()
        print(self.data)
        preprocessor.preprocess(self.data)
        x_set = preprocessor.matrixify(self.data)
        self.assertEqual(163, x_set.shape[1])

    def test_preprocess(self):
        preprocessor = Preprocess()
        preprocessor.preprocess(self.data)
        for point in self.data:
            self.assertTrue((point['controversiality'] == 0) or (point['controversiality'] == 1))
            self.assertEqual(len(point), 5)
            for word in point['text']:
                try:
                    self.assertTrue(not word.isalpha() or word.islower())
                except AssertionError:
                    print(word)






