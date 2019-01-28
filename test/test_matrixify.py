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

    def test_remove_non_alpha(self):
        preprocessor = Preprocess()
        preprocessor.preprocess_remove_non_alpha(self.data)
        for point in self.data:
            for word in point['text']:
                try:
                    self.assertTrue(word.isalpha())
                except AssertionError:
                    print(word)

    def test_add_features(self):
        preprocessor = Preprocess()
        preprocessor.preprocess(self.data)
        x_set = preprocessor.matrixify(self.data)
        new_feature = []
        other_feature = []
        for some_feature in self.data:
            new_feature.append(5)
        for some_other_feature in self.data:
            other_feature.append(3)
        x_set = preprocessor.add_features(new_feature)
        x_set = preprocessor.add_features(other_feature)
        self.assertEqual(x_set.shape,(self.test_size, 165))

