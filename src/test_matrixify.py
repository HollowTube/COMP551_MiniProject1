from unittest import TestCase
from preprocess import Preprocess
import json

with open("proj1_data.json") as fp:
    data = json.load(fp)


class TestMatrixify(TestCase):
    def setUp(self):
        self.data = data

    def test_matrixify(self):
        preprocessor = Preprocess()
        preprocessor.preprocess(self.data)
        x_set = preprocessor.matrixify(self.data)
        self.assertEqual(163, x_set.shape[0])

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






