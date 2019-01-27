from unittest import TestCase
from Evaluator import Evaluator
from preprocess import Preprocess
from LinearRegressor import LinearRegressor
import json


with open("../src/proj1_data.json") as fp:
    data = json.load(fp)


class TestEvaluator(TestCase):
    def setUp(self):
        self.test_size = 12000
        self.data = data[:self.test_size]

        self.training_set = data[:10000]
        self.validation_set = data[10000:11000]
        self.testing_set = data[11000:12000]

        preprocess1 = Preprocess()
        preprocess1.preprocess(self.training_set)
        self.x_train = preprocess1.matrixify(self.training_set)
        self.y_train = Preprocess.get_y(self.training_set)

        preprocess2 = Preprocess()
        preprocess2.preprocess(self.validation_set)
        self.x_val = preprocess2.matrixify(self.validation_set)
        self.y_val = Preprocess.get_y(self.validation_set)

    def test_forReal(self):
        regressor = LinearRegressor()
        regressor.fit(self.x_train,self.y_train)
        y_pred = regressor.predict(self.x_val)
        mse = Evaluator.mean_square_error(y_pred, self.y_val)
        print(mse)


    def test_cross_validator(self):
        self.fail()
