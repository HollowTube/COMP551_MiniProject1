from unittest import TestCase
from Evaluator import Evaluator
from preprocess import Preprocess
from LinearRegressor import LinearRegressor
import json

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)


class Main():

    def __init__(self):
        self.data = data
        self.training_set = data[:10000]
        self.validation_set = data[10000:11000]
        self.testing_set = data[11000:12000]

    def closed_form_model_full_160(self):
        preprocess1 = Preprocess()
        preprocess1.preprocess(self.data)
        x_set = preprocess1.matrixify(self.data)
        y_set = Preprocess.get_y(self.data)

        x_train = x_set[:10000]
        y_train = y_set[:10000]

        x_val = x_set[10000:11000]
        y_val = y_set[10000:11000]

        regressor = LinearRegressor()
        regressor.fit(x_train, y_train)

        y_pred = regressor.predict(x_val)
        mse = Evaluator.mean_square_error(y_pred, y_val)
        print(mse)


    def closed_form_top_60(self):
        pass

    def closed_form_extra_features(self):
        pass


if __name__ == "__main__":
    Main().closed_form_model_full_160()

