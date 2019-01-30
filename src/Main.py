import matplotlib.pyplot as plt
from Evaluator import Evaluator
from preprocess import Preprocess
from LinearRegressor import LinearRegressor
import numpy as np
import json
import math
import statsmodels.formula.api as sm
import feature_selector

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)


class Main():

    def __init__(self):
        self.data = data
        Preprocess.preprocess(self.data)
        self.training_set = data[:10000]
        self.validation_set = data[10000:11000]
        self.testing_set = data[11000:12000]

    def closed_form_model_full_160(self):
        preprocess1 = Preprocess()
        x_set = preprocess1.matrixify(self.data, 160)
        y_set = Preprocess.get_y(self.data)
        return self.run_model(x_set, y_set)

    def closed_form_model_variable(self, num_words):
        preprocess1 = Preprocess()
        x_set = preprocess1.matrixify(self.data, num_words)
        y_set = Preprocess.get_y(self.data)
        return self.run_model(x_set, y_set)

    def closed_form_top_60(self):
        preprocess1 = Preprocess()
        x_set = preprocess1.matrixify(self.data, 60)
        y_set = Preprocess.get_y(self.data)
        return self.run_model(x_set, y_set)

    def closed_form_extra_features(self):
        preprocess1 = Preprocess()

        x_set = preprocess1.matrixify(self.data, 60)
        y_set = Preprocess.get_y(self.data)
        lengths = []
        length_squared = []

        for datapoint in self.data:
            text_length = len(datapoint['text'])
            lengths.append(text_length)

        children_length_inter = []
        children_list = []
        log_children_list = []
        for datapoint in self.data:
            children_list.append(datapoint['children'])
            if datapoint['children'] != 0:
                log_children_list.append(math.log(datapoint['children']))
            else:
                log_children_list.append(0)

        for length, children in zip(lengths, children_list):
            children_length_inter.append(length * children)

        preprocess1.add_features(children_length_inter)
        x_set = preprocess1.add_features(log_children_list)
        x_set = feature_selector.backwardElimination(x_set, y_set, 0.1)
        return self.run_model(x_set, y_set)


    @staticmethod
    def run_model(x_set, y_set):
        x_train = x_set[:10000]
        y_train = y_set[:10000]

        x_val = x_set[10000:11000]
        y_val = y_set[10000:11000]

        x_test = x_set[11000:12000]
        y_test = y_set[11000:12000]

        regressor = LinearRegressor()
        regressor.fit(x_train, y_train)

        y_pred = regressor.predict(x_val)
        train_y_pred = regressor.predict(x_train)
        test_y_pred = regressor.predict(x_test)

        mse_train = Evaluator.mean_square_error(train_y_pred, y_train)
        mse_val = Evaluator.mean_square_error(y_pred, y_val)
        mse_test = Evaluator.mean_square_error(test_y_pred, y_test)

        # print("validation: " + str(mse_val))
        # print("train: " + str(mse_train))
        # print("Test: " + str(mse_test))
        return mse_val, mse_train

    def display_training_and_validation_error(self):
        num_words = 160
        word_nums = np.arange(num_words)
        val_error_list = []
        train_error_list = []
        preprocess1 = Preprocess()
        x_set = preprocess1.matrixify(self.data, num_words)
        y_set = Preprocess.get_y(self.data)
        for x in word_nums:
            cur = x_set[:, 3:3+x]
            print("Running on top " + str(x) + " words")
            val_error, train_error = self.run_model(cur, y_set)
            val_error_list.append(val_error)
            train_error_list.append(train_error)
        fig, ax = plt.subplots()
        plt.scatter(word_nums, val_error_list, color='blue', s=5, label="Validation set")
        plt.scatter(word_nums, train_error_list, color='red',s=5, label="Training set")
        plt.title("MSE vs number of words used")
        ax.set_xlabel("Words Used")
        ax.set_ylabel("MSE")
        plt.legend(loc = 'upper right')
        plt.show()


if __name__ == "__main__":
    # Main().closed_form_model_full_160()
    main = Main()
    main.display_training_and_validation_error()
