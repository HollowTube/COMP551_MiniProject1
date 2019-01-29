import matplotlib.pyplot as plt
from Evaluator import Evaluator
from preprocess import Preprocess
from LinearRegressor import LinearRegressor
import numpy as np
import json
import math

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

        x_set = preprocess1.matrixify(self.data,60)
        y_set = Preprocess.get_y(self.data)
        lengths = []
        length_squared = []


        for datapoint in self.data:
            text_length = len(datapoint['text'])
            lengths.append(text_length)
            length_squared.append(text_length ** 2)

        children_length_inter = []
        children_length_inter_log = []
        children_list = []
        for datapoint in self.data:
            children_list.append(datapoint['children'])

        for length, children in zip(lengths, children_list):
            children_length_inter.append(length * children)

        for length, children in zip(lengths, children_list):
            children_length_inter_log.append(math.log(length, 2) * children)
        x_set = preprocess1.add_features(children_length_inter)
        return self.run_model(x_set, y_set)

    @staticmethod
    def run_model(x_set, y_set):
        x_train = x_set[:10000]
        y_train = y_set[:10000]

        x_val = x_set[10000:11000]
        y_val = y_set[10000:11000]

        regressor = LinearRegressor()
        regressor.fit(x_train, y_train)

        y_pred = regressor.predict(x_val)
        train_y_pred = regressor.predict(x_train)
        mse_train = Evaluator.mean_square_error(train_y_pred, y_train)
        mse_val = Evaluator.mean_square_error(y_pred, y_val)

        print(mse_val)
        return mse_val, mse_train

    def display_training_and_validation_error(self):
        word_nums = np.arange(70)
        val_error_list = []
        train_error_list = []
        for x in word_nums:
            print("Running on top " + str(x) + " words")
            val_error, train_error = self.closed_form_model_variable(x)
            val_error_list.append(val_error)
            train_error_list.append(train_error)
        fig, ax = plt.subplots()
        plt.scatter(word_nums, val_error_list, color='blue')
        plt.scatter(word_nums, train_error_list, color='red')
        plt.title("MSE vs number of words used")
        ax.set_xlabel("Words Used")
        ax.set_ylabel("MSE")
        plt.show()


if __name__ == "__main__":
    # Main().closed_form_model_full_160()
    main = Main()
    main.closed_form_extra_features()
