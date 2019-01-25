import preprocess
import LinearRegressor
import numpy as np


class Evaluator:
    @staticmethod
    def split(data, num_splits, one_out=0):

        """ Makes it possible to divide the data set evenly, the remainder is added to the test set"""
        length = data.shape[0]
        remainder = length % num_splits
        if remainder != 0:
            divisible_data = data[:-remainder]
            remainder_data = data[-remainder:]
        else:
            divisible_data = data
            remainder_data = None

        splits = np.split(divisible_data, num_splits)

        x_test = splits[one_out]
        if remainder_data is not None:
            x_test = np.vstack((x_test, remainder_data))

        x_train = np.delete(splits, one_out, 0)
        x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])
        return x_train, x_test

    def cross_validator(self):
        pass
