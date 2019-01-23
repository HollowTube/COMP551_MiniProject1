import numpy as np


class LinearRegressor:
    def __init__(self):
        self.weights = np.ones(1)

    def add_bias(self, matrix):
        num_rows = matrix.shape[0]

        if len(matrix.shape) == 1:
            num_columns = 1
        else:
            num_columns = matrix.shape[1]

        ones = np.ones((num_rows, num_columns + 1))
        ones[:, : -1] = matrix
        return ones

    """ MAKE SURE THAT THE X_TRAIN IS OF THE FORM M X N WHERE M IS THE NUMBER OF ROWS"""
    def fit(self, x_train, y_train):
        x_train = self.add_bias(x_train)
        x_train_transpose = np.transpose(x_train)
        A = np.linalg.inv(x_train_transpose @ x_train)
        B = x_train_transpose @ y_train
        self.weights = A @ B

    def predict(self, x_test):
        x_test = self.add_bias(x_test)
        y_pred = x_test @ self.weights
        return y_pred
