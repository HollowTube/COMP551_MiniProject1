import numpy as np


def gradient(x_train, y_train, weight):
    x_train_transpose = np.transpose(x_train)
    A = x_train_transpose @ x_train @ weight
    B = x_train_transpose @ y_train
    C = A - B
    return 2 * C


def add_bias(matrix):
    num_rows = matrix.shape[0]

    if len(matrix.shape) == 1:
        num_columns = 1
    else:
        num_columns = matrix.shape[1]

    ones = np.ones((num_rows, num_columns + 1))
    ones[:, : -1] = matrix
    return ones


class LinearRegressor:
    def __init__(self):
        self.weights = np.ones(1)
        self.epsilon = 0.01

    """ MAKE SURE THAT THE X_TRAIN IS OF THE FORM M X N WHERE M IS THE NUMBER OF ROWS"""

    def fit(self, x_train, y_train):
        x_train = add_bias(x_train)
        x_train_transpose = np.transpose(x_train)
        A = np.linalg.inv(x_train_transpose @ x_train)
        B = x_train_transpose @ y_train
        self.weights = A @ B

    def fit_gradient_descent(self, x_train, y_train):
        x_train = add_bias(x_train)
        weight = np.ones(x_train.shape[1]).reshape(x_train.shape[1], 1)
        count = 1
        while True:
            alpha = 1 / ((1000 * count**2) + 10000)  # learning rate
            prev_weight = weight
            gradient_value = alpha * gradient(x_train, y_train, weight)
            weight = weight - gradient_value
            count += 1
            if self.large_steps(prev_weight, weight):
                self.weights = weight
                break

    def predict(self, x_test):
        x_test = add_bias(x_test)
        y_pred = x_test @ self.weights
        return y_pred

    def large_steps(self, prev_weight, new_weight):
        difference = new_weight - prev_weight
        return abs(np.linalg.norm(difference)) < self.epsilon
