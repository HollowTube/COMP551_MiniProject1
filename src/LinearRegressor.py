import numpy as np


class LinearRegressor:
    def __init__(self):
        self.weights = None
        self.epsilon = 0.0001
        self.initial_rate = 1 / 100000000
        self.beta = 0.0001

        self.A = None
        self.B = None

    @staticmethod
    def add_bias(matrix):
        num_rows = matrix.shape[0]

        """Checking the shape of matrix"""
        if len(matrix.shape) == 1:
            num_columns = 1
        else:
            num_columns = matrix.shape[1]

        out = np.ones((num_rows, num_columns + 1))
        out[:, : -1] = matrix
        return out

    # closed form linear regression
    def fit(self, x_train, y_train):
        x_train = LinearRegressor.add_bias(x_train)
        x_train_transpose = np.transpose(x_train)

        A = np.linalg.inv(x_train_transpose @ x_train)
        B = x_train_transpose @ y_train
        self.weights = A @ B

    # Gradient descent linear regression
    def fit_gradient_descent(self, x_train, y_train):
        x_train = LinearRegressor.add_bias(x_train)
        weight = np.zeros(x_train.shape[1]).reshape(x_train.shape[1], 1)
        count = 1
        self.A = None
        self.B = None
        while True:
            alpha = self.learning_rate(count)  # learning rate
            prev_weight = weight
            gradient_value = alpha * self.gradient(x_train, y_train, weight)
            weight = weight - gradient_value
            count += 1
            if self.check_threshold(prev_weight, weight):
                self.weights = weight
                break

    # predicts the data with the given weights
    def predict(self, x_test):
        x_test = LinearRegressor.add_bias(x_test)
        y_pred = x_test @ self.weights
        return y_pred

    def gradient(self, x_train, y_train, weight):
        if self.A is None or self.B is None:
            x_train_transpose = np.transpose(x_train)
            self.A = 2 * (x_train_transpose @ x_train)
            self.B = 2 * (x_train_transpose @ y_train)
        return self.A @ weight - self.B

    def learning_rate(self, step):
        return self.initial_rate / (1 + self.beta * step)

    def check_threshold(self, prev_weight, new_weight):
        difference = new_weight - prev_weight
        return abs(np.linalg.norm(difference)) < self.epsilon
