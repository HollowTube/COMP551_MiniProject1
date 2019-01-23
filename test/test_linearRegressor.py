from unittest import TestCase
from LinearRegressor import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt


class TestLinearRegressor(TestCase):
    def test_bias(self):
        regressor = LinearRegressor()
        test = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 2, 3, 1], [4, 5, 6, 1]])
        test = LinearRegressor.add_bias(test)
        self.assertTrue(np.array_equal(test, expected))

    def test_fit(self):
        regressor = LinearRegressor()
        x_base = np.arange(0,20,0.5)
        x_square =[]
        for x in x_base:
            x_square.append(x**2)
        x_train = np.vstack([x_base,x_square])
        if len(x_train.shape) == 1:
            x_train = x_train.reshape((x_train.shape[0], 1))
        else:
            x_train = np.transpose(x_train)

        y_train = np.random.rand(x_train.shape[0], 1)
        y_train = y_train + x_base
        regressor.fit(x_train, y_train)

        y_pred = regressor.predict(x_train)
        plt.scatter(x_train, y_train, color='blue')
        plt.plot(x_train, y_pred)
        plt.show()

    def test_fit_gradient(self):
        regressor = LinearRegressor()
        x_train = np.arange(0,20,0.5)
        x_train = x_train.reshape((x_train.shape[0],1))
        y_train = np.random.rand(x_train.shape[0], 1)
        y_train = y_train + x_train
        regressor.fit_gradient_descent(x_train, y_train)

        y_pred = regressor.predict(x_train)
        plt.scatter(x_train, y_train, color='blue')
        plt.plot(x_train, y_pred)
        plt.show()
