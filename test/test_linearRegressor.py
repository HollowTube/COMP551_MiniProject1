from unittest import TestCase
from LinearRegressor import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt


class TestLinearRegressor(TestCase):
    def setUp(self):

        self.x_base = np.arange(0,20,0.5)
        x_square = np.array([])
        for x in self.x_base:
            x_square = np.append(x_square, (x**2))
        self.x_train = np.vstack([self.x_base,x_square])
        self.x_train = np.transpose(self.x_train)

        noise = 100 * np.random.rand(self.x_train.shape[0])

        self.y_train = noise + 3 * self.x_base + 2*x_square + 5
        self.y_train = self.y_train.reshape((self.x_train.shape[0], 1))

    def test_bias(self):
        regressor = LinearRegressor()
        test = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 2, 3, 1], [4, 5, 6, 1]])
        test = LinearRegressor.add_bias(test)
        self.assertTrue(np.array_equal(test, expected))

    def test_fit(self):
        regressor = LinearRegressor()
        regressor.fit(self.x_train, self.y_train)

        y_pred = regressor.predict(self.x_train)
        plt.scatter(self.x_base, self.y_train, color='blue')
        plt.plot(self.x_base, y_pred,color ='red')
        plt.show()

    def test_fit_gradient(self):
        regressor = LinearRegressor()
        regressor.fit_gradient_descent(self.x_train, self.y_train)

        y_pred = regressor.predict(self.x_train)
        plt.scatter(self.x_base, self.y_train, color='blue')
        plt.plot(self.x_base, y_pred)
        plt.show()
