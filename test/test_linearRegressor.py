from unittest import TestCase
from LinearRegressor import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt
from Evaluator import Evaluator

class TestLinearRegressor(TestCase):
    def setUp(self):

        self.x_base = np.arange(0,20,0.5)
        x_square = np.array([])
        for x in self.x_base:
            x_square = np.append(x_square, (x**2))

        self.x_train = np.vstack([self.x_base,x_square])
        self.x_train = np.transpose(self.x_train)

        noise = 100 * np.random.rand(self.x_train.shape[0])

        self.y_train = noise +  2*self.x_base - 4*x_square + 100
        self.y_train = self.y_train.reshape((self.x_train.shape[0], 1))

    def test_bias(self):
        test = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 2, 3, 1], [4, 5, 6, 1]])
        test = LinearRegressor.add_bias(test)
        self.assertTrue(np.array_equal(test, expected))

    def test_fit(self):
        regressor = LinearRegressor()

        regressor.fit(self.x_train, self.y_train)
        print(regressor.weights)

        y_pred = regressor.predict(self.x_train)
        print(Evaluator.mean_square_error(y_pred,self.y_train))

        plt.scatter(self.x_base, self.y_train, color='blue')
        plt.plot(self.x_base, y_pred,color ='red')
        plt.show()

    def test_model(self):
        error_list = []
        time_list = []
        initial_rate = []

        for i in np.arange(5, 9, 0.2):
            initial_rate.append(i * 10 ** (-7))

        for rate in initial_rate:
            print("running with rate " + str(rate))
            result = Evaluator.hyperparameter_tester(self.x_train, self.y_train, initial_rate=rate)
            time_list.append(result[0])
            error_list.append(result[1])

        plt.subplot(211)
        plt.xscale('log')
        plt.title('MSE vs Initial Rate')
        plt.scatter(initial_rate, error_list, color='blue')

        plt.subplot(212)
        plt.xscale('log')
        plt.title('TIme vs Initial Rate')
        plt.scatter(initial_rate, time_list, color='red')
        plt.show()

    def test_fit_gradient(self):
        regressor = LinearRegressor()
        regressor.fit_gradient_descent(self.x_train, self.y_train)
        print(regressor.weights)
        y_pred = regressor.predict(self.x_train)
        print(Evaluator.mean_square_error(y_pred, self.y_train))


        plt.scatter(self.x_base, self.y_train, color='blue')
        plt.plot(self.x_base, y_pred, color ='red')
        plt.show()

