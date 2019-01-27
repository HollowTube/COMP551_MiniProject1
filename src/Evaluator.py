import preprocess
from LinearRegressor import LinearRegressor
import numpy as np
from timeit import default_timer as timer


class Evaluator:
    @staticmethod
    def mean_square_error(y_pred, y_actual):
        error_sum = 0
        for y_p, y_a in zip(y_pred, y_actual):
            diff = np.linalg.norm(y_p - y_a)
            error_sum += diff ** 2
        return error_sum / y_pred.shape[0]

    @staticmethod
    def hyperparameter_tester(x_train, y_train, initial_rate=None, beta=None):
        regressor = LinearRegressor()
        if initial_rate is not None:
            regressor.initial_rate = initial_rate
        if beta is not None:
            regressor.beta = beta

        start = timer()
        regressor.fit_gradient_descent(x_train, y_train)
        end = timer()
        time_taken = end - start

        y_pred = regressor.predict(x_train)
        mse = Evaluator.mean_square_error(y_pred, y_train)
        return time_taken, mse

    @staticmethod
    def evaluate_closed_form(training_set, validation_set):
        regressor= LinearRegressor()



        pass