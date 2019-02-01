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
    def gradient_evaluator(x_set, y_set, initial_rate=None, beta=None, epsilon = None):

        # splitting the training and validation set
        x_train = x_set[:10000]
        y_train = y_set[:10000]

        x_val = x_set[10000:11000]
        y_val = y_set[10000:11000]

        regressor = LinearRegressor()
        if initial_rate is not None:
            regressor.initial_rate = initial_rate
        if beta is not None:
            regressor.beta = beta
        if epsilon is not None:
            regressor.epsilon = epsilon

        start = timer()
        regressor.fit_gradient_descent(x_train, y_train)
        end = timer()
        time_taken = end - start



        # Predicting  and evaluating on the validation set
        y_pred = regressor.predict(x_val)
        mse = Evaluator.mean_square_error(y_pred, y_val)
        return time_taken, mse

    @staticmethod
    def evaluate_closed_form(x_set, y_set):

        # splitting the training and validation set
        x_train = x_set[:10000]
        y_train = y_set[:10000]

        x_val = x_set[10000:11000]
        y_val = y_set[10000:11000]

        x_test = x_set[11000:12000]
        y_test = y_set[11000:12000]

        regressor = LinearRegressor()

        # Runtime measurement
        start = timer()
        regressor.fit(x_train, y_train)
        end = timer()
        time_taken = end - start

        # Predicting  and evaluating on the validation set
        y_pred = regressor.predict(x_val)
        train_y_pred = regressor.predict(x_train)
        test_y_pred = regressor.predict(x_test)

        mse_train = Evaluator.mean_square_error(train_y_pred, y_train)
        mse_val = Evaluator.mean_square_error(y_pred, y_val)
        mse_test = Evaluator.mean_square_error(test_y_pred, y_test)

        print("validation: " + str(mse_val))
        print("train: " + str(mse_train))
        # print("Test: " + str(mse_test))
        return time_taken, mse_val
