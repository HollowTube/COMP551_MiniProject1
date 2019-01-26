import preprocess
import LinearRegressor
import numpy as np


class Evaluator:
    @staticmethod
    def mean_square_error(y_pred, y_actual):
        error_sum = 0
        for y_p, y_a in zip(y_pred, y_actual):
            diff = np.linalg.norm(y_p - y_a)
            error_sum += diff ** 2
        return error_sum / y_pred.shape[0]

    def cross_validator(self):
        pass
