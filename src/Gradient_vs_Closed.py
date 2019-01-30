from Evaluator import Evaluator
import json
from preprocess import Preprocess
import numpy as np
import matplotlib.pyplot as plt

# loading the data
with open("../src/proj1_data.json") as fp:
    data = json.load(fp)

Preprocess.preprocess(data)
preprocess = Preprocess()

x_set = preprocess.matrixify(data, 160)
y_set = preprocess.get_y(data)

error_list_closed = []
error_list_gradient = []

time_list_closed = []
time_list_gradient = []

feature_range = np.arange(3, 160)

for x in feature_range:
    current_x_set = x_set[:, :x]
    print("using " + str(x) + " features")
    time_closed, performance_close = Evaluator.evaluate_closed_form(current_x_set, y_set)
    time_grad, performance_grad = Evaluator.gradient_evaluator(current_x_set, y_set)

    error_list_closed.append(performance_close)
    error_list_gradient.append(performance_grad)

    time_list_closed.append(time_closed)
    time_list_gradient.append(time_grad)

current_x_set = x_set[:, :63]
time_closed, performance_close = Evaluator.evaluate_closed_form(x_set, y_set)
time_grad, performance_grad = Evaluator.gradient_evaluator(x_set, y_set)
print("Gradient, Time: " + str(time_grad) + "MSE: " + str(performance_grad))
print("Closed, Time: " + str(time_closed) + "MSE: " + str(performance_close))

size = 5

plt.subplot(211)
plt.title('MSE vs number of features')
plt.scatter(feature_range, error_list_gradient, s=size, color='blue', label='Gradient descent model')
plt.scatter(feature_range, error_list_closed, s=size, color='red', label='Closed form model')
plt.xlabel("Number of features")
plt.ylabel("MSE (s)")
plt.legend(loc='upper right')

plt.subplot(212)
plt.title('Time taken vs number of features')
plt.scatter(feature_range, time_list_gradient, s=size, color='blue', label='Gradient descent model')
plt.scatter(feature_range, time_list_closed, s=size, color='red', label='Closed model')
plt.xlabel("Number of features")
plt.ylabel("Runtime (s)")
plt.legend(loc='upper left')

plt.show()
