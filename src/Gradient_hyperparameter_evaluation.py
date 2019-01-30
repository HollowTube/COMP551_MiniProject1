from Evaluator import Evaluator
import json
from preprocess import Preprocess
import numpy as np
import matplotlib.pyplot as plt

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)

Preprocess.preprocess(data)
preprocess = Preprocess()

x_set = preprocess.matrixify(data, 160)
y_set = preprocess.get_y(data)

error_list = []
time_list = []
initial_rate = np.arange(1, 20, 0.5) * 10 ** -7
epsilon = np.arange(1,100,1) * 10 ** -5

# for rate in initial_rate:
#     print("running with rate " + str(rate))
#     time, error = Evaluator.gradient_evaluator(x_set, y_set, initial_rate=rate, )
#     time_list.append(time)
#     error_list.append(error)

for rate in epsilon:
    print("running with epsilon " + str(rate))
    time, error = Evaluator.gradient_evaluator(x_set, y_set, epsilon=rate, )
    time_list.append(time)
    error_list.append(error)

plt.subplot(211)
plt.title('MSE vs Initial Rate')
plt.scatter(epsilon, error_list, color='blue')

plt.subplot(212)
plt.title('TIme vs Initial Rate')
plt.scatter(epsilon, time_list, color='red')
plt.show()
