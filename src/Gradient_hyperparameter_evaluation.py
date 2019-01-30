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
initial_rate = []

for i in np.arange(5, 9, 0.2):
    initial_rate.append(i * 10 ** (-7))

for rate in initial_rate:
    print("running with rate " + str(rate))
    time, error = Evaluator.gradient_evaluator(x_set, y_set, initial_rate=rate, )
    time_list.append(time)
    error_list.append(error)

plt.subplot(211)
plt.xscale('log')
plt.title('MSE vs Initial Rate')
plt.scatter(initial_rate, error_list, color='blue')

plt.subplot(212)
plt.xscale('log')
plt.title('TIme vs Initial Rate')
plt.scatter(initial_rate, time_list, color='red')
plt.show()