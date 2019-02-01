import matplotlib.pyplot as plt
from Evaluator import Evaluator
from preprocess import Preprocess
from LinearRegressor import LinearRegressor
import numpy as np
import json


def run_model(x_set, y_set):

    # Splitting the dataset
    x_train = x_set[:10000]
    y_train = y_set[:10000]

    x_val = x_set[10000:11000]
    y_val = y_set[10000:11000]

    x_test = x_set[11000:12000]
    y_test = y_set[11000:12000]

    # fitting the model
    regressor = LinearRegressor()
    regressor.fit(x_train, y_train)

    # predicting results with our model with different test sets
    y_pred = regressor.predict(x_val)
    train_y_pred = regressor.predict(x_train)
    test_y_pred = regressor.predict(x_test)

    # Evaluating our error
    mse_train = Evaluator.mean_square_error(train_y_pred, y_train)
    mse_val = Evaluator.mean_square_error(y_pred, y_val)
    mse_test = Evaluator.mean_square_error(test_y_pred, y_test)

    return mse_val, mse_train

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)

Preprocess.preprocess(data)


num_words = 160
word_nums = np.arange(num_words)


preprocess1 = Preprocess()
x_set = preprocess1.matrixify(data, num_words)
y_set = Preprocess.get_y(data)

# computing our performance
val_error_list = []
train_error_list = []
for x in word_nums:
    cur = x_set[:, :3 + x]
    print("Running on top " + str(x) + " words")
    val_error, train_error = run_model(cur, y_set)
    val_error_list.append(val_error)
    train_error_list.append(train_error)

fig, ax = plt.subplots()
plt.scatter(word_nums, val_error_list, color='blue', s=5, label="Validation set")
plt.scatter(word_nums, train_error_list, color='red', s=5, label="Training set")
plt.title("MSE vs number of words used")
ax.set_xlabel("Words Used")
ax.set_ylabel("MSE")
plt.legend(loc='upper right')
plt.show()
