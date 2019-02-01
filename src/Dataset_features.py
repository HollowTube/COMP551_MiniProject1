import matplotlib.pyplot as plt
from preprocess import Preprocess
import numpy as np
import json

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)

class Dataset_features:
    @staticmethod
    def print_max_min(data):
        y_set = Preprocess.get_y(data)
        print("Max is " + str(max(y_set)))
        print("Min is " + str(min(y_set)))
        return max(y_set), min(y_set)

    @staticmethod
    def print_stdev(data):
        y_set = Preprocess.get_y(data)
        stdev = np.std(a = y_set)
        print("standard deviation is " + str(stdev))
        return stdev

    @staticmethod
    def display_dataset_histogram(data):
        y_set = Preprocess.get_y(data)
        n, bins, patches = plt.hist(y_set, 100, density=True, facecolor='g', alpha=0.75)
        plt.title("Histogram of the popularity score")
        plt.xlabel("Popularity Score")
        plt.show()

if __name__ == "__main__":
    Dataset_features.display_dataset_histogram(data)



