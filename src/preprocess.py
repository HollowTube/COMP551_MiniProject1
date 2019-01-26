import heapq
import numpy as np


class Preprocess:
    @staticmethod
    def split(data, num_splits, one_out=0):

        """ Makes it possible to divide the data set evenly, the remainder is added to the test set"""
        length = data.shape[0]
        remainder = length % num_splits
        if remainder != 0:
            divisible_data = data[:-remainder]
            remainder_data = data[-remainder:]
        else:
            divisible_data = data
            remainder_data = None

        splits = np.split(divisible_data, num_splits)

        x_test = splits[one_out]
        if remainder_data is not None:
            x_test = np.vstack((x_test, remainder_data))

        x_train = np.delete(splits, one_out, 0)
        x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])
        return x_train, x_test
    @staticmethod
    def preprocess(data):
        for data_point in data:
            data_point['text'] = data_point['text'].lower().split()
            if data_point['is_root']:
                data_point['is_root'] = 1
            else:
                data_point['is_root'] = 0

    @staticmethod
    def matrixify(data):
        h = {}
        for data_point in data:
            for word in data_point['text']:
                if word not in h:
                    h[word] = 1
                else:
                    h[word] += 1

        frequent_words = heapq.nlargest(160, h, key=h.get)
        x = []
        for data_point in data:
            row_vector = [0] * 160
            for word in data_point['text']:
                for i in range(len(frequent_words)):
                    if frequent_words[i] == word:
                        row_vector[i] += 1
            row_vector.append(data_point['is_root'])
            row_vector.append(data_point['controversiality'])
            row_vector.append(data_point['children'])
            x.append(row_vector)
        return np.array(x)
