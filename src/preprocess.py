import heapq
import numpy as np
import re

def add_bias(matrix):
    num_rows = matrix.shape[0]

    """Checking the shape of matrix"""
    if len(matrix.shape) == 1:
        num_columns = 1
    else:
        num_columns = matrix.shape[1]

    out = np.ones((num_rows, num_columns + 1))
    out[:, : -1] = matrix
    return out

class Preprocess:
    def __init__(self):
        self.extra_features = None
        self.feature_set = None
        self.frequent_words = None

    def add_features(self, features):
        length = len(features)
        feature_arr = np.array(features).reshape((length, 1))
        if self.feature_set is not None:
            self.feature_set = np.hstack((self.feature_set, feature_arr))
        else:
            self.feature_set = feature_arr
        return self.feature_set

    @staticmethod
    def preprocess(data):
        for data_point in data:
            data_point['text'] = data_point['text'].lower().split()
            if data_point['is_root']:
                data_point['is_root'] = 1
            else:
                data_point['is_root'] = 0

    @staticmethod
    def preprocess_remove_non_alpha(data):
        for data_point in data:
            data_point['text'] = re.sub(r'[^a-zA-Z]', ' ', data_point['text'])
            data_point['text'] = data_point['text'].lower().split()
            if data_point['is_root']:
                data_point['is_root'] = 1
            else:
                data_point['is_root'] = 0

    def add_base_features(self,data):
        x = []
        for data_point in data:
            row_vector = [data_point['is_root'], data_point['controversiality'], data_point['children'],
                          data_point['children'] * data_point['is_root']]
            x.append(row_vector)
        x = np.array(x)

        if self.feature_set is not None:
            self.feature_set = np.hstack((self.feature_set,x))
        else:
            self.feature_set = x

    def matrixify(self, data, wordCount=0):
        """ Specify wordNumber for the number of most frequent words, ie 0, 60 or 160. """
        h = {}
        if wordCount > 0:
            for data_point in data:
                for word in data_point['text']:
                    if word not in h:
                        h[word] = 1
                    else:
                        h[word] += 1

        self.frequent_words = heapq.nlargest(wordCount, h, key=h.get)
        x = []
        for data_point in data:
            row_vector = [0] * wordCount
            for word in data_point['text']:
                for i in range(len(self.frequent_words)):
                    if self.frequent_words[i] == word:
                        row_vector[i] += 1
            row_vector.append(data_point['is_root'])
            row_vector.append(data_point['controversiality'])
            row_vector.append(data_point['children'])
            row_vector.append(data_point['children'] * data_point['is_root'])
            x.append(row_vector)
        self.feature_set = np.array(x)
        return np.array(x)

    @staticmethod
    def get_y(data):
        y_list = []
        for data_point in data:
            y_list.append(data_point['popularity_score'])
        return np.array(y_list).reshape((len(data), 1))
