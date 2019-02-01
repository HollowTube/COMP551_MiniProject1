import heapq
import numpy as np
import re
import math


class Preprocess:
    def __init__(self):
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

    # Splits up our words  and puts them in lower case as well as encoding the 'is root' property
    @staticmethod
    def preprocess(data):
        for data_point in data:
            data_point['text'] = data_point['text'].lower().split()
            if data_point['is_root']:
                data_point['is_root'] = 1
            else:
                data_point['is_root'] = 0

    # returns a list of the log children features
    @staticmethod
    def log_children(data):
        children_list = []
        log_children_list = []
        for datapoint in data:
            children_list.append(datapoint['children'])
            if datapoint['children'] != 0:
                log_children_list.append(math.log(datapoint['children']))
            else:
                log_children_list.append(0)
        return log_children_list


    # returns a list of children and text length interaction
    @staticmethod
    def children_length_interaction(data):
        lengths = []
        children_list = []
        children_length_inter = []
        for datapoint in data:
            text_length = len(datapoint['text'])
            children_list.append(datapoint['children'])
            lengths.append(text_length)

        for length, children in zip(lengths, children_list):
            children_length_inter.append(length * children)
        return children_length_inter

    @staticmethod
    def preprocess_remove_non_alpha(data):
        for data_point in data:
            data_point['text'] = re.sub(r'[^a-zA-Z]', ' ', data_point['text'])
            data_point['text'] = data_point['text'].lower().split()
            if data_point['is_root']:
                data_point['is_root'] = 1
            else:
                data_point['is_root'] = 0

    def add_base_features(self, data):
        x = []
        for data_point in data:
            row_vector = [data_point['is_root'], data_point['controversiality'], data_point['children'],
                          data_point['children'] * data_point['is_root']]
            x.append(row_vector)
        x = np.array(x)

        if self.feature_set is not None:
            self.feature_set = np.hstack((self.feature_set, x))
        else:
            self.feature_set = x


    # adds the base features with a variable amount of word features
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

        # sorts the list according to frequency
        self.frequent_words = heapq.nlargest(wordCount, h, key=h.get)

        x = []
        for data_point in data:
            row_vector = np.zeros((3 + wordCount))

            row_vector[0] = data_point['is_root']
            row_vector[1] = data_point['controversiality']
            row_vector[2] = data_point['children']

            for word in data_point['text']:
                for i in range(len(self.frequent_words)):
                    if self.frequent_words[i] == word:
                        row_vector[i + 3] += 1
            x.append(row_vector)
        self.feature_set = np.array(x)
        return self.feature_set

    @staticmethod
    def get_y(data):
        y_list = []
        for data_point in data:
            y_list.append(data_point['popularity_score'])
        return np.array(y_list).reshape((len(data), 1))

    @staticmethod
    def export_word_count(data):
        h = {}
        for data_point in data:
            for word in data_point['text']:
                if word not in h:
                    h[word] = 1
                else:
                    h[word] += 1

        frequent_words = heapq.nlargest(160, h, key=h.get)
        f = open('words.txt', 'w')
        for word in frequent_words:
            f.write(str(word) + "\n")
        f.close()


