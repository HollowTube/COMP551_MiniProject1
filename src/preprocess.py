import heapq
import numpy as np

class Preprocess:
    @staticmethod
    def preprocess(data):
        for data_point in data:
            data_point['text'] = data_point['text'].lower().split()
            if data_point['is_root']:
                data_point['is_root'] = 1
            else:
                data_point['is_root'] = 0

    @staticmethod
    def matrixify( data):
        h = {}
        for data_point in data:
            for word in data_point['text']:
                if (word not in h):
                    h[word] = 1
                else:
                    h[word] += 1

        frequent_words = heapq.nlargest(160, h, key=h.get)
        x = []
        for data_point in data:
            row_vector = [0] * 160
            for word in data_point['text']:
                for i in range(len(frequent_words)):
                    if (frequent_words[i] == word):
                        row_vector[i] += 1
            row_vector.append(data_point['is_root'])
            row_vector.append(data_point['controversiality'])
            row_vector.append(data_point['children'])
            x.append(row_vector)
        return np.array(x)
