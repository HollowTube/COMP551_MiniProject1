from Evaluator import Evaluator
from preprocess import Preprocess
import json
import feature_selector

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)


preprocess1 = Preprocess()

Preprocess.preprocess(data)

num_words = 60

preprocess1.matrixify(data, num_words)
y_set = Preprocess.get_y(data)

children_length_inter = preprocess1.children_length_interaction(data)
log_children_list = preprocess1.log_children(data)


preprocess1.add_features(children_length_inter)
preprocess1.add_features(log_children_list)

x_set = preprocess1.feature_set
x_optimal = feature_selector.backwardElimination(x_set,y_set,0.15)
time, mse = Evaluator.evaluate_closed_form(x_optimal, y_set)
print(mse)
print(time)
