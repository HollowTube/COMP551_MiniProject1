import statsmodels.formula.api as sm
import json
from preprocess import Preprocess

with open("../src/proj1_data.json") as fp:
    data = json.load(fp)
    Preprocess.preprocess(data)

preprocess1 = Preprocess()
X = preprocess1.matrixify(data, 60)
y = Preprocess.get_y(data)

# lengths = []
# length_squared = []
#
# for datapoint in data:
#     text_length = len(datapoint['text'])
#     lengths.append(text_length)
#     length_squared.append(text_length ** 2)
#
# children_length_inter = []
# children_length_inter_log = []
# children_list = []
# for datapoint in data:
#     children_list.append(datapoint['children'])
#
# for length, children in zip(lengths, children_list):
#     children_length_inter.append(length * children)
#
# X = preprocess1.add_features(children_length_inter)

X_opt = X
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
