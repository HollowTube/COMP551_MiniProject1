import statsmodels.formula.api as sm
import json
from preprocess import Preprocess
import numpy as np


def backwardElimination(x,y, sl):
    numVars = len(x[0])
    regressor_OLS = None
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x


with open("../src/proj1_data.json") as fp:
    data = json.load(fp)
    Preprocess.preprocess(data)

preprocess1 = Preprocess()
X = preprocess1.matrixify(data, 60)
y = Preprocess.get_y(data)

SL = 0.1
X_opt = X
X_Modeled = backwardElimination(X_opt, y, SL)

regressor_OLS = sm.OLS(endog=y, exog=X_Modeled).fit()

print(regressor_OLS.summary())
