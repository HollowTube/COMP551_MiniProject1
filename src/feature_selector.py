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
