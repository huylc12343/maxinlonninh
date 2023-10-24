import numpy as numpy
from sklear.linear_model import LinearRegression
X_train = np.array([[60,2,10],[40,2,5],[100,3,7]])
Y_train = np.array([[10,12,20]])
reg = LinearRegression(fit_intercept = False)