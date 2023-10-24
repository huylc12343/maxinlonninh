import numpy as np
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('D:\code\MachineLearning\\btl\\medical_cost.csv')
data['sex'] = data['sex'].map({'male':1,'female':0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].map({'northeast':3,'southeast':2,'southwest':1,'northwest':0})
dt_train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)

X_train = dt_train.iloc[:,1:6]
y_train = dt_train.iloc[:,7]
X_test = dt_train.iloc[:,1:6]
y_test = dt_train.iloc[:,7]

y_test = np.array(y_test)

reg = LinearRegression().fit(X_train,y_train)

y_pred = reg.predict(X_test)
y = np.array(y_test)
print(("Coefficient of determination: %.2f"%r2_score(y_test, y_pred)))

print("Thuc te Du doan Chenh lech")
for i in range (0,len(y)):
    print(("%2f"%y[i],y_pred[i],abs(y[i]-y_pred[i])))