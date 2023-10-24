import numpy as np
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# data = pd.read_csv('D:\code\MachineLearning\\btl\\Data.csv')
# dt_train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)
# X_train = dt_train.iloc[:,1:4]
# y_train = dt_train.iloc[:,0]
# x_test = dt_train.iloc[:,1:4]
# y_test = dt_train.iloc[:,0]

# data = pd.read_csv('D:\code\MachineLearning\\btl\\heart_disease.csv')
data = pd.read_csv('D:\code\MachineLearning\\btl\\medical_cost.csv')
data['sex'] = data['sex'].map({'male':1,'female':0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].map({'northeast':3,'southeast':2,'southwest':1,'northwest':0})
dt_train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)

X_train = dt_train.iloc[:,1:6]
y_train = dt_train.iloc[:,7]
x_test = dt_train.iloc[:,1:6]
y_test = dt_train.iloc[:,7]

y_test = np.array(y_test)

# ham r2
def r2(actual, predicted):
    mean_actual = sum(predicted) / len(predicted)
    pre_actual = sum(actual) / len(actual)
    sse = []
    sst = []
    ssi = []
    for i in range(len(actual)):
        mean = actual[i]
        pre = predicted[i]
        sse.append((mean - mean_actual)*(pre - pre_actual))
        sst.append((mean - mean_actual) ** 2)
        ssi.append((pre - pre_actual) ** 2)
    return ((sum(sse)) /(math.sqrt(sum(sst) * sum(ssi))))**2

#ham nse
def nse(actual, predicted):
    mean_actual = sum(predicted) / len(predicted)
    sse = []
    sst = []
    for i in range(len(actual)):
        mea = actual[i]
        pre = predicted[i]
        sse.append((mea - pre) ** 2)
        sst.append((mea - mean_actual) ** 2)
    return 1 - ((sum(sse)) /(sum(sst)))

#ham mae
def mae(actual, predicted):
    sse = []
    for i in range(len(actual)):
        mea = actual[i]
        pre = predicted[i]
        sse.append(abs(mea - pre))
    return sum(sse)/len(actual)

#ham rmse
def rmse(actual, predicted):
    sse = []
    for i in range(len(actual)):
        mea = actual[i]
        pre = predicted[i]
        sse.append(pow((mea - pre),2))
    return math.sqrt(sum(sse)/len(actual))
reg = LinearRegression().fit(X_train, y_train)
f = reg.predict(x_test)
# print("Ket qua mo hinh LinearRegression: ",f[:5])
# print("Do chenh lech theo r2: %.8f" %r2(y_test, f)) #0.88141601
# print("Do chenh lech theo nse: %.8f" %nse(y_test, f)) #0.88105079
# print("Do chenh lech theo mae: %.8f" %mae(y_test, f)) #78811.31600108
# print("Do chenh lech theo rmse: %.8f" %rmse(y_test, f)) #98200.91250554

print("Do chenh lech theo r2: %.8f" %r2_score(y_test, f)) #0.88141601
print("Do chenh lech theo nse: %.8f" %nse(y_test, f)) #0.88105079
print("Do chenh lech theo mae: %.8f" %mean_absolute_error(y_test, f)) #78811.31600108
print("Do chenh lech theo rmse: %.8f" %np.sqrt(mean_squared_error(y_test, f))) #98200.91250554