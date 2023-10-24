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
data = pd.read_csv('D:\code\MachineLearning\\btl\\Cellphone.csv')
# data['sex'] = data['sex'].map({'male':1,'female':0})
# data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
# data['region'] = data['region'].map({'northeast':3,'southeast':2,'southwest':1,'northwest':0})
dt_train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)

X_train = dt_train.iloc[:,2:12]
y_train = dt_train.iloc[:,1]
x_test = dt_train.iloc[:,2:12]
y_test = dt_train.iloc[:,1]

y_test = np.array(y_test)

#hồi quy tuyến tính
reg = LinearRegression().fit(X_train, y_train)
f = reg.predict(x_test)

#ridge///
model_2 = Ridge().fit(X_train,y_train)
y_pred2 = model_2.predict(x_test)


#Lasso
model_3 = Lasso().fit(X_train,y_train)
y_pred3 = model_3.predict(x_test)


print("Do chenh lech theo r2 cho hq: %.8f" %r2_score(y_test, f)) #0.88141601
print("Do chenh lech theo nse cho hq: %.8f" %nse(y_test, f)) #0.88105079
print("Do chenh lech theo mae cho hq: %.8f" %mean_absolute_error(y_test, f)) #78811.31600108
print("Do chenh lech theo rmse cho hq: %.8f" %np.sqrt(mean_squared_error(y_test, f))) #98200.91250554

print("\n ridge")
print("Do chenh lech theo r2 cho rid: %.8f" %r2_score(y_test, y_pred2)) #0.88141601
print("Do chenh lech theo nse cho rid: %.8f" %nse(y_test, y_pred2)) #0.88105079
print("Do chenh lech theo mae cho rid: %.8f" %mean_absolute_error(y_test, y_pred2)) #78811.31600108
print("Do chenh lech theo rmse cho rid: %.8f" %np.sqrt(mean_squared_error(y_test, y_pred2))) #98200.91250554

print("\n lasso")
print("Do chenh lech theo r2 cho lasso: %.8f" %r2_score(y_test, y_pred3)) #0.88141601
print("Do chenh lech theo nse cho lasso: %.8f" %nse(y_test, y_pred3)) #0.88105079
print("Do chenh lech theo mae cho lasso: %.8f" %mean_absolute_error(y_test, y_pred3)) #78811.31600108
print("Do chenh lech theo rmse cho lasso: %.8f" %np.sqrt(mean_squared_error(y_test, y_pred3))) #98200.91250554