# import numpy as np
# import pandas as pd 
# from sklearn.linear_model import Perceptron
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# import random

# w0 = random.random()

# df = pd.read_csv('water_potability.csv')
# # df['buying'] = df['buying'].map({'high':0,'vhigh':1})
# # df['maint'] = df['maint'].map({'high': 0, 'vhigh': 1, 'med':2,'low':3})
# # df['doors'] = df['doors'].map({'2':0,'3':1,'4':2,'5more':3})
# # df['persons'] = df['persons'].map({'2':0,'3':1,'4':2,'more':3})
# # df['lug_boot'] = df['lug_boot'].map({'big':0,'med':1,'small':2})
# # df['safety'] = df['safety'].map({'high':0,'med':1,'low':2})
# # df = df.fillna(0)
# df = df.dropna()
# # print(df)
# # le = preprocessing.LabelEncoder()
# # df = df.apply(le.fit_transform)
# # so_hang = df.shape[0]

# # In số hàng ra màn hình
# # print("Số hàng trong file CSV:", so_hang)
# df = np.array(df)
# dt_train, dt_Test = train_test_split(df, test_size = 0.3, shuffle = True)

# # print(df)
# X_train = dt_train[:,:9]
# y_train = dt_train[:,9]
# x_test = dt_Test[:,:9]
# y_test = dt_Test[:,9]

# X_train = np.insert(X_train, 0, 1, axis = 0) 
# x_test = np.insert(x_test, 0, 1, axis = 0) 

# def check(w, x, y):
#     if np.linalg.pinv(w)@x != y:
#         return False
#     else:
#         return True
# def stop(x_train, y_train, w):
#     for x in x_train:
#         for y in y_train:
#             if check(w,x,y) == False:
#                 return False, x, y
#             else:
#                 return True
# w = w0
# result, x, y = stop(X_train, y_train, w)

# while (result==False):
#     w = w + 0.0001*x*y
# print(w)

# # w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
# # y_test_hat = x_test.T@w
# # print('Giá trị dự đoán mẫu mới: ', y_test_hat)
# # print('Giá trị dự đoán tập huấn luyện: ', X_train.T@w)

import numpy as np
from sklearn import preprocessing
import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_main.csv')

# df['Potability'] = df['Potability'].map({0:-1, 1:1 })
df = df.dropna()

# le = preprocessing.LabelEncoder()
# df = df.apply(le.fit_transform)
df = np.array(df)
dt_train, dt_Test = train_test_split(df, test_size = 0.3, shuffle = False)

# print(df)
X_train = dt_train[:,1:11]
y_train = dt_train[:,11]
x_test = dt_Test[:,1:11]
y_test = dt_Test[:,11]


X_train = np.insert(X_train, 0, 1, axis = 1) 
x_test = np.insert(x_test, 0, 1, axis = 1) 

# print(y_test)

# print(X_train)
# Khởi tạo trọng số ban đầu với giá trị ngẫu nhiên
w = np.random.rand(X_train.shape[1])


# Tính learning rate (tốc độ học)
learn_rate = 0.1
# Hàm kiểm tra dự đoán
def check(x, y, w):
    if np.dot(w, x) < 0:
        yi = 0
    else:
        yi = 1
    return yi == y

# Hàm kiểm tra điều kiện dừng
def stop(X_train, y_train, w):
    for xi, yi in zip(X_train, y_train):
        if check(xi, yi, w) == False:
            return False, xi, yi
    return True, None, None  # Trả về True và giá trị None cho x và y

result, x, y = stop(X_train, y_train, w)

for i in range(1000):
    if result == False: 
        w = w +  learn_rate * y*x
        result, x, y = stop(X_train, y_train, w)  
        # print(w)
y_pre = []
for xi in x_test:
    yi = np.dot(w, xi)
    if(yi<0):
        yi = 0
    else:
        yi = 1
    y_pre.append(yi)
c = 0
for i in range(0, len(y_test)):
    if(y_test[i] == y_pre[i]):
        c = c + 1
print('ty le du doan dung: ', c/len(y_pre))
# print(y_test)
# print(y_test)
# print(y_pre)