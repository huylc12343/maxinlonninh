import pandas as pd
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, precision_score
from sklearn import preprocessing

data = pd.read_csv('ketthucmon\mushrooms.csv')
# Loading the dataset
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data = np.array(data)

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

# Tính lỗi, y thực tế, y_pred: dữ liệu dự đoán
def error(y, y_pred):
    sum_error = np.sum(np.abs(y - y_pred))
    return sum_error / len(y)  # Trả về trung bình

k = 5
kf = KFold(n_splits=k, random_state=None)
for train_index, validation_index in kf.split(dt_Train):
    X_train, X_validation = dt_Train[train_index, 1:], dt_Train[validation_index, 1:]
    y_train, y_validation = dt_Train[train_index, 0], dt_Train[validation_index, 0]

    id3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=54)
    id3.fit(X_train, y_train)
    y_train_pred = id3.predict(X_train)
    y_validation_pred = id3.predict(X_validation)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)

    sum_error = error(y_train, y_train_pred) + error(y_validation, y_validation_pred)

y_test = np.array(dt_Test[:, 0])
y_pred = id3.predict(dt_Test[:, 1:])

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Recall trên tập kiểm tra:", recall)
print("Precision trên tập kiểm tra:", precision)
