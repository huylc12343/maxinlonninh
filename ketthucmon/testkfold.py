import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import precision_score
from sklearn import preprocessing

data = pd.read_csv('ketthucmon\Iris.csv')

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data = np.array(data)

X = data[:, 0:3]  # Chọn cột 0, 1, và 2 làm features
y = data[:, 4]    # Chọn cột 4 làm target
precisions = []
id3 = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=54)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=None)

for train_index, validation_index in kf.split(X):
    X_train, x_test = X[train_index], X[validation_index]
    y_train, y_test = y[train_index], y[validation_index]

    id3.fit(X_train, y_train)
    y_pred = id3.predict(x_test)

    precision = precision_score(y_test, y_pred, average='micro')
    precisions.append(precision)

avr_accuracy = sum(precisions) / k
print("Độ chính xác trung bình:", avr_accuracy)
