import pandas as pd
from sklearn.model_selection import KFold
from sklearn import tree
import numpy as np
from sklearn.metrics import precision_score
from sklearn import preprocessing

data = pd.read_csv('ketthucmon\Iris.csv')

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data = np.array(data)

# dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)
dt_Train = data[:,0:3]
dt_Test = data[:,4]
precisions = []
id3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=54)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=None)

for train_index, validation_index in kf.split(dt_Train):
    X_train,x_test = dt_Train[train_index],dt_Train[validation_index]
    y_train,y_test = dt_Train[train_index],dt_Train[validation_index]

    id3.fit(X_train,y_train)
    y_pred = id3.predict(x_test)

    precision = precision_score(y_test, y_pred, average='micro')
    precisions.append(precision)

avr_accuracy = sum(precisions)/k
print("Độ chính xác trung bình:", avr_accuracy)