import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import recall_score, precision_score


data = pd.read_csv('ketthucmon\Iris.csv')

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

X = data.iloc[:, 0:3]  
y = data.iloc[:, 4] 
id3 = DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=54)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=None)
precisions = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    id3.fit(X_train, y_train)
    
    y_pred = id3.predict(X_test)

    precision = precision_score(y_test, y_pred, average='micro')
    precisions.append(precision)

# Tính độ chính xác trung bình qua các lượt Cross-Validation
avr_accuracy = sum(precisions) / k

print("Độ chính xác trung bình:", avr_accuracy)