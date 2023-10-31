import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv('ketthucmon\Iris.csv')

le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)
dt_train, dt_Test = train_test_split(df, test_size = 0.3, shuffle = True)

x_train = dt_train [:,0:3]
y_train = dt_train [:,4]
x_test = dt_Test [:,0:3]
y_test = dt_Test [:,4]

id3 = tree.DecisionTreeClassifier(criterion='gini', max_depth=8,min_samples_split=54)
id3.fit(x_train, y_train)

y_pre = id3.predict(x_test)
c = 0
for i in range(0, len(y_pre)):
    if(y_test[i] == y_pre[i]):
        c = c + 1
print('ty le du doan dung: ', c/len(y_pre))