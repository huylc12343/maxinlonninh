from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd 
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('data_main.csv')
# df['buying'] = df['buying'].map({'high':0,'vhigh':1})
# df['maint'] = df['maint'].map({'high': 0, 'vhigh': 1, 'med':2,'low':3})
# df['doors'] = df['doors'].map({'2':0,'3':1,'4':2,'5more':3})
# df['persons'] = df['persons'].map({'2':0,'3':1,'4':2,'more':3})
# df['lug_boot'] = df['lug_boot'].map({'big':0,'med':1,'small':2})
# df['safety'] = df['safety'].map({'high':0,'med':1,'low':2})
# df['acceptability'] = df['acceptability'].map({'unacc':1,'acc':-1})

df = df.dropna()
# print(df)
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)
dt_train, dt_Test = train_test_split(df, test_size = 0.3, shuffle = True)

# print(df)
X_train = dt_train[:,1:11]
y_train = dt_train[:,11]
x_test = dt_Test[:,1:11]
y_test = dt_Test[:,11]


pla = DecisionTreeClassifier()
pla.fit(X_train, y_train)

y_pre = pla.predict(x_test)

c = 0
for i in range(0, len(y_pre)):
    if(y_test[i] == y_pre[i]):
        c = c + 1
print('ty le du doan dung: ', c/len(y_pre))