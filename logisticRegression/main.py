import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


df = pd.read_csv('logisticRegression\cars.csv')
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)

dt_Train,dt_Test = train_test_split(df,test_size = 0.3,shuffle = True)
# print(dt_Train)
X_train = dt_Train[:,1:7]
y_train = dt_Train[:,7]
X_test = dt_Test[:,1:7]
y_test = dt_Test[:,7]

clf = LogisticRegression(penalty='l2',solver='newton-cholesky',tol = 0.0001,random_state=300,C=0.5,max_iter=900)
clf.fit(X_train,y_train)
# clf.predict_proba(X_test)
# clf.predict(X_test)
# clf.score(X_test,y_test)
y_predict = clf.predict(X_test)
count = 0
for i in range(0,len(y_predict)):
    if(y_test[i]==y_predict[i]):
        count = count +1
print('Ty le du doan dung:',count/len(y_predict))