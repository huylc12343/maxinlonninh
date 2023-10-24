import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

df = pd.read_csv('neuron\cars.csv')
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)

dt_Train,dt_Test = train_test_split(df,test_size = 0.3,shuffle = True)
# print(dt_Train)

X_train = dt_Train[:,1:7]
y_train = dt_Train[:,7]
X_test = dt_Test[:,1:7]
y_test = dt_Test[:,7]

svm = SVC(max_iter=-1,kernel='linear',class_weight={0: 0.1,1:0.99},gamma='auto',shrinking=True,tol=0.001,cache_size=200)
svm.fit(X_train,y_train)
y_predict = svm.predict(X_test)

count = 0
for i in range(0,len(y_predict)):
    if(y_test[i]==y_predict[i]):
        count = count +1
print('Ty le du doan dung:',count/len(y_predict))