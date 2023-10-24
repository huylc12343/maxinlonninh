import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
data = pd.read_csv('btl2\cleaned_data2.csv')
data = np.array(data[['ph','Hardness','Chloramines','Turbidity','Potability']].values)

print(data)
dt_Train,dt_Test = train_test_split(data,test_size = 0.3,shuffle = True)
# print(dt_Train)
X_train = dt_Train[:,:4]
y_train = dt_Train[:,4]
X_test = dt_Test[:,:4]
y_test = dt_Test[:,4]

pla = Perceptron()
pla.fit(X_train,y_train)
y_predict = pla.predict(X_test)
count = 0
for i in range(0,len(y_predict)):
    if(y_test[i]==y_predict[i]):
        count = count +1
print('Ty le du doan dung:',count/len(y_predict))

print(y_test)
print(y_predict)