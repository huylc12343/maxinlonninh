import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron


data = pd.read_csv('btl2\drug200.csv')
# data['ph'] = data['ph'].map({'low':0,'medium':1,'high':2})
# data['Chloramines'] = data['Chloramines'].map({'low':0,'medium':1,'high':2})
# data['Turbidity'] = data['Turbidity'].map({'low':0,'medium':1,'high':2})

# data = np.array(data[['ph','Chloramines','Turbidity','Potability']].values)
data['Sex'] = data['Sex'].map({'F':0,'M':1})
data['BP'] = data['BP'].map({'LOW':0,'NORMAL':1,'HIGH':2})
data['Cholesterol'] = data['Cholesterol'].map({'HIGH':2,'NORMAL':1})
data['Drug'] = data['Drug'].map({'drugA':0,'drugB':1,'drugC':1,'DrugY':0,'DrugX':1})

data = np.array(data[['Age','Sex','BP','Cholesterol','Na_to_K','Drug']].values)
dt_Train,dt_Test = train_test_split(data,test_size = 0.3,shuffle = True)
# print(dt_Train)
X_train = dt_Train[:,:5]
y_train = dt_Train[:,4]
X_test = dt_Test[:,:5]
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