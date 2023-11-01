## rebuild Perceptron
# import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

# raw_data = pandas.read_csv('classification/cars.csv')

# data = raw_data.apply(LabelEncoder().fit_transform)
# data = data[['buying','maint','doors','persons','lug_boot','safety','acceptability']].values

# train, test = train_test_split(data, test_size=0.3, shuffle=True)

# xTrain = train[:,:6]
# yTrain = train[:,6]
# xTest = test[:,:6]
# yTest = test[:,6]
try:
    raw_data = pandas.read_csv('perceptron/mushrooms.csv')
except:
    raw_data = pandas.read_csv('mushrooms.csv')

## Đọc dữ liệu
data = raw_data.apply(LabelEncoder().fit_transform)
## Loại bỏ tiêu đề
data = data[['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']].values

## Tách dữ liệu 
train, test = train_test_split(data, test_size=0.3, shuffle=False)

## Nhãn là cột đầu tiên, bộ dữ liệu là 9 cột kế tiếp
xTrain, yTrain = train[:,1:10], list(train[:,0])            ## Dữ liệu train
xTest, yTest = test[:,1:10], list(test[:,0])                ## Dữ Liệu test

perc = Perceptron()
perc.fit(xTrain, yTrain)
a = perc.coef_

yPredict = perc.predict(xTest)

count = 0
for i, v in enumerate(yPredict):
    if v == yTest[i]:
        count += 1

print('Du doan dung {0}/{1} = {2}'.format(count, len(yTest), count/len(yTest)))
# print(perc.coef_)