import math
import pandas as pd
# #ham nse
data = pd.read_csv('ketthucmon\mushrooms.csv')
def nse(actual, predicted):
    mean_actual = sum(predicted) / len(predicted)
    sse = []
    sst = []
    for i in range(len(actual)):
        mea = actual[i]
        pre = predicted[i]
        sse.append((mea - pre) ** 2)
        sst.append((mea - mean_actual) ** 2)
    return 1 - ((sum(sse)) /(sum(sst)))

#ham r2
def r2(actual, predicted):
    mean_actual = sum(predicted) / len(predicted)
    pre_actual = sum(actual) / len(actual)
    sse = []
    sst = []
    ssi = []
    for i in range(len(actual)):
        mean = actual[i]
        pre = predicted[i]
        sse.append((mean - mean_actual)*(pre - pre_actual))
        sst.append((mean - mean_actual) ** 2)
        ssi.append((pre - pre_actual) ** 2)
    return ((sum(sse)) /(math.sqrt(sum(sst) * sum(ssi))))**2


#ham mae
def mae(actual, predicted):
    sse = []
    for i in range(len(actual)):
        mea = actual[i]
        pre = predicted[i]
        sse.append(abs(mea - pre))
    return sum(sse)/len(actual)

#ham rmse
def rmse(actual, predicted):
    sse = []
    for i in range(len(actual)):
        mea = actual[i]
        pre = predicted[i]
        sse.append(pow((mea - pre),2))
    return math.sqrt(sum(sse)/len(actual))
