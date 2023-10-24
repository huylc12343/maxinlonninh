import pandas as pd

data = pd.read_csv('D:\code\MachineLearning\\btl\\medical_cost.csv')
data['sex'] = data['sex'].map({'male':1,'female':0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].map({'northeast':3,'southeast':2,'southwest':1,'northwest':0})
print(data)
# def data_mahoasex(sex):
#     if sex == "male":
#         return 1
#     else:
#         return 0
# def data_mahoasmoker(smoker):
#     if smoker == "yes":
#         return 1
#     else:
#         return 0
# def data_mahoaregion(region):
#     if region == "southwest":
#         return 1
#     else:
#         return 0
# data['sex'] = data['sex'].apply(data_mahoasex)
# data['smoker'] = data['smoker'].apply(data_mahoasmoker)
# data['region'] = data['region'].apply(data_mahoaregion)

