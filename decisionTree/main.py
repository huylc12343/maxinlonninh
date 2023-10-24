import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_iris

df = pd.read_csv('decisionTree\Data.csv')
# df = np.array(df[['customer_id','credit_score','country','gender','age','tenure','balance','products_number','credit_card','active_member','estimated_salary','churn']])

le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df = np.array(df)
dt_train, dt_Test = train_test_split(df, test_size = 0.3, shuffle = True)

print(df)
df.to_csv('decisionTree\new.csv',index = False)