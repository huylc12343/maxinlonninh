import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

data = pd.read_csv('\\USA_Housing.csv')
dt_train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)
X_train = dt_train.iloc[:, :5]
y_train = dt_train.iloc[:, 5]
x_test = dt_Test.iloc[:, :5]
y_test = dt_Test.iloc[:, 5]
# X_train = np.array(data[["TB_ThuNhapKhuVuc","TB_tuoinha","TB_dientich","TB_sophong","Dansokhuvuc"]].values)
# y_train = np.array(data["Gia"].values)
# x_test = np.array([[79248.64245,6.002899808,6.730821019,3.09,40173.07217]])


#LinearRegression
reg = LinearRegression().fit(X_train, y_train)
f = reg.predict(x_test)
print("Ket qua mo hinh LinearRegression: ",f[:5])
print("Do chenh lech: %.8f" %r2_score(y_test, f))

#ridge///
model_2 = Ridge().fit(X_train,y_train)
y_pred2 = model_2.predict(x_test)
print("Ket qua mo hinh Ridge: ",y_pred2[:5])

print("Do chenh lech: %.8f" %r2_score(y_test, y_pred2))

#Lasso
model_3 = Lasso().fit(X_train,y_train)
y_pred3 = model_3.predict(x_test)
print("Ket qua mo hinh Lasso: ",y_pred3[:5])
print("Do chenh lech: %.8f" %r2_score(y_test, y_pred3))