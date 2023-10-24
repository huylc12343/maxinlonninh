import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('\\USA_Housing.csv')
X_train = np.array(data[["TB_ThuNhapKhuVuc","TB_tuoinha","TB_dientich","TB_sophong","Dansokhuvuc"]].values)
y_train = np.array(data["Gia"].values)
x_test = np.array([[83394.40783,5.601486668,5.905024344,2.08,30487.52462]])
# [81885.92718,4.42367179,8.167688003,6.1,40149.96575]
# [80000,10,8,1,8000]
reg = LinearRegression().fit(X_train, y_train)
print("w = ",reg.coef_)
print("w0 = ",reg.intercept_)
f = reg.predict((x_test))
print("f = ",f)