import numpy as np
import pandas as pd 
data = pd.read_csv('\\USA_Housing.csv')
X_train = np.array(data[["TB_ThuNhapKhuVuc","TB_tuoinha","TB_dientich","TB_sophong","Dansokhuvuc"]].values).T
y_train = np.array(data["Gia"].values).T
x_test = np.array([[81885.92718,4.42367179,8.167688003,6.1,40149.96575]]).T

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
y_test_hat = x_test.T@w
print('Giá trị dự đoán mẫu mới: ', y_test_hat)
print('Giá trị dự đoán tập huấn luyện: ', X_train.T@w)

