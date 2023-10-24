import numpy as np

X_train = np.array([[147], [150], [153], [158], [163], [165], [168], [170], [173], [175], [178], [180], [183]]).T
# weight (kg)
y_train = np.array([ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]).T

x_test = np.array([[178]]).T

# w = np.linalg.pinv(X_train) @ y_train

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
y_test_hat = x_test.T@w
print('Giá trị dự đoán mẫu mới: ', y_test_hat)
print('Giá trị dự đoán tập huấn luyện: ', X_train.T@w)


