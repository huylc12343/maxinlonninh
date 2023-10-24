import numpy as np

X_train = np.array([[1,147], [1,150], [1,153],[1,155], [1,158],[1,160], [1,163], [1,165], [1,168], [1,170], [1,173], [1,175], [1,178], [1,180], [1,183]]).T
y_train = np.array([[ 49, 50, 51, 52, 54,56, 58, 59, 60, 72, 63, 64, 66, 67, 68]]).T
x_test = np.array([[1,158]]).T

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
y_test_hat = x_test.T@w
print('Giá trị dự đoán mẫu mới: ', y_test_hat)
print('Giá trị dự đoán tập huấn luyện: ', X_train.T@w)

