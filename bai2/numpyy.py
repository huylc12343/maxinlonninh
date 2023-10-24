import numpy as np

# Tạo một ma trận ví dụ
matrix = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])

# Tạo một cột đơn vị với cùng số hàng như ma trận ban đầu
unit_column = np.ones(1,3)

# Thêm cột đơn vị vào ma trận sử dụng hàm hstack
matrix_with_unit_column = np.hstack((matrix, unit_column))

print(matrix_with_unit_column)