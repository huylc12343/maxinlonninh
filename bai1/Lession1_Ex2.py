import numpy as np

arr_A = np.array([[1, 4, -1], [2, 0, 1]])
arr_B = np.array([[-1, 0], [1, 3], [-1, 1]])
arr_BT = np.transpose(arr_B)
# arr_Af = np.array([[1,4,-1],[2,0,1],[0,0,0]]) #them so 0 vao de ma tran vuong

print("A + Bt = ", arr_A + arr_BT)
print("A - Bt = ", arr_A - arr_BT)
print("A x 2 = ", arr_A * 2)
print("A * B = ", arr_A @ arr_B)
print("A * A^-1 = ", arr_A @ np.linalg.pinv(arr_A))

# print("A * A^-1 = ", arr_A @ np.transpose(arr_Af))