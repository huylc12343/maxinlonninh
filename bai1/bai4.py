import sympy as sp
from scipy.optimize import fsolve
import numpy as np

# Khai báo biến
x = sp.symbols('x')

# Định nghĩa hàm
f = x**2 + 5*(sp.sin(x)) + 2

# Tính đạo hàm của hàm f(x)
df = sp.diff(f, x)

# Chuyển đổi biểu thức Sympy thành hàm số học sử dụng lambdify
df_numeric = sp.lambdify(x, df)

# Tìm giá trị x thỏa mãn phương trình df = 0
solutions = fsolve(df_numeric, x0=0)

print(f"Đạo hàm của hàm f(x) là: {df}")
print("Các giá trị x thỏa mãn:", solutions)