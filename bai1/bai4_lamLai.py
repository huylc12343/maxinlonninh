import sympy as sp 
from scipy.optimize import fsolve
import numpy as np
x = sp.symbols('x')
f = x**2 +5*(sp.sin(x)) + 2 
df = sp.diff(f,x)
df_numeric = sp.lambdify(x,df)
solutions = fsolve(df_numeric, x0 = 0)
print("f: ",f)
print("df_numerric:",df_numeric)
print(f"Giá trị của đạo hàm f(x) là: {df}")
print("Các giá trị x thỏa mãn:", solutions)