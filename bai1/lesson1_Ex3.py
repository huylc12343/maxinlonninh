import sympy as sp
from scipy.optimize import fsolve
import numpy as np

x = sp.symbols('x')
y = sp.symbols('y')
f = x*x + sp.exp(x) + 2

df = sp.diff(f,x)
dff = sp.diff(df,x) 
print(f"f'x = {df}")
print(f"f''x = {dff}")
g = x**2 + 9*y**2 + 2*x*y + 3*x - y + 2
dgx = sp.diff(g,x)
dgy = sp.diff(g,y)
dgxy = sp.diff(dgx,y)
print(f"dfx = {dgx}")
print(f"dfx = {dgy}")
print(f"dfxy = {dgxy}")
h = f**2 + g*2
dhx = sp.diff(h,x)
dhy = sp.diff(h,y)
print(f"dhx = {dhx}")
print(f"dhy = {dhy}")