import sympy as sp

x = sp.symbols('x')
y = sp.symbols('y')
fx = x*x + sp.exp(x) + 2

df1 = sp.diff(fx,x)
print(df1);
df2 = sp.diff(df1,x)
print(df2);
gxy = x*x + 9*y*y + 2*x*y + 3*x -y + 2
df3 = sp.diff(gxy, x)
print(df3)
df4 = sp.diff(gxy, y)
print(df4)
df5 = sp.diff(gxy, x)
df6 = sp.diff(df5, y)
print(df6)
hxy = fx**2 + 2*gxy
df7 = sp.diff(hxy, x)
print(df7)
df8 = sp.diff(hxy, y)
print(df8)
