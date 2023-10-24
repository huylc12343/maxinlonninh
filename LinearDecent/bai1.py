import math
x = -100
def daoham(x):
    return 2*x+5*math.cos(x)
while(abs(daoham(x)) > 0.0001):
    x = x - 0.001*daoham(x)
print(x)