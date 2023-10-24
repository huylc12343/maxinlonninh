import math
import numpy
def d(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)
def phanlop(data):
    x1,y1 = data[0]
    x2,y2 = data[1]
    x3,y3 = data[2]
    gr1 = []
    gr2 = []
    gr3 = []
    for i in range(2,len(data)):
        x,y = data[i]
        d1 = d(x,y,x1,y1)
        d2 = d(x,y,x2,y2)
        d3 = d(x,y,x3,y3)
        if(d1 == min(d1,d2,d3)):
            gr1.append([x,y])
        elif(d2 == min(d1,d2,d3)):
            gr2.append([x,y])
        else:
            gr3.append([x,y])
    return gr1,gr2,gr3
def trungbinhcong(data):
    sumx = 0
    sumy = 0 
    for i in data:
        x,y= i 
        sumx+=x
        sumy+=y
    tbcx = sumx/len(data)
    tbcy = sumy/len(data)
    return [tbcx,tbcy]
data = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
g1,g2,g3 = phanlop(data)
# print(trungbinhcong(g1))
# print(trungbinhcong(g2))
# print(trungbinhcong(g3))
# print('g1 =',g1)
# print('g2 =',g2)
# print('g3 =',g3)
for i in range(len(data)):
    phanlop(data)
    data.insert(0,trungbinhcong(g3))
    data.insert(0,trungbinhcong(g2))
    data.insert(0,trungbinhcong(g1))
    g1,g2,g3 = phanlop(data)
print(data)