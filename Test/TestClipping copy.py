import sys
sys.path.append('../')
from BezierSurface import BezierSurface
from Line import Line3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 
from pprint import pprint

n=3
m=3

P = []
for i in range(n+1):
    row = []
    for j in range(m+1):
        row.append(np.array([i+random.random(), j+random.random(), random.uniform(-3,3)]))
    P.append(row)

bs = BezierSurface(P)
u = random.random()
v = random.random()

line = Line3D(bs.Point(u,v), np.array([random.uniform(0,4), random.uniform(0,4), random.uniform(-3,3)]))

'''
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0,4)
ax.set_ylim(0,4)
'''
uv = bs.Clip(line)
p = []
for params in uv:
    p.append(bs.Point(params[0], params[1]))
print('P:')
pprint(P)
print('intersection point:')
print(bs.Point(u,v))

'''
bs.Plot(ax)
line.Plot(ax,0,4,0,4)
ax.scatter(bs.getColumnArray(p,0), bs.getColumnArray(p,1), bs.getColumnArray(p,2))
'''

plt.legend()
plt.show()

