import os
print(os.getcwd())
print(os.listdir(path='.'))


from Bezier.BezierSurface import BezierSurface
from Bezier.Line import Line3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


#fig = plt.figure()
#ax = Axes3D(fig)
line = Line3D(np.array([0.4,7,1]),np.array([0.4,7,0]))

P = [
        [np.array([0.,0.,0.5]), np.array([0.,30.,1.]), np.array([0.,60.,0.]), np.array([0.,90.,0.])],
        [np.array([10.,0.,0.]), np.array([10.,30.,0.]), np.array([10.,60.,0.]), np.array([10.,90.,0.])],
        [np.array([20.,0.,0.]), np.array([20.,30.,0.]), np.array([20.,60.,0.]), np.array([20.,90.,0.])],
        [np.array([31.,0.,0.]), np.array([31.,30.,0.]), np.array([31.,60.,0.]), np.array([31.,90.,0.])]
    ]

BS = BezierSurface(P)
uv = BS.Clip(line)

#BS.Plot(ax)
#line.Plot(ax,0,4,0,4)

'''
p = []
for params in uv:
    p.append(BS.Point(params[0], params[1]))
ax.scatter(BS.getColumnArray(p,0), BS.getColumnArray(p,1), BS.getColumnArray(p,2))


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
'''
plt.legend()
plt.show()
