from Bezier.BezierSurface import BezierSurface
from Bezier.Line import Line3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


fig = plt.figure()
ax = Axes3D(fig)
line = Line3D(np.array([0.4,7,1]),np.array([0.4,7,0]))

P = [
        [np.array([0.,0.,0.5]), np.array([0.,30.,1.]), np.array([0.,60.,0.]), np.array([0.,90.,0.])],
        [np.array([10.,0.,0.]), np.array([10.,23.3,0.]), np.array([10.,46.7,0.]), np.array([10.,70.,0.])],
        [np.array([20.,0.,0.]), np.array([25.,16.7,0.]), np.array([20.,33.3,0.]), np.array([20.,50.,0.])],
        [np.array([31.,0.,0.]), np.array([31.,10.,0.]), np.array([31.,20.,0.]), np.array([31.,30.,0.])]
    ]
try:
    BS = BezierSurface(P)
    uv = BS.Clip(line)
except:
    pass

BS.Plot(ax)
line.Plot(ax,0,4,0,4)


p = []
for params in uv:
    p.append(BS.Point(params[0], params[1]))
ax.scatter(BS.getColumnArray(p,0), BS.getColumnArray(p,1), BS.getColumnArray(p,2))


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.legend()
plt.show()
