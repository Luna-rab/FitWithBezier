from Bezier.BezierSurface import BezierSurface
from Bezier.Line import Line3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def test3():
    fig = plt.figure()
    ax = Axes3D(fig)
    line = Line3D(np.array([1.,0.,1.]),np.array([1.,2.,-1.]))

    P = [
            [np.array([0.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,2.,0.]), np.array([0.,3.,0.])],
            [np.array([1.,0.,0.]), np.array([1.,1.,0.]), np.array([1.,2.,0.]), np.array([1.,3.,0.])],
            [np.array([2.,0.,0.]), np.array([2.,1.,0.]), np.array([2.,2.,0.]), np.array([2.,3.,0.])],
            [np.array([3.,0.,0.]), np.array([3.,1.,0.]), np.array([3.,2.,0.]), np.array([3.,3.,0.])]
        ]

    BS = BezierSurface(P)
    uvs = BS.Clip(line)

    answer = np.array([1.,1.,0.])

    print(uvs)
    for uv in uvs:
        print(BS.Point(uv[0], uv[1]))
    print(answer)

    
    BS.Plot(ax)
    line.Plot(ax,0,4,0,4)

    p = []
    for uv in uvs:
        p.append(BS.Point(uv[0], uv[1]))
    ax.scatter(BS.getColumnArray(p,0), BS.getColumnArray(p,1), BS.getColumnArray(p,2))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.legend()
    plt.show()