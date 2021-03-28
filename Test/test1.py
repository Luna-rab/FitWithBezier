from Bezier.BezierSurface import BezierSurface
from Bezier.Line import Line3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def test1():
  fig = plt.figure()
  ax = Axes3D(fig)
  line = Line3D(np.array([0,0,1]),np.array([3,3,-0.5]))

  P = [[np.array([0.12918508, 0.12766157, 1.52149256]),
    np.array([ 0.26694044,  1.95664224, -2.73189842]),
    np.array([ 0.00594417,  2.19540763, -0.99503981]),
    np.array([ 0.1501343 ,  3.42745308, -0.22510591])],
  [np.array([ 1.30825519,  0.00456516, -1.6117688 ]),
    np.array([ 1.28998114,  1.65551717, -1.95958735]),
    np.array([ 1.21249709,  2.21310516, -0.62739403]),
    np.array([ 1.55490357,  3.90483967, -0.15813955])],
  [np.array([2.79362009, 0.26875693, 2.868869  ]),
    np.array([ 2.46356973,  1.8770085 , -0.22571524]),
    np.array([2.98883323, 2.33022236, 1.56486395]),
    np.array([2.99792536, 3.52091345, 2.02020116])],
  [np.array([3.98567669, 0.2150054 , 0.75901311]),
    np.array([ 3.8049761 ,  1.1042801 , -2.66303292]),
    np.array([3.51663586, 2.63601953, 0.95266433]),
    np.array([ 3.32802331,  3.16356859, -1.11334953])]]

  BS = BezierSurface(P)
  uv = BS.Clip(line)
  print(uv)

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
