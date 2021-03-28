from Bezier import BezierCurve
from Bezier import Line
import numpy as np
import matplotlib.pyplot as plt

#bezierの制御点
P = [
        np.array([0.,0]),
        np.array([1.,2]),
        np.array([2.,2]),
        np.array([3.,1])
    ]
bc = BezierCurve.BezierCurve(P)
line = Line.Line2D(np.array([0,2]), np.array([3,0]))
clip_xy = bc.Clip(line)
print(clip_xy)
for p in clip_xy:
    plt.scatter(p[0], p[1], c='r')
bc.Plot()
line.Plot()
plt.show()
