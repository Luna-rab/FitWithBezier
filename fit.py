from Bezier.BezierSurface import BezierSurface
from Bezier.Line import Line3D
from Shiori import CSFDataRaw
import numpy as np
import scipy.optimize as optimize

def dist_z(surface, point):
    L = Line3D(point, np.array([point[0], point[1], 0]))
    uv = surface.Clip(L)
    if len(uv) != 1:
        exit
    return np.linalg.norm(surface.Point(uv[0][0], uv[0][1]) - point ,ord=2)

def format(points_array):
    n = 3
    m = 3
    p = []
    for i in range(n+1):
        l = []
        for j in range(m+1):
            l.append(np.array([points_array[(i*(m+1)+j)*3+0], points_array[(i*(m+1)+j)*3+1], points_array[(i*(m+1)+j)*3+2]]))
        p.append(l)
    return p

def deformat(points):
    p_array = []
    for l in points:
        for p in l:
            p_array.append(p[0])
            p_array.append(p[1])
            p_array.append(p[2])
    return p_array

def loss(surface_array, data_points):
    surface = BezierSurface(format(surface_array))
    sum_dist = 0
    for p in data_points:
        sum_dist += dist_z(surface, p)**2
    
    return np.sqrt(sum_dist)/len(data_points)

if __name__ == '__main__':
    data_points = CSFDataRaw.cpd_ecc_cont1
    control_points = [
        [np.array([0.,0.,0.5]), np.array([0.,30.,1.]), np.array([0.,60.,0.]), np.array([0.,90.,0.])],
        [np.array([10.,0.,0.]), np.array([10.,23.3,0.]), np.array([10.,46.7,0.]), np.array([10.,70.,0.])],
        [np.array([20.,0.,0.]), np.array([25.,16.7,0.]), np.array([20.,33.3,0.]), np.array([20.,50.,0.])],
        [np.array([31.,0.,0.]), np.array([31.,10.,0.]), np.array([31.,20.,0.]), np.array([31.,30.,0.])]
    ]
    result = optimize.minimize(fun=loss, x0=deformat(control_points), args=data_points)
    print(result)



