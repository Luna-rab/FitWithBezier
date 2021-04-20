from Bezier.BezierSurface import BezierSurface
from Bezier.Line import Line3D
from ExpData import CSFDataRaw as dr
from ExpData import ControllPoints as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

s1 = BezierSurface(cp.P1dec)
s2 = BezierSurface(cp.P2dec)
s3 = BezierSurface(cp.P3dec)
s4 = BezierSurface(cp.P4dec)

d1 = dr.procData(dr.cpd_ecc_cont1)
d2 = dr.procData(dr.cpd_ecc_cont2)
d3 = dr.procData(dr.cpd_ecc_cont3)
d4 = dr.procData(dr.cpd_ecc_cont4)

fig = plt.figure()
ax = Axes3D(fig)

xlim = (-0.5,2.0)
ylim = (0,90)
zlim = (0,float('inf'))

s1.Plot(ax, c='Blues', xlim=xlim, ylim=ylim, zlim=zlim)
s2.Plot(ax, c='Reds', xlim=xlim, ylim=ylim, zlim=zlim)
s3.Plot(ax, c='Oranges', xlim=xlim, ylim=ylim, zlim=zlim)
s4.Plot(ax, c='Greens', xlim=xlim, ylim=ylim, zlim=zlim)

dr.Plot(ax, d1, c='blue')
dr.Plot(ax, d2, c='red')
dr.Plot(ax, d3, c='yellow')
dr.Plot(ax, d4, c='green')

ax.set_xlabel('f')
ax.set_ylabel('ecc')
ax.set_zlabel('1/cont')
ax.set_xlim3d(-0.5,2)
ax.set_ylim3d(0,90)
ax.set_zlim3d(0,2.5)

plt.show()

fs = (0.4,1,2,4,8,12,20,30)
eccs = (0,7,21,28,56,70,84)

'''
s = s1
d = dr.cpd_ecc_cont1
for f in fs:
    print(f)
    ecc_cont = s.Slicex(np.log10(f),0,2.5)
    label = 'f='+str(f)
    plt.plot(ecc_cont[0],10**ecc_cont[1],label=label)
    mat_mask = dr.getMatrix(d)[:,0] == f
    mat = dr.getMatrix(d)[mat_mask]
    plt.scatter(mat[:,1],1/mat[:,2])
plt.xlabel('ecc')
plt.yscale('log')
plt.ylabel('1/cont')
plt.xlim(0,90)
plt.ylim(1,250)
plt.legend()
plt.show()
'''