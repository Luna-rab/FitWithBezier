import sys
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pprint
from . import BezierPatch
from . import Line
import pandas as pd

class BezierSurface :
    def __init__(self,P):
        for row in P:
            if len(row) != len(P[0]):
                raise Exception('DimensionError')
        
        self._P = P   
        self._norder = len(P) - 1
        self._morder = len(P[0]) - 1

    @property
    def P(self):
        return self._P
    
    @property
    def morder(self):
        return self._morder

    @property
    def norder(self):
        return self._norder

    def getMatrix(self, P, k):
        for row in P:
            if len(row) != len(P[0]):
                raise Exception('DimensionError')
        
        mat = np.empty([len(P), len(P[0])], float)
        for i in range(0, len(P)):
            for j in range(0, len(P[0])):
                mat[i,j] = P[i][j][k]
        return mat

    def getRowArray(self, P, i):
        return np.array(P[i])
    
    def getColumnArray(self, P, i):
        col = []
        for row in P:
            col.append(row[i])
        return np.array(col)

    def Bernstein(self,n,i,t):
        return comb(n,i) * (1-t)**(n-i) * t**i

    def Point(self,u,v):
        Suv = np.array([0.,0.,0.])
        for i in range(self.norder + 1):
            for j in range(self.morder + 1):
                Suv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.P[i][j]
        return Suv
    
    def Plot(self, ax, c='blue', xlim=(-float('inf'),float('inf')), ylim=(-float('inf'),float('inf')), zlim=(-float('inf'),float('inf')), rstride=3, cstride=3):
        Suv = []
        for u in np.linspace(0.,1.,101):
            row = []
            for v in np.linspace(0.,1.,101):
                row.append(self.Point(u,v))
            Suv.append(row)
        
        x = self.getMatrix(Suv, 0)
        y = self.getMatrix(Suv, 1)
        z = self.getMatrix(Suv, 2)

        norm = plt.Normalize(vmin=max(zlim[0],z.min()), vmax=min(zlim[1],z.max()))
        colors = plt.cm.jet(norm(z))
        if c == 'Blues':
            colors = plt.cm.Blues(norm(z))
        elif c == 'Greens':
            colors = plt.cm.Greens(norm(z))
        elif c == 'Reds':
            colors = plt.cm.Reds(norm(z))
        elif c == 'Oranges':
            colors = plt.cm.Oranges(norm(z))
        else:
            colors = plt.cm.jet(norm(z))

        colors[:,:,3] = 0.3
        colors[x < xlim[0]] = (0, 0, 0, 0)
        colors[x > xlim[1]] = (0, 0, 0, 0)
        colors[y < ylim[0]] = (0, 0, 0, 0)
        colors[y > ylim[1]] = (0, 0, 0, 0)
        colors[z < zlim[0]] = (0, 0, 0, 0)
        colors[z > zlim[1]] = (0, 0, 0, 0)
        
        surf = ax.plot_surface(x, y, z, antialiased=True, facecolors=colors, rstride=rstride, cstride=cstride, edgecolor="r")
        surf.set_facecolor((0,0,0,0))

    def getBezierPatch(self,line):
        pl1,pl2 = line.intersectionPlane()
        d = [[None for i in range(self.morder+1)] for j in range(self.norder+1)]

        for i in range(self.norder+1):
            for j in range(self.morder+1):
                d[i][j] = np.array([pl1.dist2Point(self.P[i][j]), pl2.dist2Point(self.P[i][j])])
        return BezierPatch.BezierPatch(d)

    def Clip(self, line):
        return self.getBezierPatch(line).Clip()

    def Slicex(self, x, z_min, z_max):
        zarray = np.linspace(z_min,z_max,1001)
        intersections = np.empty([0,3],float)
        for z in zarray:
            line = Line.Line3D(np.array([x,0,z]),np.array([x,1,z]))
            uvs = self.Clip(line)
            for uv in uvs:
                intersections = np.append(intersections, np.array([self.Point(uv[0], uv[1])]), axis=0)
        df = pd.DataFrame(np.delete(intersections,0,1))
        df_s = df.sort_values(0)
        return df_s.values.T

    def Slicey(self, y, z_min, z_max):
        zarray = np.linspace(z_min,z_max,1001)
        intersections = np.empty([0,3],float)
        for z in zarray:
            line = Line.Line3D(np.array([0,y,z]),np.array([1,y,z]))
            uvs = self.Clip(line)
            for uv in uvs:
                intersections = np.append(intersections, np.array([self.Point(uv[0], uv[1])]), axis=0)
        df = pd.DataFrame(np.delete(intersections,1,1))
        df_s = df.sort_values(0)
        return df_s.values.T

