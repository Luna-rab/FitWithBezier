import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import ConvexHull
from . import Line
import scipy
import math
import random
#import BezierSurface

class BezierPatch:
    ploted_num = 0

    def __init__(self, d):
        for row in d:
            if len(row) != len(d[0]):
                raise Exception('DimensionError')
        
        self._d = d
        self._norder = len(d) - 1
        self._morder = len(d[0]) - 1

    @property
    def d(self):
        return self._d

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
        Puv = np.array([0.,0.])
        for j in range(self.morder + 1):
            for i in range(self.norder + 1):
                Puv += self.Bernstein(self.norder,i,u) * self.Bernstein(self.morder,j,v) * self.d[i][j]
        return Puv

    def Plot(self, color='blue'):
        for u in np.linspace(0,1,11):
            P = []
            for v in np.linspace(0,1,101):
                P.append(self.Point(u,v))
            plt.plot(self.getColumnArray(P,0), self.getColumnArray(P,1), color=color)

        for v in np.linspace(0,1,11):
            P = []
            for u in np.linspace(0,1,101):
                P.append(self.Point(u,v))
            plt.plot(self.getColumnArray(P,0), self.getColumnArray(P,1), color=color)

    def PlotDetail(self,L):
        self.Plot()
        x=self.getMatrix(self.d,0).flatten()
        y=self.getMatrix(self.d,1).flatten()
        plt.scatter(x, y, c='red')
        for u in range(self.norder+1):
            plt.plot(self.getMatrix(self.d,0)[:,u],self.getMatrix(self.d,1)[:,u], c='r', ls='--')
        for v in range(self.morder+1):
            plt.plot(self.getMatrix(self.d,0)[v,:],self.getMatrix(self.d,1)[v,:], c='r', ls='--')
        L.Plot('g',-3,3,)
        plt.legend()
        plt.show()

    def getPlotColor(self):
        colors = ['r' ,'g' ,'b', 'm', 'c']
        color = colors[BezierPatch.ploted_num % 5]
        BezierPatch.ploted_num += 1
        return color

    def Clip(self):
        return self.ClipU()

    def ClipU(self):
        V0 = self.d[0][self.morder] - self.d[0][0]
        V1 = self.d[self.norder][self.morder] - self.d[self.norder][0]
        L = Line.Line2D(np.array([0,0]), V0+V1)

        x0 = np.empty([0,2],float)
        if np.linalg.norm(self.d[0][0]-self.d[self.morder][self.norder] ,ord=2) + np.linalg.norm(self.d[self.morder][0]-self.d[0][self.norder] ,ord=2) < 1e-4:
            x0 = np.append(x0, np.array([[0.5, 0.5]]), axis=0)
            return x0

        ud = np.empty([0,2],float)
        for i in range(self.norder+1):
            for j in range(self.morder+1):
                ud = np.append(ud, np.array([[float(i)/self.norder, L.dist2Point(self.d[i][j])]]), axis=0)

        if all(ud[:,1]>0) or all(ud[:,1]<0):
            return None
        elif isHull(ud):
            hull = ConvexHull(ud)
            hull_points = hull.points[hull.vertices]
            hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)

            prev_hp = None
            u_max = 0.
            u_min = 0.
            for hp in hull_points:
                if not prev_hp is None:
                    a = hp[1] - prev_hp[1]
                    b = prev_hp[0] - hp[0]
                    c = prev_hp[0]*hp[1] - hp[0]*prev_hp[1]
                    if hp[1]*prev_hp[1] <= 0:
                        X = np.array([
                            [a,b],
                            [0,1]
                        ])
                        P = np.array([
                            [c],
                            [0]
                        ])
                        ans = (np.linalg.inv(X) @ P).flatten()
                        if u_max < ans[0]:
                            u_max, u_min = u_min, u_max
                            u_max = ans[0]
                        else:
                            u_min = ans[0]
                prev_hp = hp
            if u_max < u_min:
                u_max, u_min = u_min, u_max
        else:
            a = ud[0][1] - ud[-1][1]
            b = ud[-1][0] - ud[0][0]
            c = ud[-1][0]*ud[0][1] - ud[0][0]*ud[-1][1]
            X = np.array([
                [a,b],
                [0,1]
                ])
            P = np.array([
                [c],
                [0]
            ])
            ans = (np.linalg.inv(X) @ P).flatten()
            u_max = u_min = ans[0]
            u_max = min(u_max+1e-6, 1)
            u_min = max(0, u_min-1e-6)

        if 0.8 < u_max-u_min:
            div_patch1, div_patch2 = self.divideU(0.5)
            x = div_patch1.ClipV()
            if x is not None:
                x1 = np.array([[0.5, 0],[0,1]])
                x = x.reshape([-1, 2])
                x = x.dot(x1) + np.array([0, 0])
                x0 = np.append(x0, x, axis=0)
            x = div_patch2.ClipV()
            if x is not None:
                x1 = np.array([[0.5, 0],[0,1]])
                x = x.reshape([-1, 2])
                x = x.dot(x1) + np.array([0.5, 0])
                x0 = np.append(x0, x, axis=0)

        else:
            div_patch, _ = self.divideU(u_max)
            _, div_patch = div_patch.divideU(u_min/u_max)
            x = div_patch.ClipV()
            if x is not None:
                x1 = np.array([[u_max-u_min, 0],[0,1]])
                x = x.reshape([-1, 2])
                x = x.dot(x1) + np.array([u_min, 0])
                x0 = np.append(x0, x, axis=0)
        
        return x0


    def ClipV(self):
        V0 = self.d[self.norder][0] - self.d[0][0]
        V1 = self.d[self.norder][self.morder] - self.d[0][self.morder]
        L = Line.Line2D(np.array([0,0]), V0+V1)

        x0 = np.empty([0,2],float)
        if np.linalg.norm(self.d[0][0]-self.d[self.morder][self.norder] ,ord=2) + np.linalg.norm(self.d[self.morder][0]-self.d[0][self.norder] ,ord=2) < 1e-4:
            x0 = np.append(x0, np.array([[0.5,0.5]]), axis=0)
            return x0

        vd = np.empty([0,2],float)
        for i in range(self.norder+1):
            for j in range(self.morder+1):
                vd = np.append(vd, np.array([[float(j)/self.morder, L.dist2Point(self.d[i][j])]]), axis=0)
        
        if all(vd[:,1]>0) or all(vd[:,1]<0):
            return None
        elif isHull(vd):
            hull = ConvexHull(vd)
            hull_points = hull.points[hull.vertices]
            hull_points = np.append(hull_points, np.array([hull_points[0]]), axis=0)

            prev_hp = None
            v_max = 0.
            v_min = 0.
            for hp in hull_points:
                if not prev_hp is None:
                    a = hp[1] - prev_hp[1]
                    b = prev_hp[0] - hp[0]
                    c = prev_hp[0]*hp[1] - hp[0]*prev_hp[1]
                    if hp[1]*prev_hp[1] <= 0:
                        X = np.array([
                            [a,b],
                            [0,1]
                        ])
                        P = np.array([
                            [c],
                            [0]
                        ])
                        ans = (np.linalg.inv(X) @ P).flatten()
                        if v_max < ans[0]:
                            v_max, v_min = v_min, v_max
                            v_max = ans[0]
                        else:
                            v_min = ans[0]
                prev_hp = hp
            if v_max < v_min:
                v_max, v_min = v_min, v_max
        else:
            a = vd[0][1] - vd[-1][1]
            b = vd[-1][0] - vd[0][0]
            c = vd[-1][0]*vd[0][1] - vd[0][0]*vd[-1][1]
            X = np.array([
                [a,b],
                [0,1]
                ])
            P = np.array([
                [c],
                [0]
            ])
            ans = (np.linalg.inv(X) @ P).flatten()
            v_max = v_min = ans[0]
            v_max = min(v_max+1e-6, 1)
            v_min = max(0, v_min-1e-6)

        if 0.8 < v_max-v_min:
            div_patch1, div_patch2 = self.divideV(0.5)
            x = div_patch1.ClipU()
            if x is not None:
                x1 = np.array([[1,0],[0, 0.5]])
                x = x.reshape([-1, 2])
                x = x.dot(x1)
                x0 = np.append(x0, x, axis=0)
            x = div_patch2.ClipU()
            if x is not None:
                x1 = np.array([[1,0],[0, 0.5]])
                x = x.reshape([-1, 2])
                x = x.dot(x1) + np.array([0, 0.5])
                x0 = np.append(x0, x, axis=0)

        else:
            div_patch, _ = self.divideV(v_max)
            _, div_patch = div_patch.divideV(v_min/v_max)
            x = div_patch.ClipU()
            if x is not None:
                x1 = np.array([[1,0],[0, v_max-v_min]])
                x = x.reshape([-1, 2])
                x = x.dot(x1) + np.array([0, v_min])
                x0 = np.append(x0, x, axis=0)

        return x0

    def divideV(self, t):
        if t==0:
            return self, self
        elif t==1:
            return self ,self
        else:
            P1 = []
            P2 = []
            for i in range(self.norder+1):
                P = self.getRowArray(self.d,i)
                Ps = [P] + self._de_casteljau_algorithm(P, t)
                row1 = []
                row2 = []
                for lst in Ps:
                    row1.append(lst[0])
                    row2.append(lst[-1])
                row2.reverse()
                P1.append(row1)
                P2.append(row2)
            return BezierPatch(P1), BezierPatch(P2)

    def divideU(self, t):
        if t==0:
            return self, self
        elif t==1:
            return self ,self
        else:
            P1 = []
            P2 = []
            for j in range(self.morder+1):
                P = self.getColumnArray(self.d,j)
                Ps = [P] + self._de_casteljau_algorithm(P, t)
                row1 = []
                row2 = []
                for lst in Ps:
                    row1.append(lst[0])
                    row2.append(lst[-1])
                row2.reverse()
                P1.append(row1)
                P2.append(row2)
            P1 = list(map(list, (zip(*P1))))
            P2 = list(map(list, (zip(*P2))))
            return BezierPatch(P1), BezierPatch(P2)

    def _de_casteljau_algorithm(self, P, t):
        prev_p = None
        Q = []
        for p in P:
            if not prev_p is None:
                Q.append(np.array((1-t)*prev_p + t*p))
            prev_p = p
        if len(Q) == 1:
            return [Q]
        return [Q] + self._de_casteljau_algorithm(Q, t)

def isHull(d):
    for i in range(1, d.shape[0]):
        for j in range(i+1, d.shape[0]):
            vec1 = np.append((d[i] - d[0]),0)
            vec2 = np.append((d[j] - d[0]),0)
            if  np.linalg.norm(np.cross(vec1,vec2), ord=2) > 1.7*1e-12:
                return True
    return False