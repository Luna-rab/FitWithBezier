import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import CSFData
import CSFDataRaw
import ShioriModel

def scatData(Data,ax,c='k'):
    cpd = Data[:,0]
    ecc = Data[:,1]
    cont = Data[:,2]
    A = 1./cont
    cpd = np.log10(cpd)
    A = np.log10(A)
    ax.scatter(cpd, ecc, A, color=c,linewidth=0.3)

def shiori_3Dplot(Data, ecc, inv_params, ax):
    cpd_0 = Data[Data[:,1]==0.0][:,0]
    cont_0 = Data[Data[:,1]==0.0][:,2]
    cpd = Data[Data[:,1]==ecc][:,0]
    cont = Data[Data[:,1]==ecc][:,2]

    c = [ecc/90.,0.2,1.-(ecc/90.)]

    if cpd.size != 0 and cont.size != 0:
        params = ShioriModel.csf_fit(cpd_0, cont_0, np.array([524,0.2,1.1]))
        x = np.linspace(np.log10(0.2), np.log10(50.0), 101)
        y = np.full(101, ecc)
        z = ShioriModel.csf_view(10**x, params, inv_params, Ecc2=ecc)
        xyz = np.array([x,y,z]).T
        x = xyz[xyz[:,2]>=1.][:,0]
        y = xyz[xyz[:,2]>=1.][:,1]
        z = xyz[xyz[:,2]>=1.][:,2]
        ax.plot(x,y,np.log10(z),color=c,label=ecc)

Data = CSFData.cpd_ecc_cont1
inv_params = CSFData.inv_params_1

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('log(cpd)')
ax.set_ylabel('ecc')
ax.set_zlabel('log(1/cont)')

ax.set_xlim(np.log10(0.2), np.log10(50))
ax.set_ylim(90.,0.)
ax.set_zlim(np.log10(1), np.log10(500))



scatData(Data,ax)
for ecc in range(0, 85, 7):
    shiori_3Dplot(Data, float(ecc), inv_params, ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()