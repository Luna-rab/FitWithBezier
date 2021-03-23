import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import CSFData
import CSFDataRaw
import ShioriModel

def shiori_cont(Data, cpd, ecc, inv_params):
    cpd_0 = Data[Data[:,1]==0.0][:,0]
    cont_0 = Data[Data[:,1]==0.0][:,2]

    params = ShioriModel.csf_fit(cpd_0, cont_0, np.array([524,0.2,1.1]))
    return ShioriModel.csf_view(10**cpd, params, inv_params, Ecc2=ecc)

def shiori_plotEccCont(Data, cpd, inv_params, ax):
    ecc = np.unique(Data[:,1])
    ax.scatter(ecc, np.log10(shiori_cont(Data, cpd, ecc, inv_params)))

Data = CSFData.cpd_ecc_cont1
inv_params = CSFData.inv_params_1
#0.2<cpd<50
i=1
for cpd in np.linspace(-0.5,0.5,5):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('ecc')
    ax.set_ylabel('log(1/cont)')
    ax.set_title('log10(cpd)='+str(cpd))

    ax.set_xlim(0.,90.)
    ax.set_ylim(np.log10(1), np.log10(500))

    shiori_plotEccCont(Data, cpd, inv_params, ax)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig('EccContGraph/EccCont1-'+str(i)+'.png')
    i+=1