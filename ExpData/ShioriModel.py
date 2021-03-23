import scipy.stats as stats
import numpy as np
from scipy.optimize import curve_fit

def csf_fit(cpd,cont,p0): 
    # Mannosさんの提案モデル３変数CSFフィッティング
    A = 1/cont
    A = np.log10(A)
    def csf(cpd,a,b,c): #3変数CSF
        if b <= 0 or a <= 0 : 
            A = np.inf
        else:
            A = np.log10(a)+np.log10(b*cpd)-1*((b*cpd)**c)*np.log10(np.e)
        return A
    params, cov = curve_fit(csf,cpd,A,p0)
    return params

def scale1(E,E2):
    def scale1_e(E):
        a= 5.51378481  
        s1=3.83934837 
        s2=38.13888608
        y = a*stats.norm.pdf(E, loc=0, scale=s1)+(1-a*stats.norm.pdf(0, loc=0, scale=s1))*stats.norm.pdf(E, loc=0, scale=s2)/stats.norm.pdf(0, loc=0, scale=s2)
        return y
    return scale1_e(E2)/scale1_e(E)

def scale2(E,E2,s1,s2):
    def scale2_e(E,s1,s2):
        y = 10**(E*(s1*(E<=30)+s2*(E>30))+30*(s1-s2)*(E>30))
        return y
    return scale2_e(E2,s1,s2)/scale2_e(E,s1,s2)

def scale3(E,E2,s3):
    def scale3_e(E,s3):
        y = s3*(E*(E<=30)+30*(E>30))+1
        return y
    return scale3_e(E2,s3)/scale3_e(E,s3)

def csf_view(x,csf_params,inv_params,Ecc=0,Ecc2=0): 
    # Ecc2度での本研究で提案したCSFモデルを表示する
    # Ecc度のCSFをEcc2度のCSFに変換するため、Ecc度で求めたcsf_paramsを用いる
    return csf_params[0]*scale1(Ecc,Ecc2)*(csf_params[1]*scale2(Ecc,Ecc2,inv_params[0],inv_params[1])*x)*np.exp(-1*(csf_params[1]*scale2(Ecc,Ecc2,inv_params[0],inv_params[1])*x)**(csf_params[2]*scale3(Ecc,Ecc2,inv_params[2])))
    
