# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:24:57 2021

@author: abdul
"""
from numpy import linalg as LA
import numpy as np
from param_setting import *
#################################################
def RF_channel_gain(K,antenna_num=4):
    dist = 3 
    M = antenna_num
    Kdb = 8
    h=[]
    for k in range(K):
        h1=Pathloss_Rician_channels (dist+np.random.uniform(-0.3,0.3),1,M, Kdb)
        h.append(h1.reshape(M,1))
    #Optimal beamforming
    w=[]
    Beta=[]
    for k in range(K):
        w1, beta1 = beamforming2(h,k,M)
        w.append(w1)
        Beta.append(beta1)
    return h, Beta    
################ To find the optimal Transmisstion time #########################    
def goldensectionsearch (a,b,alpha,c,D,theta, Gamma, Epochs, tau):
    epsilon=0.000001             # accuracy value
    iter= 500                     # maximum number of iterations
    rho=(np.sqrt(5)-1)/2          # golden proportion coefficient, around 0.618
    k=0                           # number of iterations
    x1=a+(1-rho)*(b-a)            # computing x values
    x2=a+rho*(b-a)
    f_x1=fT(x1,alpha,c,D,theta, Gamma,Epochs, tau)                    # computing values in x points
    f_x2=fT(x2,alpha,c,D,theta, Gamma, Epochs, tau)
    while ((abs(b-a)>epsilon) and (k<iter)):
        k=k+1
        if(f_x1<f_x2):
            b=x2
            x2=x1
            x1=a+(1-rho)*(b-a)
            f_x1=fT(x1,alpha,c,D,theta, Gamma, Epochs, tau)
            f_x2=fT(x2,alpha,c,D,theta, Gamma, Epochs, tau) 
#             print('f_x1:',f_x1,'f_x2',f_x2)
        else:
            a=x1
            x1=x2
            x2=a+rho*(b-a)
            f_x1=fT(x1,alpha,c,D,theta, Gamma, Epochs, tau)
            f_x2=fT(x2,alpha,c,D,theta, Gamma, Epochs, tau) 
#             print('f_x1:',f_x1,'f_x2',f_x2)
        k=k+1
    
# chooses minimum point
    if(f_x1<f_x2):
        T = x1
    else:
        T = x2
    return T
################ Power Consumption #########################        
def fT(T,alpha,c,D,theta, Gamma, Epochs, tau):
    B = 10**(6)
    y = ((alpha/2)*(c*D)**3*Epochs**2)/(tau-T)**2 + T*( 2**(theta/(T*B)) -1)/Gamma
    return y   

def Pathloss_Rician_channels (dist,n_u,n_t, Kdb):
    pl = 2.6

    K = 10.**(Kdb/10)  #Rician factor
#Rician channels
    mu = np.sqrt( K/((K+1))) 

    s = np.sqrt( 1/(2*(K+1)))

    Hw = mu + s*(np.random.randn(n_t,n_u) + 1j * np.random.randn(n_t,n_u))
    M_1=np.ones((n_t,1))
    Hpl = np.multiply([np.sqrt(1/(dist**pl))*M_1],Hw)

    return Hpl

#################################################

def beamforming2(g, k,M):
    sigma2 = 10**(-10)
    Sumq= [np.matrix(g[i])*np.matrix(g[i]).T for i in range(len(g)) if i!=k]
    Sumq = Sumq + sigma2*np.eye(M)
    S2=sum(Sumq)**-1
    eigen_val, eigen_vec  = LA.eig(np.matrix(g[k])*np.matrix(g[k]).T*(S2))
    I =  np.argsort(eigen_val.real)[-1]
    w =  eigen_vec[I]
    Sumq1= [np.matrix(g[i])*np.matrix(g[i]).T for i in range(len(g)) if i!=k]
    Sumq1=Sumq1+ sigma2*np.eye(M)
    S3=sum(Sumq1)
    b=(np.matrix(w)*(S3)*np.matrix(w).T )
    a=np.matrix(w)*np.matrix(g[k])*np.matrix(g[k]).T *np.matrix(w).T
    Gamma = abs(a/b)    
    return w, Gamma    
#################################################

def get_uploading_time(deadline,D_k,D_k_new,phi,fmin,fmax,Beta,local_epochs):
  a=  deadline - ((((D_k_new)*10**(3)*phi* (local_epochs-1))/fmin) + (D_k*10**(3)*phi)/fmin)       
        
  b= deadline - ((((D_k_new)*10**(3)*phi* (local_epochs-1))/fmax) + (D_k*10**(3)*phi)/fmax)       
        
  Tupload= goldensectionsearch (a,b,alpha,phi,D_k_new*10**(3),model_size, Beta, local_epochs, deadline)  
  return Tupload
#################################################
def get_Ptransmit(Tupload, Beta):
    # print('Beta Function',Beta, ' model_size',model_size,'BW',BW,'Tupload',Tupload)
    tmp = (2**(model_size/(Tupload*BW)) - 1)
    # print('tmp',tmp)
    P_up = (tmp/Beta)
    return P_up

def get_E_cmp(T_upload,D_k, local_epochs, deadline):
    E = ((alpha/2)*(phi*D_k)**3*local_epochs**2)/(deadline-T_upload)**2 
    return E
