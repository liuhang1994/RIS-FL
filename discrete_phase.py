# -*- coding: utf-8 -*-
import time
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
import argparse
import torch
#from torch import nn
#from Nets import MLP,CNNCifar,CNNMnist
import flow
import DC_DS
import MIMO
from optlib import Gibbs
import DC_RIS
from proj import proj_theta
def initial():
    libopt = argparse.ArgumentParser()
    libopt.add_argument('--M', type=int, default=40, help='total # of devices')
    libopt.add_argument('--N', type=int, default=5, help='# of BS antennas')
    libopt.add_argument('--L', type=int, default=40, help='RIS Size')
    libopt.add_argument('--bit', type=np.ndarray, default=np.array([1,2,3,np.inf]), help='phase shift resolution')
    
    # optimization parameters
    libopt.add_argument('--nit', type=int, default=100, help='I_max,# of maximum SCA loops')                   
    libopt.add_argument('--Jmax', type=int, default=50, help='# of maximum Gibbs Outer loops')
    libopt.add_argument('--threshold', type=float, default=1e-2, help='epsilon,SCA early stopping criteria')
    libopt.add_argument('--tau', type=float, default=1, help=r'\tau, the SCA regularization term')  
    
    
    
    # simulation parameters
    libopt.add_argument('--trial', type=int, default=1, help='# of Monte Carlo Trials')  
    libopt.add_argument('--SNR', type=float, default=90.0, help='noise variance/0.1W in dB')  
    libopt.add_argument('--verbose', type=int, default=0, help=r'whether output or not')      
    libopt.add_argument('--set', type=int, default=2, help=r'=1 if concentrated devices+ euqal dataset;\
                        =2 if two clusters + unequal dataset')
    libopt.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # learning parameters
    libopt.add_argument('--gpu', type=int, default=1, help=r'Use Which Gpu')
    libopt.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    libopt.add_argument('--local_bs', type=int, default=0, help="0 for no effect,Local Bath size B")
    libopt.add_argument('--lr', type=float, default=0.01, help="learning rate,lambda")
    libopt.add_argument('--momentum', type=float, default=0.9, help="SGD momentum, used only for multiple local updates")
    libopt.add_argument('--epochs', type=int, default=500, help="rounds of training,T")
    args = libopt.parse_args()
    return args
    

if __name__ == '__main__':
    libopt = initial()
    np.random.seed(libopt.seed)
    print(libopt)
    filename='./store/discrete_trial_{}_M_{}_N_{}_L_{}_SNR_{}_Tau_{}_set_{}.npz'.format(libopt.trial,
                            libopt.M,
                            libopt.N,libopt.L,
                            libopt.SNR,libopt.tau,libopt.set)   
    print('save result to: \n {}'.format(filename))
    libopt.alpha_direct=3.76; # User-BS Path loss exponent
    fc=915*10**6 #carrier frequency, wavelength lambda=3.0*10**8/fc
    BS_Gain=10**(5.0/10) #BS antenna gain
    RIS_Gain=10**(5.0/10) #RIS antenna gain
    User_Gain=10**(0.0/10) #User antenna gain
    d_RIS=1.0/10 #dimension length of RIS element/wavelength
    libopt.BS=np.array([-50,0,10])
    libopt.RIS=np.array([0,0,10])
        
    libopt.range=20;
    x0=np.ones([libopt.M],dtype=int)
    
    
    SCA_Gibbs=np.ones([libopt.Jmax+1,libopt.trial])*np.nan
    DC_NORIS_set=np.ones([libopt.trial,])*np.nan
    DC_NODS_set=np.ones([libopt.trial,])*np.nan
    Alt_Gibbs=np.ones([libopt.Jmax+1,libopt.trial])*np.nan
    DG_NORIS=np.ones([libopt.trial,])*np.nan
    

    result_set=[]
    result_CNN_set=[]
    result_CNN_MB_set=[]
    
    
    
    for i in range(libopt.trial):
        libopt.device = torch.device('cuda:{}'.format(libopt.gpu) if torch.cuda.is_available() and libopt.gpu != -1 else 'cpu')
        print(libopt.device)
        print('This is the {0}-th trial'.format(i))
        
        ref=(1e-10)**0.5
        sigma_n=np.power(10,-libopt.SNR/10)
        libopt.sigma=sigma_n/ref**2
        
        
        if libopt.set==1:
            libopt.K=np.ones(libopt.M,dtype=int)*int(30000.0/libopt.M)
            print(sum(libopt.K))
            libopt.dx2=np.random.rand(int(libopt.M-np.round(libopt.M/2)))*libopt.range-libopt.range#[100,100+range]
        else:
            libopt.K=np.random.randint(1000,high=2001,size=(int(libopt.M)))
            lessuser_size=int(libopt.M/2)
            libopt.K2=np.random.randint(100,high=201,size=(lessuser_size))
            libopt.lessuser=np.random.choice(libopt.M,size=lessuser_size, replace=False)
            libopt.K[libopt.lessuser]=libopt.K2
            print(sum(libopt.K))
            libopt.dx2=np.random.rand(int(libopt.M-np.round(libopt.M/2)))*libopt.range+100


        libopt.dx1=np.random.rand(int(np.round(libopt.M/2)))*libopt.range-libopt.range #[-range,0]
        libopt.dx=np.concatenate((libopt.dx1,libopt.dx2))
        libopt.dy=np.random.rand(libopt.M)*20-10
        libopt.d_UR=((libopt.dx-libopt.RIS[0])**2+(libopt.dy-libopt.RIS[1])**2+libopt.RIS[2]**2
                     )**0.5
        libopt.d_RB=np.linalg.norm(libopt.BS-libopt.RIS)
        libopt.d_RIS=libopt.d_UR+libopt.d_RB
        libopt.d_direct=((libopt.dx-libopt.BS[0])**2+(libopt.dy-libopt.BS[1])**2+libopt.BS[2]**2
                         )**0.5             
        libopt.PL_direct=BS_Gain*User_Gain*(3*10**8/fc/4/np.pi/libopt.d_direct)**libopt.alpha_direct
        libopt.PL_RIS=BS_Gain*User_Gain*RIS_Gain*libopt.L**2*d_RIS**2/4/np.pi\
        *(3*10**8/fc/4/np.pi/libopt.d_UR)**2*(3*10**8/fc/4/np.pi/libopt.d_RB)**2
        #channels
        h_d=(np.random.randn(libopt.N,libopt.M)+1j*np.random.randn(libopt.N,libopt.M))/2**0.5
        h_d=h_d@np.diag(libopt.PL_direct**0.5)/ref
        H_RB=(np.random.randn(libopt.N,libopt.L)+1j*np.random.randn(libopt.N,libopt.L))/2**0.5
        h_UR=(np.random.randn(libopt.L,libopt.M)+1j*np.random.randn(libopt.L,libopt.M))/2**0.5
        h_UR=h_UR@np.diag(libopt.PL_RIS**0.5)/ref
        
        
        G=np.zeros([libopt.N,libopt.L,libopt.M],dtype = complex)
        for j in range(libopt.M):
            G[:,:,j]=H_RB@np.diag(h_UR[:,j])
        x=x0
        
        
        Noiseless=1
        Proposed=1
        start = time.time()
        print('Running the proposed algorithm')
        [x_store,obj_new,f_store,theta_store]=Gibbs(libopt,h_d,G,x,True,True)
        end = time.time()
        print("Running time: {} seconds".format(end - start))

        
        SCA_Gibbs[:,i]=obj_new

        
 
        NoDS=0
        if NoDS:
            start = time.time()
            print('Running DC algorithm for RIS optimiazation')
            obj_DC_RIS,F_DC_RIS,theta_DC_RIS=DC_RIS.DC_RIS(libopt,h_d,G,libopt.verbose)
            DC_NODS_set[i]=obj_DC_RIS
            end = time.time()
            print("Running time: {} seconds".format(end - start))
        else:
            obj_DC_RIS=np.array([0])
            F_DC_RIS=np.zeros([libopt.M,1])
            theta_DC_RIS=np.zeros([libopt.L,1])
        
        NoRIS=0
        if NoRIS:
            start = time.time()
#        [x_store_NORIS,obj_new_NORIS,f_store_NORIS,theta_store_NORIS]=Gibbs(
#                libopt,h_d,np.zeros([libopt.N,libopt.L,libopt.M]),x,False,True)
            gamma=[15]
            print('Running DC algorithm for device selection and beamforming (no RIS)')
            obj_new_NORIS,x_store_NORIS,f_store_NORIS=DC_DS.DC_NORIS(libopt,h_d,gamma,libopt.verbose)
            end = time.time()
            print("Running time: {} seconds".format(end - start))
            DC_NORIS_set[i]=obj_new_NORIS[0]
        else:
            obj_new_NORIS=[0]
            x_store_NORIS=np.zeros([libopt.M,])
            f_store_NORIS=np.zeros([libopt.N,])
        
        
        SVD=0
        if SVD:
            print('Running Differential Geometry Algorithm for Beamforming')
            obj_SVD,f_SVD=MIMO.SVD_MIMO(libopt,h_d,libopt.verbose)
            DG_NORIS[i]=obj_SVD
        else:
            obj_SVD=0
            f_SVD=np.zeros([libopt.N,])
        
        
        
        print('Algorithm2:{:.6f} Algorithm1:{:.6f} NoRIS:{:.6f} NoDS:{:.6f} DG:{:.6f}'.format(
            obj_new[libopt.Jmax],
            obj_new[0],obj_new_NORIS[0],obj_DC_RIS[len(obj_DC_RIS)-1],obj_SVD))


        dic={}
        h1=np.zeros([libopt.N,libopt.M,len(libopt.bit)],dtype=complex)
        
#        theta_store[:,libopt.Jmax]=np.exp(1j*2*np.pi*np.random.rand(libopt.L,))
        for ib in range(len(libopt.bit)):
            print('now is converting the result with bit {}'.format(libopt.bit[ib]))
            
            theta=copy.deepcopy(proj_theta(theta_store[:,libopt.Jmax],libopt.bit[ib]))
#            print(theta)
            for i in range(libopt.M):
                if Proposed:
                    h1[:,i,ib]=h_d[:,i]+G[:,:,i]@theta
            print('the mean power is {}'.format(np.mean(np.abs(h1[:,:,ib])**2)))
        dic['x_store']=copy.deepcopy(x_store)
        dic['f_store']=copy.deepcopy(f_store)
        libopt.transmitpower=0.1

        start = time.time()
        libopt.lr=0.01
        libopt.epochs=500
        result,_=flow.learning_flow2(libopt,Noiseless,Proposed,
                                    h1,dic)
        end = time.time()
        print("Running time: {} seconds".format(end - start))
        result_CNN_set.append(result)
        
        
#        start = time.time()
#        libopt.lr=0.005
#        libopt.epochs=100
#        libopt.local_bs=128
#        result,_=flow.learning_flow2(libopt,Noiseless,Proposed,
#                                    h1,dic)
#        result_CNN_MB_set.append(result)
#        end = time.time()
#        print("Running time: {} seconds".format(end - start))
        
    np.savez(filename,vars(libopt),result_set,result_CNN_set,
                 result_CNN_MB_set,SCA_Gibbs,Alt_Gibbs,DC_NORIS_set,DC_NODS_set,DG_NORIS)
#    
#    
#    filename2='./store/discrete_{}_M_{}_N_{}_L_{}.npz'.format(libopt.trial,
#                            libopt.M,
#                            libopt.N,libopt.L)  
#    np.savez(filename2,vars(libopt),theta_store,h_d,G,x_store,f_store)
#    