# -*- coding: utf-8 -*-

import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)

def proj_theta(theta,bit):
    if np.isinf(bit):
        return theta
    L_bit=2**bit
    discrete_set=np.arange(0,L_bit)
    pro_set=np.exp(1j*discrete_set*1.0*2*np.pi/L_bit)
    
#    print(pro_set)
    L=len(theta)
    theta_r=np.zeros([L,],dtype=complex)
    for l in range(L):
#        print(l)
#        print(theta[l])
#        print(np.abs(theta[l]-pro_set))
#        print(np.argmin(np.abs(theta[l]-pro_set)))
        theta_r[l]=pro_set[np.argmin(np.abs(theta[l]-pro_set))]
#        print(theta_r[l])
        
#    print(theta)
#    print(np.abs(theta-1))
#    print(np.abs(theta+1))
#    print(theta_r)
    return theta_r


if __name__ == '__main__':
    
    
    theta_store=a['arr_1']
    h_d=a['arr_2']
    G=a['arr_3']
    x_store=a['arr_4']
    f_store=a['arr_5']
    Jmax=50
    
    bit_list=[1,2,3,np.inf]
    
#    libopt.M=40;
#    libopt.N=5;
    
    h1=np.zeros([5,40],dtype=complex)
    for bit in bit_list:
        theta=proj_theta(theta_store[:,Jmax],bit)
        for i in range(40):
            h1[:,i]=h_d[:,i]+G[:,:,i]@theta
        print(np.mean(np.abs(h1)**2)*0.5)