# -*- coding: utf-8 -*-

#import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)


def SVD_MIMO(libopt,h,verbose):
    N=libopt.N
    M=libopt.M
    K=libopt.K
    K2=K**2
    
    G=np.zeros([N,N],dtype='complex')
    for m in range(M):
        G+=np.outer(h[:,m],h[:,m].conj())
        
    _,V=np.linalg.eigh(G)
    f=V[:,N-1]
    f=f/np.linalg.norm(f)
    
    gain=K2/(np.abs(np.conjugate(f)@h)**2)*libopt.sigma
    obj=np.max(gain)/(sum(K))**2
    if verbose:
        print('obj={:.6f}\n'.format(obj))
    
    
    return obj,f
    