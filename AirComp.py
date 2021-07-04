# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
#import copy
# import argparse


def transmission(libopt,d,signal,x,f,h):
    index=(x==1)
    #I=sum(x)
    N=libopt.N
    #M=libopt.M
    K=libopt.K[index]
    K2=K**2
    #print(x)

    
    
    inner=f.conj()@h[:,index]
    inner2=np.abs(inner)**2
    #print(inner[index])
    g=signal
    #mean and variance
    mean=np.mean(g,axis=1)
    g_bar=K@mean
    
    
    
    var=np.var(g,axis=1)
    
    var[var<1e-3]=1e-3
#    if min(var)<1e-5:
#         var=1
         
    var_sqrt=var**0.5
#    print(g)
#    print(mean)
#    print(var)
    #weighted-sum mean
    
    
    eta=np.min(libopt.transmitpower*inner2/K2/var)
    eta_sqrt=eta**0.5
    b=K*eta_sqrt*var_sqrt*inner.conj()/inner2
    
    
    noise_power=libopt.sigma*libopt.transmitpower
    
    
    n=(np.random.randn(N,d)+1j*np.random.randn(N,d))/(2)**0.5*noise_power**0.5
#    n=0
    x_signal=np.tile(b/var_sqrt,(d,1)).T*(g-np.tile(mean,(d,1)).T)
    y=h[:,index]@x_signal+n
    w=np.real((f.conj()@y/eta_sqrt+g_bar))/sum(K)
    

    #print(abs(inner*b/eta_sqrt/var_sqrt-K))
    
        
#    true_w=K@g/sum(K)
#    avg_mse=np.linalg.norm((true_w-w))**2/d
    
    #print(np.linalg.norm((true_w-w))**2/d)
    #print(libopt.sigma/eta/sum(K)**2)
    #print(libopt.sigma/sum(K)**2*np.max(K2*np.linalg.norm(g,axis=1)**2/inner2)/d)
    return w


if __name__ == '__main__':
   pass
    
    