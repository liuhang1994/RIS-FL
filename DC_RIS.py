# -*- coding: utf-8 -*-

#import argparse
#from scipy.optimize import minimize
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
import warnings
import cvxpy as cp
import sys

def DC_F(N,K,L,h_d,G,maxiter,theta_DC_RIS,verbose,epislon):
    rho=5
    h=np.zeros([N,K],dtype=complex)
    H=np.zeros([N,N,K],dtype=complex)
#    scale=10000
    for i in range(K):
        h[:,i]=h_d[:,i]+G[:,:,i]@theta_DC_RIS 
#        h[:,i]=h[:,i]*scale
        H[:,:,i]=np.outer(h[:,i],h[:,i].conj())
#    print(H)
    M = np.random.randn(N,1)+1j*np.random.randn(N,1);
    M= copy.deepcopy(np.outer(M,M.conj()))
    _,V=np.linalg.eigh(M)
    u=V[:,N-1]
#    for i in range(K):
#        print(np.trace(M@H[:,:,i]))
    # define the optimization problem
    M_var=cp.Variable((N,N), complex =True)
    M_partial=cp.Parameter((N,N), hermitian =True)
    M_partial.value = copy.deepcopy(np.outer(u,u.conj()))
#    print(M_partial.value)
    constraints = [M_var >> 0]
    constraints += [cp.real(M_var@H[:,:,k])>=1 for k in range(K)]
    cost=cp.real(cp.trace(M_var))+rho*cp.real(cp.trace((np.eye(N)-M_partial)@M_var))
    prob = cp.Problem(cp.Minimize(cost),constraints)
    obj0=0
    #iteritively solve:
    for iter in range(maxiter):
        if verbose>1:
            print('Solving f, Inner iter={}'.format(iter))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open('out.log','w+') as f:
                sys.stdout.flush()
                stream=sys.stdout
                sys.stdout=f
                prob.solve(solver=cp.SCS,verbose=False,scale=1e-10,max_iters=1000)
                sys.stdout.flush()
                sys.stdout=stream
        if prob.status=='infeasible' or prob.value is None:
            break
#        print(prob.value)
        err=np.abs(prob.value-obj0)
        M=copy.deepcopy(M_var.value)
        _,V=np.linalg.eigh(M)
        u=V[:,N-1]
        M_partial.value = copy.deepcopy(np.outer(u,u.conj()))
        
        obj0 = prob.value
        if err<epislon:
            break
    u,_,_=np.linalg.svd(M,compute_uv=True,hermitian=True)
    m=u[:,0]
    return m/np.linalg.norm(m)
def DC_theta(N,K,L,h_d,G,maxiter,f,verbose,epsilon):
    #Compute R,c:
    A=np.zeros([L,K],dtype=complex)
    c=np.zeros([K,],dtype=complex)
    R=np.zeros([L+1,L+1,K],dtype=complex)
    for k in range(K):
        c[k]=f.conj()@h_d[:,k]
        A[:,k]=(f.conj()@G[:,:,k])
        R[0:L,0:L,k]=np.outer(A[:,k],A[:,k].conj())
        R[0:L,L,k]=A[:,k]*c[k]
        R[L,0:L,k]=R[0:L,L,k].conj()
#        R[:,:,k]=copy.deepcopy((R[:,:,k].conj().T+R[:,:,k])/2)
    #initial V:
    V = np.random.randn(L+1,1)+1j*np.random.randn(L+1,1);
    V=V/np.abs(V)
    V= copy.deepcopy(np.outer(V,V.conj()))
#    for k in range(K):
#        print(np.trace(R[:,:,k]@V))
#        print(np.abs(c[k])**2)

    _,v=np.linalg.eigh(V)
#    print(v.shape)
    u=v[:,L]
    u= np.random.randn(L+1,1)+1j*np.random.randn(L+1,1);
    #initial other parameters:
    infeasible_check=False
    #initial the optimization problem:
    V_var=cp.Variable((L+1,L+1), hermitian =True)
    V_var.value=V
    V_partial=cp.Parameter((L+1,L+1), hermitian =True)
    V_partial.value = copy.deepcopy(np.outer(u,u.conj()))
#    print(M_partial.value)
    constraints = [V_var >> 0]
    constraints += [V_var[n,n]==1 for n in range(L)]
    constraints += [cp.real(V_var@R[:,:,k])+np.abs(c[k])**2>=1 for k in range(K)]
    cost=cp.real(cp.trace((np.eye(L+1)-V_partial)@V_var))
    prob = cp.Problem(cp.Minimize(cost),constraints)
    obj0=0
    for iter in range(maxiter):
        if verbose>1:
            print('Solving theta, Inner iter={}'.format(iter))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open('out.log','w+') as f:
                sys.stdout.flush()
                stream=sys.stdout
                sys.stdout=f
                prob.solve(solver=cp.SCS,verbose=False,scale=1e-10,max_iters=200,warm_start=True)
                sys.stdout.flush()
                sys.stdout=stream
        print(prob.value)
        if verbose:
            print('Status={}, Value={}'.format(prob.status,prob.value))
        if prob.status=='infeasible' or prob.value is None:
            infeasible_check=True
            break

        err=np.abs(prob.value-obj0)
        V=copy.deepcopy(V_var.value)
        _,v=np.linalg.eigh(V)
        u=v[:,L]
        V_partial.value = copy.deepcopy(np.outer(u,u.conj()))
        obj0 = prob.value
        if err<epsilon:
            break
    u,_,_=np.linalg.svd(V,compute_uv=True,hermitian=True)
    v_tilde=u[:,0]
    vv=v_tilde[0:L]/v_tilde[L]
    vv=copy.deepcopy(vv/np.abs(vv))
    return vv,infeasible_check

def DC_main(N,K,L,h_d,G,maxiter,iter_num,epsilon,verbose,epsilon2):
    F_DC_RIS=np.zeros([N,],dtype='complex')
#    theta_DC_RIS=np.zeros([L,],dtype='complex')
    theta_DC_RIS=np.ones([L],dtype=complex)
    
    h=np.zeros([N,K],dtype=complex)
    for i in range(K):
        h[:,i]=h_d[:,i]+G[:,:,i]@theta_DC_RIS
    obj_pre=min(np.abs(np.conjugate(F_DC_RIS)@h)**2)
    infeasible=False
    stop=False
    for iter in range(maxiter):
        if verbose:
            print('iter={}'.format(iter))
        #Given theta, update F
        F_DC_RIS=DC_F(N,K,L,h_d,G,iter_num,theta_DC_RIS,verbose,epsilon2)
#        print(F_DC_RIS.shape)
        #Given F, update theta
        theta_DC_RIS,infeasible=DC_theta(N,K,L,h_d,G,iter_num,F_DC_RIS,verbose,epsilon2)
        h=np.zeros([N,K],dtype=complex)
        for i in range(K):
            h[:,i]=h_d[:,i]+G[:,:,i]@theta_DC_RIS 
        obj=min(np.abs(np.conjugate(F_DC_RIS)@h)**2)
        if verbose:
            print('Gain value={}'.format(obj))
        if abs(obj-obj_pre)<epsilon or infeasible==True:
            stop=True
        obj_pre=obj
        if stop:
            break
        
    return F_DC_RIS,theta_DC_RIS
def DC_RIS(libopt,h_d,G,verbose):
    N=libopt.N
    M=libopt.M
    L=libopt.L
    K=libopt.K
    K2=K**2
#    Ksum2=sum(K)**2
    maxiter=50
    iter_num=50
#    maxiter=1
    epsilon=1e-3
    epsilon2=1e-8
#    obj_DC_RIS =0
    F_DC_RIS, theta_DC_RIS = DC_main(N,M,L,h_d,G,maxiter,iter_num,epsilon,verbose,epsilon2)
    h=np.zeros([N,M],dtype=complex)
    for i in range(M):
        h[:,i]=h_d[:,i]+G[:,:,i]@theta_DC_RIS
    gain=K2/(np.abs(np.conjugate(F_DC_RIS)@h)**2)*libopt.sigma
    obj=np.max(gain)/(sum(K))**2
    obj_DC_RIS=copy.deepcopy(obj)
    if verbose:
        print('obj={:.6f}\n'.format(obj))
    return obj_DC_RIS,F_DC_RIS,theta_DC_RIS

if __name__ == '__main__':
    pass