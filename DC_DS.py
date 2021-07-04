# -*- coding: utf-8 -*-

#import argparse
#from scipy.optimize import minimize
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
import warnings
import cvxpy as cp
import sys




def feasibility_DC(h,gamma,maxiter,epsilon):
    size=h.shape
    N=size[0]
    K=size[1]
    h_var = cp.Parameter((N, K),complex=True)
    h_var.value=copy.deepcopy(h)
    M_var=cp.Variable((N,N), hermitian =True)
    M_partial=cp.Parameter((N,N), hermitian =True)
    
    
    M = np.random.randn(N,N)+1j*np.random.randn(N,N);
    obj0 = 0
    _,V=np.linalg.eigh(M)
    u=V[:,N-1]
    M_partial.value = copy.deepcopy(np.outer(u,u.conj()))
    
    h_var = cp.Parameter((N, K),complex=True)
    h_var.value=copy.deepcopy(h)
    constraints = [cp.real(cp.trace(M_var))-1 >= 0]
    constraints += [M_var >> 0]
    constraints += [cp.real(cp.trace(M_var))-gamma*cp.real(h_var[:,k].H@M_var@h_var[:,k])<=0 for k in range(K)]
    cost=cp.real(cp.trace((np.eye(N)-M_partial.H)@M_var))

    prob = cp.Problem(cp.Minimize(cost),constraints)
    
    for iter in range(maxiter):
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open('out.log','w+') as f:
                    sys.stdout.flush()
                    stream=sys.stdout
                    sys.stdout=f
                    prob.solve(solver=cp.SCS,verbose=False)
                    sys.stdout.flush()
                    sys.stdout=stream
#        print(prob.status)
        if prob.status=='infeasible' or prob.value is None:
            return None,False
            
        err=abs(prob.value-obj0)
        M=copy.deepcopy(M_var.value)
#        print(M)
        _,V=np.linalg.eigh(M)
        u=V[:,N-1]
        
        M_partial.value = copy.deepcopy(np.outer(u,u.conj()))
        obj0 = prob.value
        if err<1e-9 or prob.value<1e-7:
            break
    
    u,s,_=np.linalg.svd(M,compute_uv=True,hermitian=True)
    m=u[:,0]
    feasibility=sum(s[1:])<1e-6
    if feasibility:
        for i in range(K):
            flag=np.linalg.norm(m)**2/np.linalg.norm(m.conj()@h[:,i])**2 <=gamma
            if not flag:
                feasibility=False
    
    
#    print(feasibility)
    return m,feasibility

def user_selection_DC(N,K,h,gamma,maxiter,epsilon,verbose):
    
    M_var=cp.Variable((N,N), hermitian =True
                      )
    x_var =cp.Variable(K,nonneg=True)
    x_partial=cp.Parameter(K)
    M_partial=cp.Parameter((N,N), hermitian =True
                      )
    

    constraints = [cp.real(cp.trace(M_var))-1 >= 0]
    constraints += [M_var >> 0]
    h_var = cp.Parameter((N, K),complex=True)
    h_var.value=copy.deepcopy(h)

    constraints+= [cp.real(cp.trace(M_var))-gamma*cp.real(h_var[:,k].H@M_var@h_var[:,k])-x_var[k]<=0 for k in range(K)]
    cost=cp.norm(x_var,1)-x_partial.H@x_var+cp.real(cp.trace((np.eye(N)-M_partial.H)@M_var))

 
    prob = cp.Problem(cp.Minimize(cost),constraints)
#    
    for c in range(K+1):
        x = np.random.randn(K,)
        M = np.random.randn(N,N)+1j*np.random.randn(N,N);
        M=M@M.conj().T
        x_abs = np.abs(x)
        x_p = np.zeros([K,])
        ind=np.argsort(-x_abs)
#        x_partial[ind[c+1:end]] = 0;
        x_p[ind[0:c]]=copy.deepcopy(np.sign(x[ind[0:c]]));
        
#        print(x_p)
        x_partial.value=copy.deepcopy(x_p)
        _,V=np.linalg.eigh(M)
        u=V[:,N-1]
        
        M_partial.value = copy.deepcopy(np.outer(u,u.conj()))
        
        obj0 = 0
        for iter in range(maxiter):
            if verbose:
                print('c={} iter={}'.format(c,iter))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
#                sys.stdout=r_obj
                with open('out.log','w+') as f:
                    stream=sys.stdout
                    sys.stdout.flush()
                    sys.stdout=f
                    prob.solve(solver=cp.SCS,verbose=False)
                    sys.stdout.flush()
                    sys.stdout=stream
#            print(prob.status)
            if prob.status=='infeasible' or prob.value is None:
                break
            x=copy.deepcopy(x_var.value)
            M=copy.deepcopy(M_var.value)
            err= abs(prob.value-obj0)
            x_p = np.zeros([K,])
            ind=np.argsort(-x_abs)
            x_p[ind[0:c]]=copy.deepcopy(np.sign(x[ind[0:c]]));
            x_partial.value=copy.deepcopy(x_p)
            _,V=np.linalg.eigh(M)
            u=V[:,N-1]
        
            M_partial.value = copy.deepcopy(np.outer(u,u.conj()))
            obj0 = prob.value
            if err<1e-9 or prob.value<1e-7:
                break
        s=np.linalg.svd(M,compute_uv=False,hermitian=True)
        feasibility = sum(s[1:])<1e-6
        if feasibility:
            break
        
    ind=np.argsort(x)
    
#    print(x)
#    print(ind[0:1])
    for i in np.arange(K):
        active_user_num=K-i
        active_user=np.asarray(ind[0:active_user_num])
#        print(active_user)
#        print(type(active_user))
        m, feasibility = feasibility_DC(h[:,active_user],gamma,maxiter,epsilon)
        if verbose:
            print('try user num: {}, feasible:{}'.format( active_user_num, feasibility))
        if feasibility:
#            num_of_users=active_user_num
            break
        
    if not feasibility:
        m=None
        active_user=[]
        
#            pa
    
    return m,active_user

def DC_NORIS(libopt,h_d,gamma_set,verbose):
    N=libopt.N
    M=libopt.M
    K=libopt.K/np.mean(libopt.K)
    K2=K**2
    Ksum2=sum(K)**2
    
    maxiter=100
#    maxiter=1
    epsilon=1e-5
    obj_DC =np.zeros([len(gamma_set),])
    X_DC=np.zeros([M,len(gamma_set)])
    F_DC=np.zeros([N,len(gamma_set)],dtype='complex')
    
    for i in range(len(gamma_set)):
        gamma=10**(gamma_set[i]/10)
        if verbose:
            print('gamma:{:.6f},\n'.format(gamma))
    
        m, active_user = user_selection_DC(N,M,h_d,gamma,maxiter,epsilon,verbose)
        
        x=np.zeros([M,])
        if verbose:
            print(x)
        if m is not None:
            F_DC[:,i]=copy.deepcopy(m)
            x[active_user]=1
            
        else:
            m=h_d[:,0]/np.linalg.norm(h_d[:,0])
            F_DC[:,i]=copy.deepcopy(m)
            x[0]=1
            
        
        
        X_DC[:,i]=copy.deepcopy(x)
        if not x.any():
            print('Selection is failed! no device selected, Check it.')
            obj=np.inf
        else:
            index=(x==1)
            gain=K2/(np.abs(np.conjugate(m)@h_d)**2)*libopt.sigma
            obj=np.max(gain[index])/(sum(K[index]))**2+4/Ksum2*(sum(K[~index]))**2
 
        obj_DC[i]=copy.deepcopy(obj)
        if verbose:
            print('obj={:.6f}\n'.format(obj))
    
    return obj_DC,X_DC,F_DC

if __name__ == '__main__':
    pass