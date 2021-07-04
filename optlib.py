# -*- coding: utf-8 -*-
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
from scipy.optimize import minimize



def sca_fmincon(libopt,h_d,G,f,theta,x,K2,RISON):
    N=libopt.N
    L=libopt.L
    I=sum(x)
    tau=libopt.tau
    if theta is None:
        theta=np.ones([L],dtype=complex)
    if not RISON:
        theta=np.zeros([L],dtype=complex)
    result=np.zeros(libopt.nit)
    h=np.zeros([N,I],dtype=complex)
    for i in range(I):
        h[:,i]=h_d[:,i]+G[:,:,i]@theta
        
    if f is None:
        f=h[:,0]/np.linalg.norm(h[:,0])
   
    obj=min(np.abs(np.conjugate(f)@h)**2/K2)
    threshold=libopt.threshold
    
    for it in range(libopt.nit):
        obj_pre=copy.deepcopy(obj)
        a=np.zeros([N,I],dtype=complex)
        b=np.zeros([L,I],dtype=complex)
        c=np.zeros([1,I],dtype=complex)
        F_cro=np.outer(f,np.conjugate(f));
        for i in range(I):
            a[:,i]=tau*K2[i]*f+np.outer(h[:,i],np.conjugate(h[:,i]))@f
            if RISON:

                b[:,i]=tau*K2[i]*theta+G[:,:,i].conj().T@F_cro@h[:,i]
                c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]*(L+1)+2*np.real((theta.conj().T)@(G[:,:,i].conj().T)@F_cro@h[:,i])
            else:
                c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]
        
        
        #print(c.shape)
        
        fun=lambda mu: np.real(2*np.linalg.norm(a@mu)+2*np.linalg.norm(b@mu,ord=1)-c@mu)
        
        cons = ({'type': 'eq', 'fun': lambda mu:  K2@mu-1})
        bnds=((0,None) for i in range(I))
        res = minimize(fun, 1/K2,   bounds=tuple(bnds), constraints=cons)
        if ~res.success:
            pass
            #print('Iteration: {}, solution:{} obj:{:.6f}'.format(it,res.x,res.fun[0]))
            #print(res.message)
            #return
        fn=a@res.x
        thetan=b@res.x
        fn=fn/np.linalg.norm(fn)
#        thetan=thetan/np.abs(thetan)
        if RISON:
            thetan=thetan/np.abs(thetan)
            theta=thetan
        f=fn
        for i in range(I):
            h[:,i]=h_d[:,i]+G[:,:,i]@theta
        obj=min(np.abs(np.conjugate(f)@h)**2/K2)
        result[it]=copy.deepcopy(obj)
        if libopt.verbose>2:
            print('  Iteration {} Obj {:.6f} Opt Obj {:.6f}'.format(it,result[it],res.fun[0]))
        if np.abs(obj-obj_pre)/min(1,abs(obj))<=threshold:
            break
        
        #print(res)
    if libopt.verbose>1:
        print(' SCA Take {} iterations with final obj {:.6f}'.format(it+1,result[it]))
    result=result[0:it]
    return f,theta,result




def find_obj_inner(libopt,x,K,K2,Ksum2,h_d,G,f0,theta0,RISON):
    N=libopt.N
    L=libopt.L
    M=libopt.M
    if sum(x)==0:
        obj=np.inf
        
        theta=np.ones([L],dtype=complex)
        f=h_d[:,0]/np.linalg.norm(h_d[:,0])
        if not RISON:
            theta=np.zeros([L])
    else:
         index=(x==1)
         #print(index)

         f,theta,_=sca_fmincon(libopt,h_d[:,index],G[:,:,index],f0,theta0,x,K2[index],RISON)

         h=np.zeros([N,M],dtype=complex)
         for i in range(M):
             h[:,i]=h_d[:,i]+G[:,:,i]@theta
         gain=K2/(np.abs(np.conjugate(f)@h)**2)*libopt.sigma
         #print(gain)
         #print(gain)
         #print(2/Ksum2*(sum(K[~index]))**2)
         #print(np.max(gain[index])/(sum(K[index]))**2)
         obj=np.max(gain[index])/(sum(K[index]))**2+4/Ksum2*(sum(K[~index]))**2
    return obj,x,f,theta
def Gibbs(libopt,h_d,G,x0,RISON,Joint):
    #initial
    
    N=libopt.N
    L=libopt.L
    M=libopt.M
    Jmax=libopt.Jmax
    K=libopt.K/np.mean(libopt.K) #normalize K to speed up floating computation
    K2=K**2
    Ksum2=sum(K)**2
    x=x0
    # inital the return values
    obj_new=np.zeros(Jmax+1)
    f_store=np.zeros([N,Jmax+1],dtype = complex)
    theta_store=np.zeros([L,Jmax+1],dtype = complex)
    x_store=np.zeros([Jmax+1,M],dtype=int)
    
    #the first loop
    ind=0
    [obj_new[ind],x_store[ind,:],f,theta]=find_obj_inner(libopt,x,K,K2,Ksum2,h_d,G,None,None,RISON)
    
    theta_store[:,ind]=copy.deepcopy(theta)
    f_store[:,ind]=copy.deepcopy(f)
#    beta=min(max(obj_new[ind],1)
    beta=min(1,obj_new[ind])
    # print(beta)
    alpha=0.9
    if libopt.verbose>1:
        print('The inital guess: {}, obj={:.6f}'.format(x,obj_new[ind]))
    elif libopt.verbose==1:
        print('The inital guess obj={:.6f}'.format(obj_new[ind]))
    f_loop=np.tile(f,(M+1,1))

    theta_loop=np.tile(theta,(M+1,1))
    #print(theta_loop.shape)
    #print(theta_loop[0].shape)
    for j in range(Jmax):
        if libopt.verbose>1:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f}'.format(j+1,beta));
        
        #store the possible transition solution and their objectives
        X_sample=np.zeros([M+1,M],dtype=int)
        Temp=np.zeros(M+1)
        #the first transition => no change
        X_sample[0,:]=copy.deepcopy(x)
        Temp[0]=copy.deepcopy(obj_new[ind])
        f_loop[0]=copy.deepcopy(f)
        theta_loop[0]=copy.deepcopy(theta)
        #2--M+1-th trnasition, change only 1 position
        for m in range(M):
            if libopt.verbose>1:
                print('the {}-th:'.format(m+1))
            #filp the m-th position
            x_sam=copy.deepcopy(x)
            x_sam[m]=copy.deepcopy((x_sam[m]+1)%2)
            X_sample[m+1,:]=copy.deepcopy(x_sam);
            Temp[m+1],_,f_loop[m+1],theta_loop[m+1]=find_obj_inner(libopt,
                x_sam,K,K2,Ksum2,h_d,G,f_loop[m+1],theta_loop[m+1],RISON)
            if libopt.verbose>1:
                print('          sol:{} with obj={:.6f}'.format(x_sam,Temp[m+1]))
        temp2=Temp;
        
        Lambda=np.exp(-1*temp2/beta);
        Lambda=Lambda/sum(Lambda);
        while np.isnan(Lambda).any():
            if libopt.verbose>1:
                print('There is NaN, increase beta')
            beta=beta/alpha;
            Lambda=np.exp(-1.*temp2/beta);
            Lambda=Lambda/sum(Lambda);
        
        if libopt.verbose>1:
            print('The obj distribution: {}'.format(temp2))
            print('The Lambda distribution: {}'.format(Lambda))
        kk_prime=np.random.choice(M+1,p=Lambda)
        x=copy.deepcopy(X_sample[kk_prime,:])
        f=copy.deepcopy(f_loop[kk_prime])
        theta=copy.deepcopy(theta_loop[kk_prime])
        ind=ind+1
        obj_new[ind]=copy.deepcopy(Temp[kk_prime])
        x_store[ind,:]=copy.deepcopy(x)
        theta_store[:,ind]=copy.deepcopy(theta)
        f_store[:,ind]=copy.deepcopy(f)
        
        if libopt.verbose>1:
            print('Choose the solution {}, with objective {:.6f}'.format(x,obj_new[ind]))
            
        if libopt.verbose:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f},obj={:.6f}'.format(j+1,beta,obj_new[ind]));
        beta=max(alpha*beta,1e-4);
        
    return x_store,obj_new,f_store,theta_store