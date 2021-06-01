# -*- coding: utf-8 -*-


import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
import torch
from Nets import CNNMnist
from AirComp import transmission
import train_script



def FedAvg_grad(w_glob,grad,device):
    ind=0
    w_return=copy.deepcopy(w_glob)
    
    for item in w_return.keys():
        a=np.array(w_return[item].size())
        if len(a):
            b=np.prod(a)
            w_return[item]=copy.deepcopy(w_return[item])-torch.from_numpy(
                    np.reshape(grad[ind:ind+b],a)).float().to(device)
            ind=ind+b
    return w_return



def Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
              train_images,train_labels,test_images,test_labels,
              trans_mode,x,f,h):
    
    
    len_active=len(idxs_users)
    loss_train = []
    accuracy_test=[]
    loss_test_set=[]
    
    #test the model at round 0
    net_glob.eval()
    acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
    accuracy_test.append(acc_test)
    
    # training the models
    net_glob.train()
    for iter in range(libopt.epochs):
        grad_store_per_iter=np.zeros([len_active,d])
        loss_locals = []
        ind=0
        for idx in idxs_users:
            if libopt.local_bs==0:
                #batch gradient desent, use the full training data
                size=int(libopt.K[idx])
            else:
                #mini-batch gradient desent, use local_bs data for each batch
                size=min(int(libopt.K[idx]),libopt.local_bs)
                
            #compute the gradient    
            w,loss,gradient=train_script.local_update(libopt,d,copy.deepcopy(net_glob).to(libopt.device),
                                         train_images,train_labels,idx,size)
        
            #store the local training loss
            loss_locals.append(copy.deepcopy(loss))
            #truncating too large entries 
            copyg=copy.deepcopy(gradient)
            copyg[np.isnan(copyg)]=1e2
            copyg[copyg>1e2]=1e2
            copyg[copyg<-1e2]=-1e2
            grad_store_per_iter[ind,:]=copyg
            ind=ind+1
            
            
        if trans_mode==0: #noise-less benchmark, global gradient is the weighted sum of local ones
            grad=np.average(copy.deepcopy(grad_store_per_iter),axis=0,weights=
                            libopt.K[idxs_users]/sum(libopt.K[idxs_users]))
        elif trans_mode==1: #Over-the-air computation for aggregating the gradient
            grad=transmission(libopt,d,copy.deepcopy(grad_store_per_iter),
                                    x,f,h)
        #truncating too large entries 
        grad[grad>1e2]=1e2
        grad[grad<-1e2]=-1e2
        #update the global model: w_{t+1}=w_t-lr*grad
        w_glob=copy.deepcopy(FedAvg_grad(w_glob,libopt.lr*grad,libopt.device))
        net_glob.load_state_dict(w_glob)
        
        #average training loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if libopt.verbose:
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        
        #testing accuracy/loss
        acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
        accuracy_test.append(acc_test)
        loss_test_set.append(loss_test)
        net_glob.train()
    return loss_train,accuracy_test,loss_test_set


def learning_flow(libopt,Noiseless,Proposed,
                  h_d,G,dic):
    
    #result: store learning results as a dictionary
    torch.manual_seed(libopt.seed)
    result={}
    
    #retrieve the optimization results
    x_store=dic['x_store']
    f_store=dic['f_store']
    theta_store=dic['theta_store']

    

    
    #initialize the training/testing data
    train_images,train_labels,test_images,test_labels=train_script.Load_FMNIST_IID(libopt.M,libopt.K)
    #define the learning model
    net_glob = CNNMnist(num_classes=10,num_channels=1,batch_norm=True).to(libopt.device)
    if libopt.verbose:
        print(net_glob)
    w_glob = net_glob.state_dict()
    w_0=copy.deepcopy(w_glob)
    d=0
    for item in w_glob.keys():
        d=d+int(np.prod(w_glob[item].shape))
    print('Total Number of Parameters={}'.format(d))
    
    
    
    
    if Noiseless:
        print('Noiseless Case is running')
        #all devices are active
        idxs_users=range(libopt.M)
        #learning iterations
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,
                                               0,None,None,None)
        
        result['loss_train']=np.asarray(loss_train)
        result['accuracy_test']=np.asarray(accuracy_test)
        result['loss_test']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test'][len(result['accuracy_test'])-1]))
    
    
    

        
    if Proposed:
        
        theta1=theta_store[:,libopt.Jmax]
        #the optimized channel
        h1=np.zeros([libopt.N,libopt.M],dtype=complex)
        for i in range(libopt.M):
                h1[:,i]=h_d[:,i]+G[:,:,i]@theta1
                
        
        print('Proposed Algorithm is running')
        
        #re-initialize the model
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)

        #select the m-th devices if x_m=1
        idxs_users=np.asarray(range(libopt.M))
        idxs_users=idxs_users[x_store[libopt.Jmax]==1]
        
        
        loss_train1,accuracy_test1,loss_test1=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 x_store[libopt.Jmax],f_store[:,libopt.Jmax],h1
                                                 )
        result['loss_train1']=np.asarray(loss_train1)
        result['accuracy_test1']=np.asarray(accuracy_test1)
        result['loss_test1']=np.asarray(loss_test1)
        print('result {}'.format(result['accuracy_test1'][len(result['accuracy_test1'])-1]))


    return result,d
    





if __name__ == '__main__':
    pass