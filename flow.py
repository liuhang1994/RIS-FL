# -*- coding: utf-8 -*-


#import matplotlib
#matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import copy
import numpy as np
# import argparse
np.set_printoptions(precision=6,threshold=1e3)
import torch
# from torch import nn
# from Gibbs_main import initial

#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
#from torchvision import models
#import torch.nn.functional as F


from Nets import CNNMnist
from AirComp import transmission
import train_script
# import mse_lib
# from torch.utils.data import DataLoader
# from sklearn.datasets import make_blobs
# from torch.utils.data.sampler import SubsetRandomSampler

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
    
    net_glob.eval()
    acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
    accuracy_test.append(acc_test)
    net_glob.train()
    for iter in range(libopt.epochs):
        #print('Overall Epoch: {}'.format(iter))
        grad_store_per_iter=np.zeros([len_active,d])
        #w_locals=[]
        loss_locals = []
        ind=0
        for idx in idxs_users:
            #print('Active User: {}'.format(idx))
            #print(len(dict_users[idx]))
            #print(int(libopt.K[idx]))
            if libopt.local_bs==0:
                size=int(libopt.K[idx])
            else:
                size=min(int(libopt.K[idx]),libopt.local_bs)
                
            w,loss,gradient=train_script.local_update(libopt,d,copy.deepcopy(net_glob).to(libopt.device),
                                         train_images,train_labels,idx,size)
        
            
            loss_locals.append(copy.deepcopy(loss))
            copyg=copy.deepcopy(gradient)
            copyg[np.isnan(copyg)]=1e2
            
            copyg[copyg>1e2]=1e2
            copyg[copyg<-1e2]=-1e2
            grad_store_per_iter[ind,:]=copyg
            ind=ind+1
        if trans_mode==0:
            grad=np.average(copy.deepcopy(grad_store_per_iter),axis=0,weights=
                            libopt.K[idxs_users]/sum(libopt.K[idxs_users]))
        elif trans_mode==1:
            grad=transmission(libopt,d,copy.deepcopy(grad_store_per_iter),
                                    x,f,h)
#            print(mse)
        # if libopt.verbose:
            # print(np.mean(np.abs(grad_store_per_iter)**2))
            # print(np.mean(np.abs(grad)**2))
        
        grad[grad>1e2]=1e2
        grad[grad<-1e2]=-1e2
#        print(grad)
        w_glob=copy.deepcopy(FedAvg_grad(w_glob,libopt.lr*grad,libopt.device))
        net_glob.load_state_dict(w_glob)
        #loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if libopt.verbose:
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
        accuracy_test.append(acc_test)
        loss_test_set.append(loss_test)
        net_glob.train()
    return loss_train,accuracy_test,loss_test_set
def Learning_iter3(libopt,d,net_glob,w_glob,
              train_images,train_labels,test_images,test_labels,
              x_store,f_store,h_store):
    
    idxs_users=np.asarray(range(libopt.M))
    loss_train = []
    accuracy_test=[]
    loss_test_set=[]
    
    net_glob.eval()
    acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
    accuracy_test.append(acc_test)
    net_glob.train()
    
    
    
    for iter in range(libopt.epochs):
        
        realization=int(iter/100)
        
        idxs_users2=idxs_users[x_store[realization]==1]
        len_active=len(idxs_users2)
        #print('Overall Epoch: {}'.format(iter))
        grad_store_per_iter=np.zeros([len_active,d])
        #w_locals=[]
        loss_locals = []
        ind=0
        for idx in idxs_users2:
            #print('Active User: {}'.format(idx))
            #print(len(dict_users[idx]))
            #print(int(libopt.K[idx]))
            if libopt.local_bs==0:
                size=int(libopt.K[idx])
            else:
                size=min(int(libopt.K[idx]),libopt.local_bs)
                
            w,loss,gradient=train_script.local_update(libopt,d,copy.deepcopy(net_glob).to(libopt.device),
                                         train_images,train_labels,idx,size)
        
            
            loss_locals.append(copy.deepcopy(loss))
            copyg=copy.deepcopy(gradient)
            copyg[np.isnan(copyg)]=1e2
            
            copyg[copyg>1e2]=1e2
            copyg[copyg<-1e2]=-1e2
            grad_store_per_iter[ind,:]=copyg
            ind=ind+1
            

        grad=transmission(libopt,d,copy.deepcopy(grad_store_per_iter),
                                    x_store[realization],
                                    f_store[realization],h_store[realization])
#            print(mse)
        # if libopt.verbose:
            # print(np.mean(np.abs(grad_store_per_iter)**2))
            # print(np.mean(np.abs(grad)**2))
        
        grad[grad>1e2]=1e2
        grad[grad<-1e2]=-1e2
#        print(grad)
        w_glob=copy.deepcopy(FedAvg_grad(w_glob,libopt.lr*grad,libopt.device))
        net_glob.load_state_dict(w_glob)
        #loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if libopt.verbose:
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        acc_test, loss_test = train_script.test_model(net_glob, libopt,test_images,test_labels)
        accuracy_test.append(acc_test)
        loss_test_set.append(loss_test)
        net_glob.train()
    return loss_train,accuracy_test,loss_test_set
def learning_flow(libopt,Noiseless,Proposed,NoRIS,NoDS,SVD,
                  h_d,G,dic):
    
    
    x_store=dic['x_store']
    f_store=dic['f_store']
    theta_store=dic['theta_store']

    
    x_store_NORIS=dic['x_store_NORIS']
    f_store_NORIS=dic['f_store_NORIS']
    f_store_NODS=dic['f_store_NODS']
    theta_store_NODS=dic['theta_store_NODS']
    f_SVD=dic['f_SVD']
    
    torch.manual_seed(libopt.seed)
    result={}
    

    train_images,train_labels,test_images,test_labels=train_script.Load_FMNIST_IID(libopt.M,libopt.K)
    net_glob = CNNMnist(num_classes=10,num_channels=1,batch_norm=True).to(libopt.device)
        


        
#    img_size = train_images[0][0].shape
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
        idxs_users=range(libopt.M)
    
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,
                                               0,None,None,None)
        result['loss_train']=np.asarray(loss_train)
        result['accuracy_test']=np.asarray(accuracy_test)
        result['loss_test']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test'][len(result['accuracy_test'])-1]))
    
    
    

        
    if Proposed:
        theta1=theta_store[:,libopt.Jmax]
        theta2=theta_store_NODS
        
    
    
        h1=np.zeros([libopt.N,libopt.M],dtype=complex)
        h2=np.zeros([libopt.N,libopt.M],dtype=complex)
        for i in range(libopt.M):
            if Proposed:
                h1[:,i]=h_d[:,i]+G[:,:,i]@theta1
            if NoDS:
                h2[:,i]=h_d[:,i]+G[:,:,i]@theta2
        
        print('Proposed Algorithm is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
#    
        
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


    if NoDS:
        print('No Device Selection Case is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
        idxs_users=np.asarray(range(libopt.M))
        loss_train2,accuracy_test2,loss_test2=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 np.ones([libopt.M,]),f_store_NODS,h2
                                                 )
        
        result['loss_train2']=np.asarray(loss_train2)
        result['accuracy_test2']=np.asarray(accuracy_test2)
        result['loss_test2']=np.asarray(loss_test2)
        print('result {}'.format(result['accuracy_test2'][len(result['accuracy_test2'])-1]))
    if NoRIS:
        print('No RIS Case is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
    
        idxs_users=np.asarray(range(libopt.M))
        
#        print(x_store_NORIS==1)
#        print(idxs_users[x_store_NORIS==1])
        idxs_users=idxs_users[x_store_NORIS[:,0]==1]
        loss_train3,accuracy_test3,loss_test3=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 x_store_NORIS[:,0],f_store_NORIS[:,0],h_d
                                                 )
        result['loss_train3']=np.asarray(loss_train3)
        result['accuracy_test3']=np.asarray(accuracy_test3)
        result['loss_test3']=np.asarray(loss_test3)
        print('result {}'.format(result['accuracy_test3'][len(result['accuracy_test3'])-1]))
    
    if SVD:
        print('MIMO Beamforming Case is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
        idxs_users=np.asarray(range(libopt.M))
        loss_train5,accuracy_test5,loss_test5=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 np.ones([libopt.M,]),f_SVD,h_d
                                                 )
        
        result['loss_train5']=np.asarray(loss_train5)
        result['accuracy_test5']=np.asarray(accuracy_test5)
        result['loss_test5']=np.asarray(loss_test5)
        print('result {}'.format(result['accuracy_test5'][len(result['accuracy_test5'])-1]))
    
    return result,d
    
def learning_flow3(libopt,Noiseless,Proposed,NoDS,
                  dic):
    x_store=dic['x_store']
    f_store=dic['f_store']
    h1=dic['h_store']
    f_store_NODS=dic['f_store_NODS']
    h2=dic['h_store_NODS']

    x_store2=[]
    for tri in range(libopt.total_time_trial):
        x_store2.append(np.ones([40,],dtype=int))

    torch.manual_seed(libopt.seed)
    result={}
    train_images,train_labels,test_images,test_labels=train_script.Load_FMNIST_IID(libopt.M,libopt.K)
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
        idxs_users=range(libopt.M)
    
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,
                                               0,None,None,None)
        result['loss_train']=np.asarray(loss_train)
        result['accuracy_test']=np.asarray(accuracy_test)
        result['loss_test']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test'][len(result['accuracy_test'])-1]))
    
    
    if Proposed:
        print('Proposed Algorithm is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
#    
        
        
#        idxs_users=np.asarray(range(libopt.M))
    
#        idxs_users=idxs_users[x_store[libopt.Jmax]==1]
        loss_train1,accuracy_test1,loss_test1=Learning_iter3(libopt,d,net_glob,w_glob,
                                                             train_images,train_labels,test_images,test_labels,
                                                             x_store,f_store,h1)
        result['loss_train1']=np.asarray(loss_train1)
        result['accuracy_test1']=np.asarray(accuracy_test1)
        result['loss_test1']=np.asarray(loss_test1)
        print('result {}'.format(result['accuracy_test1'][len(result['accuracy_test1'])-1]))


    if NoDS:
        print('No Device Selection Case is running')
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
#        idxs_users=np.asarray(range(libopt.M))
        loss_train2,accuracy_test2,loss_test2=Learning_iter3(libopt,d,net_glob,w_glob,
                                                             train_images,train_labels,test_images,test_labels,
                                                             x_store2,f_store_NODS,h2)
        
        result['loss_train2']=np.asarray(loss_train2)
        result['accuracy_test2']=np.asarray(accuracy_test2)
        result['loss_test2']=np.asarray(loss_test2)
        print('result {}'.format(result['accuracy_test2'][len(result['accuracy_test2'])-1]))
    
    return result,d

def learning_flow2(libopt,Noiseless,Proposed,h,dic):
    
    
    x_store=dic['x_store']
    f_store=dic['f_store']
    torch.manual_seed(libopt.seed)
    result={}
    

    train_images,train_labels,test_images,test_labels=train_script.Load_FMNIST_IID(libopt.M,libopt.K)
    net_glob = CNNMnist(num_classes=10,num_channels=1,batch_norm=True).to(libopt.device)
        

    if libopt.verbose:
        print(net_glob)
    w_glob = net_glob.state_dict()
    w_0=copy.deepcopy(w_glob)
    d=0
    for item in w_glob.keys():
        d=d+int(np.prod(w_glob[item].shape))
    print('Total Number of Parameters={}'.format(d))
    
        
    if Proposed:
        print('Proposed Algorithm is running')
        idxs_users=np.asarray(range(libopt.M))
        idxs_users=idxs_users[x_store[libopt.Jmax]==1]
        for ib in range(len(libopt.bit)):
            print('--now {} bit case is running'.format(libopt.bit[ib]))
        
            w_glob=copy.deepcopy(w_0)
            net_glob.load_state_dict(w_glob)
            
            a='loss_train_bit_{}'.format(libopt.bit[ib])
            b='accuracy_test_bit_{}'.format(libopt.bit[ib])
            c='loss_test_bit_{}'.format(libopt.bit[ib])
#    
            loss_train1,accuracy_test1,loss_test1=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                                 train_images,train_labels,test_images,test_labels,1,
                                                 x_store[libopt.Jmax],f_store[:,libopt.Jmax],h[:,:,ib]
                                                 )
#            loss_train1=np.array([0])
#            accuracy_test1=np.array([0])
#            loss_test1=np.array([0])
            result[a]=copy.deepcopy(np.asarray(loss_train1))
            result[b]=copy.deepcopy(np.asarray(accuracy_test1))
            result[c]=copy.deepcopy(np.asarray(loss_test1))
            print('result {}'.format(result[b][len(result[b])-1]))

    if Noiseless:
        print('Noiseless Case is running')
        idxs_users=range(libopt.M)
        w_glob=copy.deepcopy(w_0)
        net_glob.load_state_dict(w_glob)
        loss_train,accuracy_test,loss_test=Learning_iter(libopt,d,net_glob,w_glob,idxs_users,
                                               train_images,train_labels,test_images,test_labels,
                                               0,None,None,None)
        result['loss_train']=np.asarray(loss_train)
        result['accuracy_test']=np.asarray(accuracy_test)
        result['loss_test']=np.asarray(loss_test)
        print('result {}'.format(result['accuracy_test'][len(result['accuracy_test'])-1]))
     
    
    return result,d



if __name__ == '__main__':
    pass