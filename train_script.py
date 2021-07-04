# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
#import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
import torch

#from torch import nn, autograd
from torchvision import datasets, transforms
import copy
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader


def mnist_iid(dataset, K,M):
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(M):
        dict_users[i] = set(np.random.choice(all_idxs, int(K[i]), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def Load_FMNIST_IID(M,K):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                    ])
    dataset_train = datasets.FashionMNIST('./data/FASHION_MNIST/', download = True, train = True, transform = transform)
    dataset_test = datasets.FashionMNIST('./data/FASHION_MNIST/', download = True, train = False, transform = transform)
    
    loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    images, labels = next(enumerate(loader))[1]
    images, labels = images.numpy(), labels.numpy()
    train_images = []
    train_labels = []
    dict_users = {i: np.array([], dtype='int64') for i in range(M)}
    all_idxs = np.arange(len(labels))
    for i in range(M):
        dict_users[i] = set(np.random.choice(all_idxs, int(K[i]), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        train_images.append(images[list(dict_users[i])])
        train_labels.append(labels[list(dict_users[i])])
        
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)
    test_images, test_labels = next(enumerate(test_loader))[1]

    return train_images,train_labels,test_images.numpy(),test_labels.numpy()





def local_update(libopt,d,model1, train_images, train_labels, idx,batch_size):
    
    
    
    initital_weight = copy.deepcopy(model1.state_dict())
    
    model=copy.deepcopy(model1)
    model.train()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=libopt.lr, momentum=libopt.momentum)
    
#    optimizer = torch.optim.Adam(model.parameters(),lr=libopt.lr)
    epoch_loss = []
    images = np.array_split(train_images[idx], len(train_images[idx]) // batch_size)
    labels = np.array_split(train_labels[idx], len(train_labels[idx]) // batch_size)
    
    for epoch in range(libopt.local_ep):
        batch_loss = []
        for b_idx in range(len(images)):
            model.zero_grad()

            log_probs = model(torch.tensor(images[b_idx].copy(), device=libopt.device))
            local_loss = loss_function(log_probs, torch.tensor(labels[b_idx].copy(), device=libopt.device))

                
            local_loss.backward()
            optimizer.step()
            if libopt.verbose==2:
                print('User: {},Epoch: {}, Batch No: {}/{} Loss: {:.6f}'.format(idx,
                        epoch, b_idx+1,len(images), local_loss.item()))
            batch_loss.append(local_loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    copyw=copy.deepcopy(model.state_dict())
    gradient2=np.array([[]]);
    for item in copyw.keys():
            gradient2=np.hstack((gradient2,np.reshape((initital_weight[item]-copyw[item]).cpu().numpy(),
                                                    [1,-1])/libopt.lr))
#    print(copyw)
#    print((gradient2))
    return model.state_dict(),sum(epoch_loss) / len(epoch_loss),gradient2

def test_model(model,libopt,test_images,test_labels):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    accuracy=0

    images=torch.tensor(test_images).to(libopt.device)
    labels=torch.tensor(test_labels).to(libopt.device)
    outputs = model(images).to(libopt.device)
    loss_function = nn.CrossEntropyLoss()
    batch_loss = loss_function(outputs, labels)
    loss += batch_loss.item()
    _, pred_labels = torch.max(outputs, 1)
    pred_labels = pred_labels.view(-1)


    correct += torch.sum(torch.eq(pred_labels, labels)).item()
    total += len(labels)
    accuracy = correct / total
    
    if libopt.verbose:
        print('Average loss: {:.4f}   Accuracy: {}/{} ({:.2f}%)'.format(
            loss, int(correct), int(total), 100.0*accuracy))
    return accuracy, loss
