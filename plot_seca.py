# -*- coding: utf-8 -*-

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')
#from pylab import *
import copy

def pl(result,num1,num2):
#    params = {
#            'axes.labelsize': 10,
##            'text.fontsize': 8,
#            'legend.fontsize': 12,
#            'xtick.labelsize': 12,
#            'ytick.labelsize': 12,
#            'text.usetex': True,
#            'figure.figsize': [6, 6]
#            }
#    rcParams.update(params)
    
    plt.figure(num1)
    plt.plot(range(len(result['accuracy_test'])), result['accuracy_test'],label=r'Noiseless Channel')
    plt.plot(range(len(result['accuracy_test1'])),result['accuracy_test1'],label=r'The Proposed Algorithm')
    plt.plot(range(len(result['accuracy_test3'])), result['accuracy_test3'],label=r'Wuthout RIS')
    plt.plot(range(len(result['accuracy_test2'])),result['accuracy_test2'],label=r'DC Programming')
    plt.plot(range(len(result['accuracy_test5'])), result['accuracy_test5'],label=r'Deffiential Geometry')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Training Round')
    plt.legend()
    
    plt.figure(num2)
    plt.plot(range(len(result['loss_train'])), result['loss_train'],label=r'Noiseless Channel')
    plt.plot(range(len(result['loss_train1'])), result['loss_train1'],label=r'The Proposed Algorithm')
    plt.plot(range(len(result['loss_train3'])), result['loss_train3'],label=r'Wuthout RIS')
    plt.plot(range(len(result['loss_train2'])), result['loss_train2'],label=r'DC Programming')
    plt.plot(range(len(result['loss_train5'])), result['loss_train5'],label=r'Deffiential Geometry')
    
    plt.ylabel('Training Loss')
    plt.xlabel('Training Round')
    plt.legend()
    plt.ylim([0, 50])
    len1=len(result['accuracy_test'])
    a=np.zeros([5,len1])
    a[0,:]=result['accuracy_test']
    a[1,:]=result['accuracy_test1']
    a[2,:]=result['accuracy_test3']
    a[3,:]=result['accuracy_test2']
    a[4,:]=result['accuracy_test5']
    return a

def plot_figure_mse(result,num1,num2):
    len1=len(result['loss_test'])
    return result['loss_test'][len1-1],result['loss_test1'][len1-1],\
result['loss_test3'][len1-1],result['loss_test2'][len1-1],result['loss_test5'][len1-1]


def plot_figure(result,num1,num2):
    len1=len(result['accuracy_test'])
    return result['accuracy_test'][len1-1],result['accuracy_test1'][len1-1],\
result['accuracy_test3'][len1-1],result['accuracy_test2'][len1-1],result['accuracy_test5'][len1-1],\
result['loss_train'][len1-2],result['loss_train1'][len1-2],\
result['loss_train3'][len1-2],result['loss_train2'][len1-2],result['loss_train5'][len1-2]


def find_conver(dic,threshold):

    len1=len(dic['accuracy_test'])
    ret=np.ones([5,])*len1*np.nan
    for i in range(len1):
        if dic['accuracy_test'][i]>threshold:
            ret[0]=i
            break
    for i in range(len1):
        if dic['accuracy_test1'][i]>threshold:
            ret[1]=i
            break
    for i in range(len1):
        if dic['accuracy_test3'][i]>threshold:
            ret[2]=i
            break
    for i in range(len1):
        if dic['accuracy_test2'][i]>threshold:
            ret[3]=i
            break
    for i in range(len1):
        if dic['accuracy_test5'][i]>threshold:
            ret[4]=i
            break
        
    return ret






#/result['accuracy_test'][len1-1]


if __name__ == '__main__':
    M_set=[10,20,30,40,50,60]
#    M_set=[40]

    Noiseless=np.zeros([len(M_set),5])
    Proposed=np.zeros([len(M_set),5])
    TWC=np.zeros([len(M_set),5])
    DC=np.zeros([len(M_set),5])
    DG=np.zeros([len(M_set),5])
    BGD_acc=np.zeros([5,len(M_set)])
    MBGD_acc=np.zeros([5,len(M_set)])
    OBJ_acc=np.zeros([5,len(M_set)])
    BGD_cov=np.zeros([5,len(M_set)])
    MBGD_cov=np.zeros([5,len(M_set)])
    #SNR{}/
    thres=0.7
    trial=50
    SNR=90.0
    testmode=2
    for m in range(len(M_set)):
        mm=M_set[m]
        filename='./store/trial_{}_M_{}_N_{}_L_{}_\
SNR_{}_Tau_{}_set_{}.npz'.format(trial,
                40,5,mm,SNR,1,1)
        a=np.load(filename,allow_pickle=1)
        result_CNN_set=a['arr_2']
        res_CNN={}
        res_CNN_MB={}      
        mn=0
        for i in range(trial):
    #        print(i)
            if i==0:
                    res_CNN=copy.deepcopy(result_CNN_set[0])
            else:
                    for item in res_CNN.keys():
                        res_CNN[item]+=copy.deepcopy(result_CNN_set[i][item])

        for item in res_CNN.keys():
            res_CNN[item]=copy.deepcopy(res_CNN[item]/trial)



        a,b,c,d,e,f,g,h,i,j=plot_figure(res_CNN,3,4)
        BGD_acc[0,m]=a
        BGD_acc[1,m]=b
        BGD_acc[2,m]=c
        BGD_acc[3,m]=d
        BGD_acc[4,m]=e


    
    
    







    