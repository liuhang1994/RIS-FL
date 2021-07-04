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
    plt.plot(range(len(result['accuracy_test1'])),result['accuracy_test1'],label=r'bit=1')
    plt.plot(range(len(result['accuracy_test2'])), result['accuracy_test2'],label=r'bit=2')
#    plt.plot(range(len(result['accuracy_test_bit_3.0'])),result['accuracy_test_bit_3.0'],label=r'bit=3')
#    plt.plot(range(len(result['accuracy_test_bit_inf'])), result['accuracy_test_bit_inf'],label=r'No quantizationy')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Training Round')
    plt.legend()
    
#    plt.figure(num2)
#    plt.plot(range(len(result['loss_train'])), result['loss_train'],label=r'Noiseless Channel')
#    plt.plot(range(len(result['loss_train1'])), result['loss_train1'],label=r'The Proposed Algorithm')
#    plt.plot(range(len(result['loss_train3'])), result['loss_train3'],label=r'Wuthout RIS')
#    plt.plot(range(len(result['loss_train2'])), result['loss_train2'],label=r'DC Programming')
#    plt.plot(range(len(result['loss_train5'])), result['loss_train5'],label=r'Deffiential Geometry')
#    
#    plt.ylabel('Training Loss')
#    plt.xlabel('Training Round')
#    plt.legend()
#    plt.ylim([0, 50])
#    len1=len(result['accuracy_test'])
#    a=np.zeros([5,len1])
#    a[0,:]=result['accuracy_test']
#    a[1,:]=result['accuracy_test1']
#    a[2,:]=result['accuracy_test3']
#    a[3,:]=result['accuracy_test2']
#    a[4,:]=result['accuracy_test5']
    return



def plot_figure(result,num1,num2):
    len1=len(result['accuracy_test'])
    
    return result['accuracy_test'][len1-1],result['accuracy_test1'][len1-1],\
result['accuracy_test2'][len1-1]







#/result['accuracy_test'][len1-1]


if __name__ == '__main__':
    M_set=[10,20,30,40,50,60]
    M_set=[1,2,3,4,5]

    #SNR{}/
    thres=0.7
    trial=10
    SNR=90.0
    testmode=2
    res_CNN_MB={}
    
    for m in range(len(M_set)):
        mm=M_set[m]
        filename='./store/vary_trial_{}_M_{}_N_{}_L_{}_\
SNR_{}_Tau_{}_seed_{}_onlyds.npz'.format(trial,
                40,5,40,SNR,1,mm)
        a=np.load(filename,allow_pickle=1)
        result_CNN_set=a['arr_1']
        for i in range(trial):
            if i==0 and mm==1:
                res_CNN=copy.deepcopy(result_CNN_set[0])
            else:
                for item in res_CNN.keys():
                    res_CNN[item]+=copy.deepcopy(result_CNN_set[i][item])
#                for item in res_CNN.keys():
#                    res_CNN_MB[item]+=copy.deepcopy(result_CNN_MB_set[i][item])
    for item in res_CNN.keys():
        res_CNN[item]=copy.deepcopy(res_CNN[item]/50)


    a,b,c=plot_figure(res_CNN,3,4)

    
    







    