# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
import copy


def pl(result, num1, num2):
    plt.figure(num1)
    plt.plot(range(len(result['accuracy_test'])), result['accuracy_test'], label=r'Noiseless Channel')
    plt.plot(range(len(result['accuracy_test1'])), result['accuracy_test1'], label=r'The Proposed Algorithm')
    #    plt.plot(range(len(result['accuracy_test3'])), result['accuracy_test3'],label=r'Wuthout RIS')
    #    plt.plot(range(len(result['accuracy_test2'])),result['accuracy_test2'],label=r'DC Programming')
    #    plt.plot(range(len(result['accuracy_test5'])), result['accuracy_test5'],label=r'Deffiential Geometry')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Training Round')
    plt.legend()

    plt.figure(num2)
    plt.plot(range(len(result['loss_train'])), result['loss_train'], label=r'Noiseless Channel')
    plt.plot(range(len(result['loss_train1'])), result['loss_train1'], label=r'The Proposed Algorithm')
    #    plt.plot(range(len(result['loss_train3'])), result['loss_train3'],label=r'Wuthout RIS')
    #    plt.plot(range(len(result['loss_train2'])), result['loss_train2'],label=r'DC Programming')
    #    plt.plot(range(len(result['loss_train5'])), result['loss_train5'],label=r'Deffiential Geometry')
    plt.ylabel('Training Loss')
    plt.xlabel('Training Round')
    plt.legend()
    plt.ylim([0, 50])
    len1 = len(result['accuracy_test'])
    a = np.zeros([5, len1])
    a[0, :] = result['accuracy_test']
    a[1, :] = result['accuracy_test1']
    #    a[2,:]=result['accuracy_test3']
    #    a[3,:]=result['accuracy_test2']
    #    a[4,:]=result['accuracy_test5']
    return a


if __name__ == '__main__':
    # load the stored running result, average the Monte Carlo trials to compute the average loss/accuracy
    #    M_set=[10,20,30,40,50,60]
    M_set = [40]

    Noiseless = np.zeros([len(M_set), 5])
    Proposed = np.zeros([len(M_set), 5])

    trial = 5
    SNR = 90.0
    for m in range(len(M_set)):
        mm = M_set[m]
        filename = './store/result_trial_{}_M_{}_N_{}_L_{}_\
SNR_{}_Tau_{}_set_{}.npz'.format(trial,
                                 40, 5, mm, SNR, 1, 2)
        a = np.load(filename, allow_pickle=1)
        result_set = a['arr_1']
        result_CNN_set = a['arr_2']
        result_CNN_MB_set = a['arr_3']
        SCA_Gibbs = a['arr_4']

        res_CNN = {}  # batch gradient desent
        res_CNN_MB = {}  # mini-batch gradient desent

        for i in range(trial):
            if i == 0:
                res_CNN = copy.deepcopy(result_CNN_set[0])
                res_CNN_MB = copy.deepcopy(result_CNN_MB_set[0])
            else:
                for item in res_CNN.keys():
                    res_CNN[item] += copy.deepcopy(result_CNN_set[i][item])
                for item in res_CNN.keys():
                    res_CNN_MB[item] += copy.deepcopy(result_CNN_MB_set[i][item])

        for item in res_CNN.keys():
            res_CNN[item] = copy.deepcopy(res_CNN[item] / trial)
            res_CNN_MB[item] = copy.deepcopy(res_CNN_MB[item] / trial)
