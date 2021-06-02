# -*- coding: utf-8 -*-
import time
import copy
import numpy as np

np.set_printoptions(precision=6, threshold=1e3)
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import argparse
import torch
import flow
from optlib import Gibbs


def initial():
    libopt = argparse.ArgumentParser()
    libopt.add_argument('--M', type=int, default=40, help='total # of devices')
    libopt.add_argument('--N', type=int, default=5, help='# of BS antennas')
    libopt.add_argument('--L', type=int, default=40, help='RIS Size')

    # optimization parameters
    libopt.add_argument('--nit', type=int, default=100, help='I_max,# of maximum inner SCA loops')
    libopt.add_argument('--Jmax', type=int, default=50, help='# of maximum outer Gibbs loops')
    libopt.add_argument('--threshold', type=float, default=1e-2, help='epsilon,SCA early stopping criteria')
    libopt.add_argument('--tau', type=float, default=1, help=r'\tau, the SCA regularization term')

    # simulation parameters
    libopt.add_argument('--trial', type=int, default=50, help='# of Monte Carlo Trials')
    libopt.add_argument('--SNR', type=float, default=90.0, help='noise variance/0.1W(transmit budget P_0) in dB')
    libopt.add_argument('--verbose', type=int, default=0, help
    =r'whether output or not, =0 for no;\
                        =1 for important messages only; \
                        =2 for detailed messages')
    libopt.add_argument('--set', type=int, default=2, help=r'=1 if concentrated devices+ euqal dataset;\
                        =2 if two clusters + unequal dataset')
    libopt.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # learning parameters
    libopt.add_argument('--gpu', type=int, default=1, help=r'Use Which Gpu')
    libopt.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    libopt.add_argument('--local_bs', type=int, default=0, help="Local Bath size B,0 for Batch Gradient Descent")
    libopt.add_argument('--lr', type=float, default=0.01, help="learning rate,lambda")
    libopt.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum, used only for multiple local updates")
    libopt.add_argument('--epochs', type=int, default=500, help="rounds of training T")
    args = libopt.parse_args()
    return args


if __name__ == '__main__':

    # initialize system parameters
    libopt = initial()
    np.random.seed(libopt.seed)
    print(libopt)
    libopt.transmitpower = 0.1  # P_0
    # path of the file to store
    filename = './store/result_trial_{}_M_{}_N_{}_L_{}_SNR_{}_Tau_{}_set_{}.npz'.format(libopt.trial,
                                                                                        libopt.M,
                                                                                        libopt.N, libopt.L,
                                                                                        libopt.SNR, libopt.tau,
                                                                                        libopt.set)
    print('Will save result to: \n {}'.format(filename))

    libopt.alpha_direct = 3.76  # User-BS Path loss exponent
    fc = 915 * 10 ** 6  # carrier frequency, wavelength lambda_c=3.0*10**8/fc
    BS_Gain = 10 ** (5.0 / 10)  # BS antenna gain
    RIS_Gain = 10 ** (5.0 / 10)  # RIS antenna gain
    User_Gain = 10 ** (0.0 / 10)  # User antenna gain
    d_RIS = 1.0 / 10  # length of one RIS element/wavelength

    libopt.BS = np.array([-50, 0, 10])  # location of the BS/PS
    libopt.RIS = np.array([0, 0, 10])  # location of the RIS

    x0 = np.ones([libopt.M], dtype=int)  # initial the device selection such that all devices are selected

    # arrays to store the results
    SCA_Gibbs = np.ones([libopt.Jmax + 1, libopt.trial]) * np.nan
    #    DC_NORIS_set=np.ones([libopt.trial,])*np.nan
    #    DC_NODS_set=np.ones([libopt.trial,])*np.nan
    #    Alt_Gibbs=np.ones([libopt.Jmax+1,libopt.trial])*np.nan
    #    DG_NORIS=np.ones([libopt.trial,])*np.nan

    # list to store the learning training loss/test accuracy/test loss
    result_set = []
    result_CNN_set = []
    result_CNN_MB_set = []

    # select the GPU
    libopt.device = torch.device(
        'cuda:{}'.format(libopt.gpu) if torch.cuda.is_available() and libopt.gpu != -1 else 'cpu')
    print('The selected GPU index is {}'.format(libopt.device))

    sigma_n = np.power(10, -libopt.SNR / 10)  # noise power=P_0/SNR=0.1/SNR

    # to facilitate numerical optimization, we simultaneously scale up the channel coefficents and the noise
    # without loss of generality.
    # To this end, the noise variance is multipled by 1e10 and the channel coefficents are multipled by 1e5(their power scale 1e10)
    # By doing so, their values are guaranteed to be significant to allieviate error propogation in optimization.
    ref = (1e-10) ** 0.5
    libopt.sigma = sigma_n / ref ** 2  # effective noise power after scaling

    for i in range(libopt.trial):
        print('This is the {0}-th trial'.format(i))

        # half devices have x\in[-20,0]
        libopt.dx1 = np.random.rand(int(np.round(libopt.M / 2))) * 20 - 20
        if libopt.set == 1:
            # Setting 1:

            # For M=40, K=750
            libopt.K = np.ones(libopt.M, dtype=int) * int(30000.0 / libopt.M)
            print(sum(libopt.K))
            # the other half devices also have x\in[-20,0]
            libopt.dx2 = np.random.rand(int(libopt.M - np.round(libopt.M / 2))) * 20 - 20  # [100,100+range]
        else:
            # Setting 2:

            # Half (random selected) devices have Uniform[1000,2000] data, the other half have Uniform[100,200] data
            libopt.K = np.random.randint(1000, high=2001, size=(int(libopt.M)))
            lessuser_size = int(libopt.M / 2)
            libopt.K2 = np.random.randint(100, high=201, size=(lessuser_size))
            libopt.lessuser = np.random.choice(libopt.M, size=lessuser_size, replace=False)
            libopt.K[libopt.lessuser] = libopt.K2
            print(sum(libopt.K))

            # the other half devices have x\in[100,120]
            libopt.dx2 = np.random.rand(int(libopt.M - np.round(libopt.M / 2))) * 20 + 100

        # concatenate all the x locations
        libopt.dx = np.concatenate((libopt.dx1, libopt.dx2))
        # y\in[-10,10]
        libopt.dy = np.random.rand(libopt.M) * 20 - 10

        # distance between User to RIS
        libopt.d_UR = ((libopt.dx - libopt.RIS[0]) ** 2 + (libopt.dy - libopt.RIS[1]) ** 2 + libopt.RIS[2] ** 2
                       ) ** 0.5

        # distance between RIS to BS/PS
        libopt.d_RB = np.linalg.norm(libopt.BS - libopt.RIS)
        # distance of direct User-BS channel
        libopt.d_direct = ((libopt.dx - libopt.BS[0]) ** 2 + (libopt.dy - libopt.BS[1]) ** 2 + libopt.BS[2] ** 2
                           ) ** 0.5

        # Path loss of direct channel
        libopt.PL_direct = BS_Gain * User_Gain * (3 * 10 ** 8 / fc / 4 / np.pi / libopt.d_direct) ** libopt.alpha_direct

        # Path loss of RIS channel
        libopt.PL_RIS = BS_Gain * User_Gain * RIS_Gain \
                        * libopt.L ** 2 * (d_RIS * 3 * 10 ** 8 / fc) ** 2 / 64 / np.pi ** 3 \
                        * (3 * 10 ** 8 / fc / libopt.d_UR / libopt.d_RB) ** 2

        # channels coefficents (after scaling)
        h_d = (np.random.randn(libopt.N, libopt.M) + 1j * np.random.randn(libopt.N, libopt.M)) / 2 ** 0.5
        h_d = h_d @ np.diag(libopt.PL_direct ** 0.5) / ref
        H_RB = (np.random.randn(libopt.N, libopt.L) + 1j * np.random.randn(libopt.N, libopt.L)) / 2 ** 0.5
        h_UR = (np.random.randn(libopt.L, libopt.M) + 1j * np.random.randn(libopt.L, libopt.M)) / 2 ** 0.5
        h_UR = h_UR @ np.diag(libopt.PL_RIS ** 0.5) / ref

        # Cascaded RIS channel
        G = np.zeros([libopt.N, libopt.L, libopt.M], dtype=complex)
        for j in range(libopt.M):
            G[:, :, j] = H_RB @ np.diag(h_UR[:, j])

        # initial x as x0
        x = x0

        Noiseless = 1  # =1 run error-free benchmark;=0 run the proposed algorithm
        Proposed = 1
        if Proposed:
            start = time.time()
            print('Running the proposed algorithm')
            [x_store, obj_new, f_store, theta_store] = Gibbs(libopt, h_d, G, x, True, True)
            end = time.time()
            print("Running time: {} seconds".format(end - start))
            SCA_Gibbs[:, i] = obj_new
        else:
            x_store = 0
            obj_new = 0
            f_store = 0
            theta_store = 0

        # dictionary used to store the optimization result to pass to the learning script
        dic = {}
        dic['x_store'] = copy.deepcopy(x_store)
        dic['f_store'] = copy.deepcopy(f_store)
        dic['theta_store'] = copy.deepcopy(theta_store)

        # Running batch gradient desent
        start = time.time()
        libopt.lr = 0.01
        libopt.epochs = 500
        libopt.local_bs = 0
        print('lr{} batch{} ep{}'.format(libopt.lr, libopt.local_bs, libopt.epochs))
        result, _ = flow.learning_flow(libopt, Noiseless, Proposed,
                                       h_d, G, dic)
        end = time.time()
        print("Running time: {} seconds".format(end - start))
        result_CNN_set.append(result)

        # Running mini-batch gradient desent
        start = time.time()
        libopt.lr = 0.005
        libopt.epochs = 100
        libopt.local_bs = 128
        result, _ = flow.learning_flow(libopt, Noiseless, Proposed,
                                       h_d, G, dic)
        end = time.time()
        print("Running time: {} seconds".format(end - start))
        result_CNN_MB_set.append(result)

    np.savez(filename, vars(libopt), result_set, result_CNN_set,
             result_CNN_MB_set, SCA_Gibbs)
