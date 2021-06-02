# RIS-FL

This is the simulation code package for the following paper:

Hang Liu, Xiaojun Yuan, and Ying-Jun Angela Zhang. "Reconfigurable intelligent surface enabled federated learning: A unified communication-learning design approach," to appear at *IEEE Transactions on Wireless Communications*, 2020. [[ArXiv Version](https://arxiv.org/abs/2011.10282)]

The package, written on Python 3, reproduces the numerical results of the proposed algorithm in the above paper.


## Abstract of Article:

> To exploit massive amounts of data generated at mobile edge networks, federated learning (FL) has been proposed as an attractive substitute for centralized machine learning (ML). By collaboratively training a shared learning model at edge devices, FL avoids direct data transmission and thus overcomes high communication latency and privacy issues as compared to centralized ML. To improve the communication efficiency in FL model aggregation, over-the-air computation has been introduced to support a large number of simultaneous local model uploading by exploiting the inherent superposition property of wireless channels. However, due to the heterogeneity of communication capacities among edge devices, over-the-air FL suffers from the straggler issue in which the device with the weakest channel acts as a bottleneck of the model aggregation performance. This issue can be alleviated by device selection to some extent, but the latter still suffers from a tradeoff between data exploitation and model communication. In this paper, we leverage the reconfigurable intelligent surface (RIS) technology to relieve the straggler issue in over-the-air FL. Specifically, we develop a learning analysis framework to quantitatively characterize the impact of device selection and model aggregation error on the convergence of over-the-air FL. Then, we formulate a unified communication-learning optimization problem to jointly optimize device selection, over-the-air transceiver design, and RIS configuration. Numerical experiments show that the proposed design achieves substantial learning accuracy improvement compared with the state-of-the-art approaches, especially when channel conditions vary dramatically across edge devices.

 
## Dependencies
This package is written on Python 3. It requires the following libraries:
* Python >= 3.5
* torch
* torchvision
* scipy
* CUDA (if GPU is used)

## How to Use
The main file is **main.py**. It can take the following user-input parameters by a parser (also see the function **initial()** in main.py):

To be completed soon:

| Parameter Name  | Meaning| Default Value| Type/Range |
| ---------- | -----------|-----------|-----------|
| M   | total number of devices   |40   |int   |
| N   | total number of receive antennas   |5   |int   |
| L   | total number of RIS elements   |40   |int   |
| nit   | maximum number of iterations for Algorithm 1, I_max   |100   |int   |
| Jmax   | number of sampling iterations for Gibbs sampling   |50   |int   |
| threshold   | threshold epsilon for the early stopping criteria of Algorithm 1   |1e-2   |float   |
| tau   | SCA regularization term for Algorithm 1   |1   |float   |
| trial   | total number of Monte Carlo trials   |50   |int   |
| SNR   | signal-to-noise ratio, P_0/sigma^2_n  |90.0   |float   |
| verbose   | Output no/importatnt/detailed messages   |0   |0,1,2   |
| set   | Which simulation setting (1 or 2) to use; see Section V-A   |2   |1,2   |
| seed   | random seed   |1   |int   |
|  gpu  | GPU index used for learning (if possible)   |1   |int   |
| momentum   | SGD momentum, only used for multiple local updates   |0.9   |int   |
| epochs   | number of training rounds T   |500   |int   |

Here is an example for executing the scripts in Linux terminal is
> python -u main.py --gpu=0 --trial=50 --set=2


## Documentations (Please also see each file for further documentation):

* __main.py__:
    * __initial()__:
    
* __optlib.py__:
* __flow.py__: 
* __Nets.py__: 
    * __CNNMnist()__: Specify the convolutional neural network structure used for learning
* __AirComp.py__:
* __train_script.py__:
* __Monte_Carlo_Averaging.py__:
* __data/__: Store the Fashion-MNIST dataset. When one uses the codes at the first time, it will automatically download the dataset from the Interenet.
* __store/__: Store the learning outputs as npz files
  


## Referencing

If you in any way use this code for research that results in publications, please cite our original article listed above.


