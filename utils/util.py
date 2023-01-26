# Eucliden distance metrics based on gptorch
# code by Steven Atkinson (steven@atkinson.mn)
# Copyright (c) 2018 Steven Atkinson
# ------------------------------------------------
# Copyright (c) 2021 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch.nn import Parameter
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.gp.parameterized import Parameterized
from torchvision import transforms

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from models.gplvm import SASGP, BayesianSASGP

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import glob
import pickle
import datetime

ROOT_MNIST = "./data/"
ROOT_SAVE = "./data/results/"
ROOT_IMAGES = "./data/results/figures/"
ROOT_EXPERIMENTS = "./exp/"

##################################################
#   Save + Load Models
##################################################
def load_exp_dict(dir_name, exp_name=None):
    
    exp_dict_list = []

    if exp_name is None:
        pt_files = glob.glob(ROOT_EXPERIMENTS + str(dir_name) + '/*.pt')
        pt_files.sort()
    else:
        pt_files = glob.glob(ROOT_EXPERIMENTS + str(dir_name) + '/' + exp_name + '*.pt')
        pt_files.sort()

    for pt in pt_files:
        with open(pt, 'rb') as handle:
            exp_dict_list.append(pickle.load(handle))

    return exp_dict_list



def load_model(dir_name, exp_name=None,seed=None):

    if exp_name is None:
        pt_files = glob.glob(ROOT_EXPERIMENTS + str(dir_name) + '/*.pt')
        pt_files.sort()
    else:
        if seed is None:
            pt_files = glob.glob(ROOT_EXPERIMENTS + str(dir_name) + '/' + exp_name + '*.pt')
            pt_files.sort()
        else:
            pt_files = glob.glob(ROOT_EXPERIMENTS + str(dir_name) + '/' + exp_name + '_s' + str(seed) + '*.pt')
            pt_files.sort()


    loss = []
    time = []
    model_params = []
    parser_args = []

    for pt in pt_files:
        with open(pt, 'rb') as handle:
            to_load = pickle.load(handle)
            loss.append(to_load['losses'])
            time.append(to_load['runtimes'])
            model_params.append(to_load['model'])
            parser_args.append(to_load['args'])

    return loss, time, model_params, parser_args


def save_model(loss, time, model, args, filename=None):

    model_params = model.state_dict()
    save_dir = ROOT_EXPERIMENTS + str(args.experiment)
    create_directory(save_dir)
    to_save = {'runtimes': time, 'losses': loss, 'model': model_params, 'args': args}

    if filename is None:
        file_dir = save_dir + '/' + getDateTime() + '_' + str(args.seed) + '.pt'
    else:
        file_dir = save_dir + '/' + filename + '.pt'
    with open(file_dir, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return filename

def select_model(args, data_dimension=784):

    model = None
    if args.model == 'sas':
        # Kernel + Likelihood Setup --
        kernel_ls = torch.tensor(args.kernel_ls, requires_grad=True)
        kernel_a = torch.tensor(args.kernel_a, requires_grad=True)
        sigma = torch.tensor(args.sigma, requires_grad=True)

        kernel = RBF(length_scale=kernel_ls, variance=kernel_a, jitter=args.jitter, input_dim=args.latent_dim, ARD=True)
        likelihood = Gaussian(sigma=sigma, fit_noise=True)

        if args.amortized == 'yes':
            if args.inference == 'gplvm':
                model = SASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                                      latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations)
            elif args.inference == 'bayesian':
                model = BayesianSASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                                      latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations)
            else:
                raise NotImplementedError
        elif args.amortized == 'no':
            if args.inference == 'gplvm':
                model = NonAmortizedSASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                              latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations, dataset=args.dataset)
            elif args.inference == 'bayesian':
                model = NonAmortizedBayesianSASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                                      latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations)
            else:
                raise NotImplementedError

    elif args.model == 'pyro':

        X_init = pyro.param("X", torch.zeros(args.nof_observations, args.latent_dim, dtype=args.dtype))
        kernel = gp.kernels.RBF(input_dim=args.latent_dim,
                                variance=args.kernel_a * torch.tensor(1., dtype=args.dtype, requires_grad=True),
                                lengthscale=args.kernel_ls * torch.ones(2, dtype=args.dtype))

        Xu = Parameter(dist.Normal(torch.zeros([args.num_inducing, args.latent_dim], dtype=args.dtype), 1.0).to_event().sample())
        dummy_y = torch.zeros(args.nof_observations, args.data_dim, dtype=args.dtype, requires_grad=False).t()

        if args.amortized == 'yes':
            if args.inference == 'gplvm':
                raise NotImplementedError
            elif args.inference == 'bayesian':


                gpmodule = gp.models.SparseGPRegression(X_init, dummy_y, kernel, Xu, jitter=args.jitter, noise=torch.tensor(args.sigma, dtype=args.dtype, requires_grad=True))
                model = aBGPLVM(gpmodule)

                if args.obs_inducing:
                    model.base_model.Xu = Parameter(torch.normal(0, 1., size=(args.num_inducing, args.data_dim), requires_grad=True, dtype=args.dtype))
                else:
                    model.base_model.Xu = Parameter(torch.normal(0, 1., size=(args.num_inducing, args.latent_dim), requires_grad=True,dtype=args.dtype))
            else:
                raise NotImplementedError

        elif args.amortized == 'no':
            if args.inference == 'gplvm':
                raise NotImplementedError
            elif args.inference == 'bayesian':
                pass
            else:
                raise NotImplementedError

    return model

def get_data(name, batch_size=32):
    if name == "mnist":
        dataset = MNIST('../', train=True, download=True, transform=transforms.ToTensor())
        mnist_train, mnist_val = random_split(dataset, [55000, 5000])

        train_loader = DataLoader(mnist_train, batch_size=batch_size, pin_memory=True)
        val_loader = DataLoader(mnist_val, batch_size=batch_size, pin_memory=True)

    else:
        raise NotImplementedError

    return train_loader, val_loader

def sample_gp_function(args, data_size, test_size):
    min_x = 0.0
    max_x = 5.0

    kernel_ls = args.true_ls
    kernel_a = args.true_a
    sigma = args.true_sigma

    # Uniform random (train) inputs
    x_train = (max_x - min_x) * torch.rand(data_size, 1) + min_x

    # Uniform random (test) inputs
    x_test = (max_x - min_x) * torch.rand(test_size, 1) + min_x

    x_data = torch.cat((x_train, x_test), 0)

    # Sample from Vanilla RBF kernel
    kernel_ls = torch.tensor(kernel_ls, requires_grad=True)
    kernel_a = torch.tensor(kernel_a, requires_grad=True)
    sigma = torch.tensor(sigma, requires_grad=True)

    kernel = RBF(length_scale=kernel_ls, variance=kernel_a, jitter=args.jitter, input_dim=args.input_dim, ARD=True)
    K = kernel.K(x_data, x_data) + sigma * torch.eye(x_data.shape[0])
    zero_mean = torch.zeros(data_size + test_size)
    normal_dist = MultivariateNormal(loc=zero_mean, covariance_matrix=K)
    y_data = normal_dist.rsample()

    y_train = y_data[:data_size]
    y_test = y_data[data_size:]

    return x_train, y_train, x_test, y_test

def true_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        5*torch.cos(7*np.pi*x + 2.4*np.pi)
    return y

def smooth_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi)
    return y

def smooth_function_bias(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        3.0*x - 7.5
    return y

def squared_distance(x1, x2=None):
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.
    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        return squared_distance(x1, x1)

    x1s = x1.pow(2).sum(1, keepdim=True)
    x2s = x2.pow(2).sum(1, keepdim=True)

    r2 = x1s + x2s.t() -2.0 * x1 @ x2.t()

    # Prevent negative squared distances using torch.clamp
    # NOTE: Clamping is for numerics.
    # This use of .detach() is to avoid breaking the gradient flow.
    return r2 - (torch.clamp(r2, max=0.0)).detach()

class DataGPLVM(Dataset):
    def __init__(self, y):
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.y[item]

def create_directory(pathname, verbose=False):
    import os
    try:
        os.mkdir(pathname, 0o755)
    except OSError:
        if verbose:
            print("Creation of the directory %s failed" % pathname)
    else:
        if verbose:
            print("Successfully created the directory %s" % pathname)
    return

def getDate():
    now = datetime.datetime.now()
    dt = now.isoformat()
    date = dt[:10].replace(':', '-')
    return date

def getTime():
    now = datetime.datetime.now()
    dt = now.isoformat()
    tid = dt[11:19].replace(':', '-')
    return tid

def getDateTime():
    tid = getTime()
    dato = getDate()
    return dato + 'T' + tid

class DataGP(Dataset):
    def __init__(self, x):
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x)
        #if not torch.is_tensor(y):
        #    self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    #def __getitem__(self, item):
    #    return self.x[item], self.y[item]

    def __getitem__(self, item):
        return self.x[item]


def get_git_version():
    import subprocess
    gitrevision = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    return gitrevision
    

def create_directory(pathname,verbose=False):
    import os
    try:
        os.mkdir(pathname, 0o755)
    except OSError:
        if verbose:
            print ("Creation of the directory %s failed" % pathname)
    else:
        if verbose:
            print ("Successfully created the directory %s" % pathname)
    return
