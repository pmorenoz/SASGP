
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)


import numpy as np
from scipy import sparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, MNIST, FashionMNIST, CIFAR10
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.model_selection import train_test_split

from kernels.rbf import RBF
from utils.util import DataGP, smooth_function_bias, smooth_function, sample_gp_function

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn import preprocessing

ROOT_REUTERS = "./data/reuters/"



def all_data(data_loader):
    all_data_loader = DataLoader(dataset=data_loader.dataset, batch_size=len(data_loader.dataset), pin_memory=True, shuffle=True)
    return all_data_loader

def load_dataset(args):
    label_scaler = None  # for regression datasets we return label scaler to be able to untransform data
    if args.dataset == 'mnist':
        ## MNIST // TRAIN=60.000, TEST=10.000
        transform = transforms.ToTensor()
        mnist_train = MNIST(root='./data/', train=True, download=True, transform=transform)
        mnist_test = MNIST(root='./data/', train=False, download=True, transform=transform)

        if args.nof_observations < 60000:
            indices = list(range(args.nof_observations))
            mnist_train = torch.utils.data.Subset(mnist_train, indices)
        elif args.nof_observations > 60000:
            raise AssertionError('NOF Observations larger than dataset size')

        # Data Loaders / Train & Test
        data_loader = DataLoader(mnist_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        if args.nof_test < len(mnist_test):
            test_loader = DataLoader(mnist_test, batch_size=args.nof_test, pin_memory=True, shuffle=True)
        else:
            test_loader = DataLoader(mnist_test, batch_size=len(mnist_test), pin_memory=True, shuffle=True)
        data_dimension = 784  # for folding later

    elif args.dataset == 'fmnist':
        ## FashionMNIST // TRAIN=60.000, TEST=10.000
        transform = transforms.ToTensor()
        fmnist_train = FashionMNIST(root='./data/', train=True, download=True, transform=transform)
        fmnist_test = FashionMNIST(root='./data/', train=False, download=True, transform=transform)

        if args.nof_observations < 60000:
            indices = list(range(args.nof_observations))
            fmnist_train = torch.utils.data.Subset(fmnist_train, indices)
        elif args.nof_observations > 60000:
            raise AssertionError('NOF Observations larger than dataset size')

        # Data Loaders / Train & Test
        data_loader = DataLoader(fmnist_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        if args.nof_test < len(fmnist_test):
            test_loader = DataLoader(fmnist_test, batch_size=args.nof_test, pin_memory=True, shuffle=True)
        else:
            test_loader = DataLoader(fmnist_test, batch_size=len(fmnist_test), pin_memory=True, shuffle=True)
        data_dimension = 784  # for folding later

    elif args.dataset == 'cifar':
        ## CIFAR10 // TRAIN=50.000, TEST=10.000
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # mean+std normalization
        cifar10_train = CIFAR10(root='./data/', train=True, download=False, transform=transform)
        cifar10_test = CIFAR10(root='./data/', train=False, download=False, transform=transform)

        if args.nof_observations < 50000:
            indices = list(range(args.nof_observations))
            cifar10_train = torch.utils.data.Subset(cifar10_train, indices)
        elif args.nof_observations > 50000:
            raise AssertionError('NOF Observations larger than dataset size')

        # Data Loaders / Train & Test
        data_loader = DataLoader(cifar10_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        if args.nof_test < len(cifar10_test):
            test_loader = DataLoader(cifar10_test, batch_size=args.nof_test, pin_memory=True, shuffle=True)
        else:
            test_loader = DataLoader(cifar10_test, batch_size=len(cifar10_test), pin_memory=True, shuffle=True)
        data_dimension = 3072  # for folding later // CIFAR10 is 32x32x3

    elif args.dataset == 'reuters':
        data_dimension = 2000  # for folding later
        data_size = 500000
        aug_factor = 1.0  # augmentation factor
        reuters_train = torch.load(ROOT_REUTERS + 'reuters_train.pt')
        reuters_test = torch.load(ROOT_REUTERS + 'reuters_test.pt')

        data_loader = DataLoader(aug_factor * reuters_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        test_loader = DataLoader(reuters_test, batch_size=1, pin_memory=True, shuffle=True)

    elif args.dataset == 'airline':
        file = './data/us_flight/us_flight_data_year08.npz'
        data = np.load(file)

        x_data = data['x_data']
        y_data = data['y_data']

        p, _ = x_data.shape
        x_min = np.tile(np.min(x_data, 0)[None, :], (p, 1))
        x_max = np.tile(np.max(x_data, 0)[None, :], (p, 1))

        x = (x_data - x_min) / (x_max - x_min)
        y = np.log(y_data + 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        airline_train = DataGP(x=x_train)
        airline_test = DataGP(x=x_test)

        data_loader = DataLoader(airline_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        test_loader = DataLoader(airline_test, batch_size=1, pin_memory=True, shuffle=True)
        data_dimension = 6
    else:
        raise NotImplementedError

    return data_loader, test_loader, data_dimension

def select_dataset(args):
    if args.dataset == 'mnist':
        data_loader,test_loader = mnist(args)
    elif args.dataset == 'fashion':
        data_loader,test_loader = fashion(args)
    else:
        print("Only mnist and fashion mnist implemented so far")
        raise TypeError("Wrong dataset choice. Use mnist or fashion")

    return data_loader,test_loader

class indexedFashionMNIST(FashionMNIST):
    def __init__(self,dtype,root: str,train: bool = True,transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,download: bool = False) -> None:

        super().__init__('./data/', train=train, download=download, transform=transform)
        self.dtype = dtype

    # Subclass of MNIST for returning the index
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
   x         tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.to(self.dtype), target, index

class indexedMNIST(MNIST):
    def __init__(self,dtype,root: str,train: bool = True,transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,download: bool = False) -> None:

        super().__init__('./data/', train=train, download=download, transform=transform)
        self.dtype = dtype

    # Subclass of MNIST for returning the index
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.to(self.dtype), target, index


class ToyRegression(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = y
        self.x = x
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class BikeUCI(Dataset): # N = 17379
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)


    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item


class ElevatorsUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item


class Kin40kUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class ProteinUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class GasUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class SkillcraftUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class SmlUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class ParkinsonsUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class PumadynUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class KeggUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class SliceUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class KegguUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class SongUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class BuzzUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

class ElectricUCI(Dataset):
    def __init__(self, x, y, transform=None):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.input_dim = self.x.size(dim=1)

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, idx):
        y_item = self.y[idx]
        x_item = self.x[idx]

        return x_item, y_item

def reuters():
    from sklearn.datasets import fetch_rcv1
    import scipy
    rcv1 = fetch_rcv1(shuffle = True)

    D = rcv1.data.tocsc()
    D_sum_AR = np.asarray(D.sum(axis=0)).flatten()
    cols_in = D_sum_AR.argsort()[-2000:]

    cols = []
    for i in cols_in:
        cols.append(D.getcol(i))

    data = scipy.sparse.hstack(cols).tocsr()
    y = torch.tensor(data)

    print("Not implemented")
    quit()
    return

def mnist(args):
    train_set = indexedMNIST(args.dtype,'./data/', train=True, download=True, transform=transforms.ToTensor())
    test_set = indexedMNIST(args.dtype,'./data/', train=False, download=True, transform=transforms.ToTensor())
    if args.nof_observations != 60000:

        # select N random numbers between 0 and 60000 without replacement
        indices = list(range(args.nof_observations))#random.sample(range(60000), args.nof_observations)#list(range(0, len(trainset), 2))
        train_set = torch.utils.data.Subset(train_set, indices)

    #mnist_train, mnist_val = random_split(dataset, [args.nof_observations, 60000 - args.nof_observations])

    data_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,drop_last=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=True)
    #data_dimension = 784  # for folding later
    args.data_dim = 784
    return data_loader,test_loader

def fashion(args):
    train_set = indexedFashionMNIST(args.dtype,'./data/', train=True, download=True, transform=transforms.ToTensor())
    test_set = indexedFashionMNIST(args.dtype,'./data/', train=False, download=True, transform=transforms.ToTensor())

    if args.nof_observations != 60000:

        # select N random numbers between 0 and 60000 without replacement
        indices = list(range(args.nof_observations))#random.sample(range(60000), args.nof_observations)#list(range(0, len(trainset), 2))
        train_set = torch.utils.data.Subset(train_set, indices)

    #mnist_train, mnist_val = random_split(dataset, [args.nof_observations, 60000 - args.nof_observations])

    data_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=True)
    #data_dimension = 784  # for folding later
    args.data_dim = 784
    return data_loader,test_loader

def freyfaces():
    import scipy.io as sio
    face_data = sio.loadmat('data/frey_rawface')
    data = torch.tensor(face_data['ff'])
    return data,None

def oilflow():
    import urllib.request
    import tarfile

    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    urllib.request.urlretrieve(url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')

    Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
    labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
    labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)
    #unique_labels = [str(int(i.item())) for i in labels]
    return Y.t(),labels#, unique_labels

def mnist_5000(subset_size = 5000):
    mnist_data = torchvision.datasets.MNIST('data/', train=True,download=True)
    data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=4,
                                          shuffle=True)
    N_original, img_width, img_height = mnist_data.data.shape
    y = mnist_data.data.reshape([N_original, img_width*img_height])
    labels = mnist_data.targets


    indices = torch.randperm(len(y),generator=torch.Generator().manual_seed(42))[:subset_size]
    data = y[indices,:]
    labels = labels[indices]
    y = data.t()

    #unique_labels = [str(int(i.item())) for i in labels]
    return y,labels#, unique_labels
    #return y, labels

def mnist_full(subset_size = 60000):
    mnist_data = torchvision.datasets.MNIST('data/', train=True,download=True)
    data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=4,
                                          shuffle=True)
    N_original, img_width, img_height = mnist_data.data.shape
    y = mnist_data.data.reshape([N_original, img_width*img_height])
    labels = mnist_data.targets


    indices = torch.randperm(len(y),generator=torch.Generator().manual_seed(42))[:subset_size]
    data = y[indices,:]
    labels = labels[indices]
    y = data.t()
    #unique_labels = [str(int(i.item())) for i in labels]
    return y,labels#, unique_labels
    #return y, labels

# def singlecell():
#     URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
#
#     df = pd.read_csv(URL, index_col=0)
#     data = torch.tensor(df.values, dtype=torch.get_default_dtype())
#     # we need to transpose data to correct its shape
#     y = data.t()
#
#     labels = df.index#.unique()
#     #unique_labels = df.index.unique()
#     return y,labels#,unique_labels


def sample_gp_function(args, data_size, test_size, data_dim=1):
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

    kernel = RBF(length_scale=kernel_ls, variance=kernel_a, jitter=args.jitter, input_dim=data_dim, ARD=True)
    K = kernel.K(x_data, x_data) + sigma * torch.eye(x_data.shape[0])
    zero_mean = torch.zeros(data_size + test_size)
    normal_dist = MultivariateNormal(loc=zero_mean, covariance_matrix=K)
    y_data = normal_dist.rsample()

    y_train = y_data[:data_size]
    y_test = y_data[data_size:]

    x_train = x_train.detach()
    y_train = y_train.detach()
    x_test = x_test.detach()
    y_test = y_test.detach()

    return x_train, y_train, x_test, y_test