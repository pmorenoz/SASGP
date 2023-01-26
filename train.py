
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

from datetime import datetime
from tqdm import trange
import argparse

import numpy as np
import random
import torch
from torch.nn import Parameter
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.gp.parameterized import Parameterized
from torchvision import transforms

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from models.gplvm import SASGP, BayesianSASGP
from utils.command_line_parser import create_parser
from utils.options import train, select_model
from utils.data_loaders import load_dataset
from utils.trainers import train_sas, train_pyro


ROOT_MNIST = "./data/"
ROOT_SAVE = "./data/results/"
ROOT_IMAGES = "./data/results/figures/"
ROOT_EXPERIMENTS = "./exp/"


def main(args):

    # SEED setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gpu = args.gpu
    device = torch.device("cuda:{}".format(gpu))
    print('-'*50)

    ############################################
    # DATASETS
    ############################################
    data_loader, test_loader, data_dimension = load_dataset(args)

    ############################################
    # MODEL LOADING
    ############################################
    model = select_model(args, data_dimension)

    ############################################
    # MODEL TRAINING
    ############################################
    train(model, data_loader, args)


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    main(args)