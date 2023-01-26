
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)


import argparse
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_parser():
    parser = argparse.ArgumentParser(description='SAS Gaussian Processes')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')

    # File
    date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    parser.add_argument('--filename', default="model_" + date_time, type=str, help='name of the saved model file')

    # Dataset
    parser.add_argument('--dataset', '-data', type=str, default='mnist', help='which dataset / options [mnist, fmnist, cifar]')
    parser.add_argument('--nof_observations', '-n', type=int, default=40000, help='size of training data')
    parser.add_argument('--nof_test', '-n_test', type=int, default=1000, help='size of test data')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='batch size')

    # Model
    parser.add_argument('--model', '-model', default='sas', type=str, help='model - choose: [sas, baseline]')
    parser.add_argument('--amortized', default='yes', type=str, help='amortization / [yes, no]')
    parser.add_argument('--inference', '-inf', default='gplvm', type=str, help='model to run / [gplvm, bayesian]')
    parser.add_argument('--latent_dim', default=2, type=int, help='dimensionality of the latent space')
    parser.add_argument('--num_active', '-a',  default=100, type=int, help='size for the active set of the SASGP')
    parser.add_argument('--dtype', type=int, default=64, help='which precision / [32, 64]')     

    # Kernel initialisation   
    parser.add_argument('--sigma', '-sn', default=0.5, type=float, help='likelihood noise')
    parser.add_argument('--kernel_ls', '-ls', default=0.5, type=float, help='lengthscale of the RBF kernel')
    parser.add_argument('--kernel_a', '-am', default=0.5, type=float, help='amplitude of the RBF kernel')
    parser.add_argument('--ard', '-ard', default=True, type=str2bool, help='ARD kernel? [yes, no, true, false]')

    # Specific for the toy regression experiment
    parser.add_argument('--true_sigma', default=0.2, type=float, help='true likelihood noise')
    parser.add_argument('--true_ls', default=0.5, type=float, help='true lengthscale of the RBF kernel')
    parser.add_argument('--true_a', default=1.0, type=float, help='true amplitude of the RBF kernel')

    # Optimization
    parser.add_argument('--jitter', type=float, default=1e-5, help='jitter')
    parser.add_argument('--seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument('--num_epochs', '-e', type=int, default=100, help='number of epochs in the optimization')
    parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate for the model')

    # Extra
    parser.add_argument('--comment', type=str, default='', help='brief explain explanation')
    parser.add_argument('--experiment', type=str, default='', help='experiment_number')  # folder for saving experiments / revision
    parser.add_argument('--plot', type=str2bool, default='no', help='plot training figure? -- [yes, no, false, true]')
    parser.add_argument('--save', '-s', type=str2bool, default=False, help='do we save results? [true, false]')

    # Results for figures and tables
    parser.add_argument('--marginal_lik', '-ev', default='no', type=str2bool, help='marginal likelihood computation? [true false]')
    parser.add_argument('--metrics', '-met', type=str2bool, default=False, help='metrics? [true false]')

    return parser